# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Dataset class encapsulates the data loading"""
import os
from pathlib import Path

import h5py
import tensorflow as tf
import numpy as np
import xarray as xr
import horovod.tensorflow as hvd
from scipy import interpolate

from sciml_bench.core.data_loader import DataLoader
from sciml_bench.slstr_cloud.constants import PATCH_SIZE, PATCHES_PER_IMAGE

class SLSTRDataLoader(DataLoader):

    def __init__(self, data_dir, seed=0):
        self._data_dir = data_dir

        self._train_images = Path(self._data_dir).joinpath('train').glob('S3A*.hdf')
        self._train_images = list(map(str, self._train_images))

        self._test_images = Path(self._data_dir).joinpath('test').glob('S3A*.hdf')
        self._test_images = list(map(str, self._test_images))

        assert len(self._train_images) > 0, "Could not find any training data!"
        assert len(self._test_images) > 0, "Could not find any test data!"

        self._seed = seed

    @property
    def dimensions(self):
        return (PATCH_SIZE, PATCH_SIZE, 9)

    @property
    def train_size(self):
        return len(self._train_images) * PATCHES_PER_IMAGE

    @property
    def test_size(self):
        return len(self._test_images) * PATCHES_PER_IMAGE

    def _load_data(self, path):
        with h5py.File(path, 'r') as handle:
            refs = handle['refs'][:]
            bts = handle['bts'][:]
            msk = handle['bayes'][:]

        refs = np.expand_dims(refs, 0)
        bts = np.expand_dims(bts, 0)
        msk = np.expand_dims(msk, 0)

        return refs, bts, msk

    def _parse_file(self, path):
        path = path.decode()

        rads, bts, mask = self._load_data(path)

        bt_nan_mask = tf.math.is_finite(bts)
        fill_value = tf.math.reduce_min(tf.boolean_mask(bts, bt_nan_mask))
        bts = tf.where(bt_nan_mask, bts, fill_value)
        bts = tf.cast(bts, tf.float32)

        # Mean - Std norm
        bts = (bts - 270.0) / 22.0

        rads_nan_mask = tf.math.is_finite(rads)
        fill_value = tf.reduce_min(tf.boolean_mask(rads, rads_nan_mask))
        fill_value = tf.where(tf.math.is_finite(fill_value), fill_value, 0)
        rads = tf.where(rads_nan_mask, rads, fill_value)
        # Mean - Std norm, already in range 0-1, convert to (-1, 1) range.
        rads = rads - 0.5

        mask = tf.cast(mask, tf.float32)
        mask = tf.where(tf.math.is_finite(mask), mask, 0.)
        mask = tf.where(tf.math.greater(mask, 0), 1, 0)

        bts = self._transform_image(bts, PATCH_SIZE)
        rads = self._transform_image(rads, PATCH_SIZE)
        mask = self._transform_image(mask, PATCH_SIZE)

        channels = tf.concat([rads, bts], axis=-1)
        msk = tf.concat([mask, 1-mask], axis=-1)

        yield (channels, msk)

    def _transform_image(self, img, patch_size):
        b, h, w, c  = img.shape
        dims = [1, patch_size, patch_size, 1]
        img = tf.image.central_crop(img, .9)
        img = tf.image.extract_patches(img, dims, dims, [
                                       1, 1, 1, 1], padding='VALID')
        n, nx, ny, np = img.shape
        img = tf.reshape(img, (n*nx*ny, patch_size, patch_size, c))
        return img

    def _generator(self, path):
        types = (tf.float32, tf.float32)
        shapes = (tf.TensorShape([None, PATCH_SIZE, PATCH_SIZE, 9]),
                  tf.TensorShape([None, PATCH_SIZE, PATCH_SIZE, 2]))
        dataset = tf.data.Dataset.from_generator(self._parse_file,
                                                 output_types=types,
                                                 output_shapes=shapes,
                                                 args=(path, ))
        return dataset

    def train_fn(self, batch_size=8):
        """Input function for training"""
        dataset = tf.data.Dataset.from_tensor_slices(self._train_images)
        dataset = dataset.shard(hvd.size(), hvd.rank())
        dataset = dataset.shuffle(500)
        dataset = dataset.interleave(self._generator, cycle_length=16, num_parallel_calls=None)
        dataset = dataset.unbatch()
        dataset = dataset.shuffle(batch_size * 3)
        dataset = dataset.prefetch(batch_size * 3)
        dataset = dataset.batch(batch_size)
        return dataset.repeat()

    def test_fn(self, batch_size=8):
        dataset = tf.data.Dataset.from_tensor_slices(self._test_images)
        dataset = dataset.shard(hvd.size(), hvd.rank())
        dataset = dataset.interleave(self._generator, cycle_length=16, num_parallel_calls=None)
        dataset = dataset.unbatch()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(batch_size)
        return dataset.repeat()

class Sentinel3Dataset(DataLoader):
    """Load, separate and prepare the data for training and prediction"""

    def __init__(self, data_dir, seed=0):
        self._data_dir = data_dir

        self._train_images = Path(self._data_dir).joinpath('train').glob('S3A*')
        self._train_images = list(map(str, self._train_images))

        self._test_images = Path(self._data_dir).joinpath('test').glob('S3A*')
        self._test_images = list(map(str, self._test_images))

        self._seed = seed

    @property
    def dimensions(self):
        return (PATCH_SIZE, PATCH_SIZE, 9)

    @property
    def train_size(self):
        return len(self._train_images) * PATCHES_PER_IMAGE

    @property
    def test_size(self):
        return len(self._test_images) * PATCHES_PER_IMAGE

    def _load_data(self, path):
        loader = ImageLoader(path)
        rads = loader.load_radiances()
        bts = loader.load_bts()
        flags = loader.load_flags()

        rads = rads.to_array().values
        bts = bts.to_array().values
        mask = flags.bayes_in.values

        rads = tf.convert_to_tensor(rads)
        bts = tf.convert_to_tensor(bts)
        mask = tf.convert_to_tensor(mask)

        return rads, bts, mask

    def _parse_file(self, path):
        path = path.decode()

        rads, bts, mask = self._load_data(path)

        bt_nan_mask = tf.math.is_finite(bts)
        fill_value = tf.math.reduce_min(tf.boolean_mask(bts, bt_nan_mask))
        bts = tf.where(bt_nan_mask, bts, fill_value)

        # Mean - Std norm
        bts = (bts - 270.0) / 22.0
        bts = tf.transpose(bts, [1, 2, 0])
        bts = tf.expand_dims(bts, axis=0)

        rads_nan_mask = tf.math.is_finite(rads)
        fill_value = tf.reduce_min(tf.boolean_mask(rads, rads_nan_mask))
        fill_value = tf.where(tf.math.is_finite(fill_value), fill_value, 0)
        rads = tf.where(rads_nan_mask, rads, fill_value)
        # Mean - Std norm, already in range 0-1, convert to (-1, 1) range.
        rads = rads - 0.5
        rads = tf.transpose(rads, [1, 2, 0])
        rads = tf.expand_dims(rads, axis=0)

        mask = tf.cast(mask, tf.float32)
        mask = tf.where(tf.math.is_finite(mask), mask, 0.)
        mask = tf.where(tf.math.greater(mask, 0), 1, 0)
        mask = tf.expand_dims(mask, axis=0)
        mask = tf.expand_dims(mask, axis=-1)

        _, h, w, _ = bts.shape
        rads = tf.image.resize(rads, (h, w))

        bts = self._transform_image(bts, PATCH_SIZE)
        rads = self._transform_image(rads, PATCH_SIZE)
        mask = self._transform_image(mask, PATCH_SIZE)

        channels = tf.concat([rads, bts], axis=-1)
        msk = tf.concat([mask, 1-mask], axis=-1)

        yield (channels, msk)

    def _transform_image(self, img, patch_size):
        b, h, w, c  = img.shape
        dims = [1, patch_size, patch_size, 1]
        img = tf.image.central_crop(img, .9)
        img = tf.image.extract_patches(img, dims, dims, [
                                       1, 1, 1, 1], padding='VALID')
        n, nx, ny, np = img.shape
        img = tf.reshape(img, (n*nx*ny, patch_size, patch_size, c))
        return img


    def _generator(self, path):
        types = (tf.float32, tf.float32)
        shapes = (tf.TensorShape([None, PATCH_SIZE, PATCH_SIZE, 9]),
                  tf.TensorShape([None, PATCH_SIZE, PATCH_SIZE, 2]))
        dataset = tf.data.Dataset.from_generator(self._parse_file,
                                                 output_types=types,
                                                 output_shapes=shapes,
                                                 args=(path, ))
        return dataset

    def train_fn(self, batch_size=8):
        """Input function for training"""
        dataset = tf.data.Dataset.from_tensor_slices(self._train_images)
        dataset = dataset.shuffle(1000)
        dataset = dataset.interleave(self._generator, cycle_length=16, num_parallel_calls=16)
        dataset = dataset.unbatch()
        dataset = dataset.prefetch(batch_size*2)
        dataset = dataset.shuffle(batch_size*2)
        dataset = dataset.batch(batch_size)
        dataset = dataset.cache()
        dataset = dataset.repeat()
        return dataset

    def test_fn(self, batch_size=8):
        dataset = tf.data.Dataset.from_tensor_slices(self._test_images)
        dataset = dataset.interleave(self._generator, cycle_length=2, num_parallel_calls=2)
        dataset = dataset.unbatch()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(batch_size)
        return dataset


class ImageLoader:

    def __init__(self, path, engine='h5netcdf'):
        self.path = path
        self._engine = engine

    def load_radiances(self, view='an'):
        rads = [
            self.load_radiance_channel(
                self.path,
                i,
                view) for i in range(
                1,
                7)]
        rads = xr.merge(rads)
        return rads

    def load_irradiances(self, view='an'):
        irradiances = {}
        for i in range(1, 7):
            name = 'S{}_solar_irradiance_{}'.format(i, view)
            file_name = os.path.join(
                self.path, 'S{}_quality_{}.nc'.format(
                    i, view))
            irradiance = xr.open_dataset(file_name, engine=self._engine)[name][:].data[0]
            irradiances[name] = irradiance
        return irradiances

    def load_reflectance(self, view='an'):
        refs = [
            self.load_reflectance_channel(
                self.path,
                i,
                view) for i in range(
                1,
                7)]
        refs = xr.merge(refs)
        return refs

    def load_reflectance_channel(self, path, channel_num, view='an'):
        rads = self.load_radiance_channel(path, channel_num, view)
        names = {name: name.replace('radiance', 'reflectance')
                 for name in rads}
        rads = rads.rename(names)
        irradiances = self.load_irradiances(view)
        geometry = self.load_geometry()

        solar_zenith = geometry.solar_zenith_tn[:]
        solar_zenith = np.nan_to_num(solar_zenith, 0.0)

        x = (np.arange(rads.dims['columns']) - rads.track_offset) * float(
            rads.resolution.split()[1]) / float(geometry.resolution.split()[1]) + geometry.track_offset
        y = np.arange(rads.dims['rows']) * float(rads.resolution.split()[2]) / \
            float(geometry.resolution.split()[2]) + geometry.start_offset

        f = interpolate.RectBivariateSpline(np.arange(
            geometry.dims['rows']), np.arange(geometry.dims['columns']), solar_zenith)
        solar_zenith = f(y, x)

        DTOR = 0.017453292
        mu0 = np.where(solar_zenith < 90, np.cos(DTOR * solar_zenith), 1.0)

        name = 'S{}_reflectance_{}'.format(channel_num, view)
        rads[name] = rads[name] / \
            (irradiances[name[:2] +
                         '_solar_irradiance_{}'.format(view)] * mu0) * np.pi
        return rads

    def load_radiance_channel(self, path, channel_num, view='an'):
        excluded_vars = [
            "S{}_exception_{}".format(channel_num, view),
            "S{}_radiance_orphan_{}".format(channel_num, view),
            "S{}_exception_orphan_{}".format(channel_num, view)
        ]

        path = os.path.join(
            path, 'S{}_radiance_{}.nc'.format(
                channel_num, view))
        radiance = xr.open_dataset(
            path, decode_times=False, engine=self._engine, drop_variables=excluded_vars)
        return radiance

    def load_bts(self, view='in'):
        bts = [self.load_bt_channel(self.path, i, view) for i in range(7, 10)]
        bts = xr.merge(bts)
        return bts

    def load_bt_channel(self, path, channel_num, view='in'):
        excluded_vars = [
            "S{}_exception_{}".format(channel_num, view),
            "S{}_BT_orphan_{}".format(channel_num, view),
            "S{}_exception_orphan_{}".format(channel_num, view)
        ]

        path = os.path.join(path, 'S{}_BT_{}.nc'.format(channel_num, view))
        bt = xr.open_dataset(path, decode_times=False, engine=self._engine,
                             drop_variables=excluded_vars)
        return bt

    def load_flags(self):
        flags_path = os.path.join(self.path, 'flags_in.nc')
        excluded = [
            'confidence_orphan_in',
            'pointing_orphan_in',
            'pointing_in',
            'cloud_orphan_in',
            'bayes_orphan_in',
            'probability_cloud_dual_in']
        flags = xr.open_dataset(flags_path, decode_times=False, engine=self._engine,
                                drop_variables=excluded)

        flag_masks = flags.confidence_in.attrs['flag_masks']
        flag_meanings = flags.confidence_in.attrs['flag_meanings'].split()
        flag_map = dict(zip(flag_meanings, flag_masks))

        expanded_flags = {}
        for key, bit in flag_map.items():
            msk = flags.confidence_in & bit
            msk = xr.where(msk > 0, 1, 0)
            expanded_flags[key] = msk

        return flags.assign(**expanded_flags)

    def load_geometry(self):
        path = os.path.join(self.path, 'geometry_tn.nc')
        geo = xr.open_dataset(path, decode_times=False, engine=self._engine)
        return geo

    def load_met(self):
        met_path = os.path.join(self.path, 'met_tx.nc')
        met = xr.open_dataset(met_path, decode_times=False, engine=self._engine)
        met = met[['total_column_water_vapour_tx', 'cloud_fraction_tx',
                   'skin_temperature_tx', 'sea_surface_temperature_tx',
                   'total_column_ozone_tx', 'soil_wetness_tx',
                   'snow_albedo_tx', 'snow_depth_tx', 'sea_ice_fraction_tx',
                   'surface_pressure_tx']]
        met = met.squeeze()
        return met

    def load_geodetic(self, view='an'):
        flags_path = os.path.join(self.path, 'geodetic_{}.nc'.format(view))
        excluded = ['elevation_orphan_an', 'elevation_an',
                    'latitude_orphan_an', 'longitude_orphan_an']
        flags = xr.open_dataset(flags_path, decode_times=False, engine=self._engine,
                                drop_variables=excluded)
        return flags
