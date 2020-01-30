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
import math
import os
import multiprocessing
from glob import glob

import tensorflow as tf
import numpy as np
import xarray as xr
from scipy import interpolate
from skimage import transform
from PIL import Image, ImageSequence


class Dataset():
    """Load, separate and prepare the data for training and prediction"""

    def __init__(self, data_dir, batch_size, augment=False, gpu_id=0, num_gpus=1, seed=0):
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._augment = augment

        self._seed = seed

        self._train_images = \
            self._load_multipage_tiff(os.path.join(
                self._data_dir, 'train-volume.tif'))
        self._train_masks = \
            self._load_multipage_tiff(os.path.join(
                self._data_dir, 'train-labels.tif'))
        self._test_images = \
            self._load_multipage_tiff(os.path.join(
                self._data_dir, 'test-volume.tif'))

        self._num_gpus = num_gpus
        self._gpu_id = gpu_id

    @property
    def train_size(self):
        return len(self._train_images)

    @property
    def test_size(self):
        return len(self._test_images)

    def _load_multipage_tiff(self, path):
        """Load tiff images containing many images in the channel dimension"""
        return np.array([np.array(p) for p in ImageSequence.Iterator(Image.open(path))])

    def _normalize_inputs(self, inputs):
        """Normalize inputs"""
        inputs = tf.expand_dims(tf.cast(inputs, tf.float32), -1)

        # Center around zero
        inputs = tf.divide(inputs, 127.5) - 1

        inputs = tf.image.resize_images(inputs, (392, 392))

        return tf.image.resize_image_with_crop_or_pad(inputs, 572, 572)

    def _normalize_labels(self, labels):
        """Normalize labels"""
        labels = tf.expand_dims(tf.cast(labels, tf.float32), -1)
        labels = tf.divide(labels, 255)

        labels = tf.image.resize_images(labels, (388, 388))
        labels = tf.image.resize_image_with_crop_or_pad(labels, 572, 572)

        cond = tf.less(labels, 0.5 * tf.ones(tf.shape(labels)))
        labels = tf.where(cond, tf.zeros(tf.shape(labels)),
                          tf.ones(tf.shape(labels)))

        return tf.one_hot(tf.squeeze(tf.cast(labels, tf.int32)), 2)

    def _preproc_samples(self, inputs, labels, augment=True):
        """Preprocess samples and perform random augmentations"""
        inputs = self._normalize_inputs(inputs)
        labels = self._normalize_labels(labels)

        if self._augment and augment:
            # Horizontal flip
            h_flip = tf.random_uniform([]) > 0.5
            inputs = tf.cond(h_flip, lambda: tf.image.flip_left_right(
                inputs), lambda: inputs)
            labels = tf.cond(h_flip, lambda: tf.image.flip_left_right(
                labels), lambda: labels)

            # Vertical flip
            v_flip = tf.random_uniform([]) > 0.5
            inputs = tf.cond(v_flip, lambda: tf.image.flip_up_down(
                inputs), lambda: inputs)
            labels = tf.cond(v_flip, lambda: tf.image.flip_up_down(
                labels), lambda: labels)

            # Prepare for batched transforms
            inputs = tf.expand_dims(inputs, 0)
            labels = tf.expand_dims(labels, 0)

            # Elastic deformation

            alpha = tf.random.uniform([], minval=0, maxval=34)

            # Create random vector flows
            delta_x = tf.random.uniform([1, 4, 4, 1], minval=-1, maxval=1)
            delta_y = tf.random.uniform([1, 4, 4, 1], minval=-1, maxval=1)

            # Build 2D flow and apply
            flow = tf.concat([delta_x, delta_y], axis=-1) * alpha
            inputs = tf.contrib.image.dense_image_warp(inputs,
                                                       tf.image.resize_images(flow, (572, 572)))
            labels = tf.contrib.image.dense_image_warp(labels,
                                                       tf.image.resize_images(flow, (572, 572)))

            # Rotation invariance

            # Rotate by random angle\
            radian = tf.random_uniform([], maxval=360) * math.pi / 180
            inputs = tf.contrib.image.rotate(inputs, radian)
            labels = tf.contrib.image.rotate(labels, radian)

            # Shift invariance

            # Random crop and resize
            left = tf.random_uniform([]) * 0.3
            right = 1 - tf.random_uniform([]) * 0.3
            top = tf.random_uniform([]) * 0.3
            bottom = 1 - tf.random_uniform([]) * 0.3

            inputs = tf.image.crop_and_resize(
                inputs, [[top, left, bottom, right]], [0], (572, 572))
            labels = tf.image.crop_and_resize(
                labels, [[top, left, bottom, right]], [0], (572, 572))

            # Gray value variations

            # Adjust brightness and keep values in range
            inputs = tf.image.random_brightness(inputs, max_delta=0.2)
            inputs = tf.clip_by_value(
                inputs, clip_value_min=-1, clip_value_max=1)

            inputs = tf.squeeze(inputs, 0)
            labels = tf.squeeze(labels, 0)

        # Bring back labels to network's output size and remove interpolation artifacts
        labels = tf.image.resize_image_with_crop_or_pad(
            labels, target_width=388, target_height=388)
        cond = tf.less(labels, 0.5 * tf.ones(tf.shape(labels)))
        labels = tf.where(cond, tf.zeros(tf.shape(labels)),
                          tf.ones(tf.shape(labels)))

        return (inputs, labels)

    def train_fn(self):
        """Input function for training"""
        dataset = tf.data.Dataset.from_tensor_slices(
            (self._train_images, self._train_masks))
        dataset = dataset.shuffle(self._batch_size * 3)
        dataset = dataset.repeat()
        dataset = dataset.shard(self._num_gpus, self._gpu_id)
        dataset = dataset.apply(
            tf.data.experimental.map_and_batch(map_func=self._preproc_samples,
                                               batch_size=self._batch_size,
                                               num_parallel_calls=multiprocessing.cpu_count()))
        dataset = dataset.prefetch(self._batch_size)

        return dataset

    def test_fn(self, count):
        """Input function for testing"""
        dataset = tf.data.Dataset.from_tensor_slices(
            self._test_images)
        dataset = dataset.repeat(count=count)
        dataset = dataset.map(self._normalize_inputs)
        dataset = dataset.batch(self._batch_size)
        dataset = dataset.prefetch(self._batch_size)

        return dataset

    def synth_fn(self):
        """Synthetic data function for testing"""
        inputs = tf.truncated_normal((572, 572, 1), dtype=tf.float32, mean=127.5, stddev=1, seed=self._seed,
                                     name='synth_inputs')
        masks = tf.truncated_normal((388, 388, 2), dtype=tf.float32, mean=0.01, stddev=0.1, seed=self._seed,
                                    name='synth_masks')

        dataset = tf.data.Dataset.from_tensors((inputs, masks))

        dataset = dataset.cache()
        dataset = dataset.repeat()
        dataset = dataset.batch(self._batch_size)

        dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

        return dataset


class Sentinel3Dataset():
    """Load, separate and prepare the data for training and prediction"""

    def __init__(self, data_dir, batch_size, augment=False, gpu_id=0, num_gpus=1, seed=0):
        self._data_dir = data_dir
        self._batch_size = batch_size

        self._train_images = glob(os.path.join(self._data_dir, 'S3A*'))
        self._test_images = glob(os.path.join(self._data_dir, 'S3A*'))[:10]

        self._num_gpus = num_gpus
        self._gpu_id = gpu_id
        self._seed = seed


    @property
    def train_size(self):
        return len(self._train_images)

    @property
    def test_size(self):
        return len(self._test_images)

    def _parse_file(self, path):
        path = path.decode()

        loader = ImageLoader(path)
        rads = loader.load_radiances()
        bts = loader.load_bts()
        flags = loader.load_flags()
        bts = bts.to_array().values
        mask = flags.summary_cloud.values
        mask[mask > 0] = 1

        bts = np.nan_to_num(bts, nan=bts[~np.isnan(bts)].min())
        # Mean - Std norm
        bts = (bts - 270.0) / 22.0
        bts = transform.resize(bts, (1200, 1500, 3))

        rads = rads.to_array().values
        rads = np.nan_to_num(rads, nan=rads[~np.isnan(rads)].min())
        # Mean - Std norm, already in range 0-1, convert to (-1, 1) range.
        rads = rads - 0.5
        rads = np.transpose(rads, [1, 2, 0])
        rads = transform.resize(rads, (1200, 1500, 6))

        channels = np.concatenate([rads, bts], axis=-1)
        channels = channels[100:-100, 100:-100]

        mask = mask.astype(np.float32)
        mask = np.nan_to_num(mask)
        mask = transform.resize(
            mask, (1200, 1500), order=0, anti_aliasing=False)
        mask = mask[100:-100, 100:-100]


        msk = np.zeros((1000, 1300, 2))
        msk[..., 0] = mask
        msk[..., 1] = 1-mask


        yield (channels, msk)

    def _generator(self, path):
        types = ( tf.float32, tf.float32)
        shapes = (tf.TensorShape( [1000, 1300, 9]), tf.TensorShape([1000,
            1300, 2]))
        dataset = tf.data.Dataset.from_generator(self._parse_file,
                                                 output_types=types,
                                                 output_shapes=shapes,
                                                 args=(path, ))
        return dataset

    def _transform(self, img, msk):
        img = tf.expand_dims(img, axis=0)
        msk = tf.expand_dims(msk, axis=0)

        img = tf.extract_image_patches(img, [1, 64, 64, 1], [1, 64, 64, 1], [
                                       1, 1, 1, 1], padding='VALID')
        msk = tf.extract_image_patches(msk, [1, 64, 64, 1], [1, 64, 64, 1], [
                                       1, 1, 1, 1], padding='VALID')

        n, nx, ny, np = img.shape
        img = tf.reshape(img, (n*nx*ny, 64, 64, 9))
        msk = tf.reshape(msk, (n*nx*ny, 64, 64, 2))

        return img, msk

    def train_fn(self):
        """Input function for training"""
        dataset = tf.data.Dataset.from_tensor_slices(self._train_images)
        dataset = dataset.shuffle(1000)
        dataset = dataset.repeat()
        dataset = dataset.shard(self._num_gpus, self._gpu_id)
        dataset = dataset.apply(tf.contrib.data.parallel_interleave(
            self._generator, cycle_length=1))
        dataset = dataset.map(self._transform)
        dataset = dataset.apply(tf.data.experimental.unbatch())
        dataset = dataset.shuffle(self._batch_size * 3)
        dataset = dataset.batch(self._batch_size)
        dataset = dataset.prefetch(self._batch_size)

        return dataset

    def test_fn(self, count):
        dataset = tf.data.Dataset.from_tensor_slices(self._test_images)
        dataset = dataset.repeat()
        dataset = dataset.shard(self._num_gpus, self._gpu_id)
        dataset = dataset.apply(tf.contrib.data.parallel_interleave(
            self._generator, cycle_length=1))
        dataset = dataset.map(self._transform)
        dataset = dataset.apply(tf.data.experimental.unbatch())
        dataset = dataset.batch(self._batch_size)
        dataset = dataset.take(count)
        return dataset

    def synth_fn(self):
        inputs = tf.truncated_normal((256, 256, 1), dtype=tf.float32, mean=127.5, stddev=1, seed=self._seed,
                                     name='synth_inputs')
        masks = tf.truncated_normal((256, 256, 2), dtype=tf.float32, mean=0.01, stddev=0.1, seed=self._seed,
                                    name='synth_masks')
        dataset = tf.data.Dataset.from_tensors((inputs, masks))
        dataset = dataset.shard(self._num_gpus, self._gpu_id)
        dataset = dataset.shuffle(self._batch_size * 5)
        dataset = dataset.repeat()
        dataset = dataset.batch(self._batch_size)
        dataset = dataset.prefetch(self._batch_size)
        return dataset


class ImageLoader:

    def __init__(self, path):
        self.path = path

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
            irradiance = xr.open_dataset(file_name, engine='h5netcdf')[name][:].data[0]
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
            path, decode_times=False, engine='h5netcdf', drop_variables=excluded_vars)
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
        bt = xr.open_dataset(path, decode_times=False, engine='h5netcdf',
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
        flags = xr.open_dataset(flags_path, decode_times=False, engine='h5netcdf',
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
        geo = xr.open_dataset(path, decode_times=False, engine='h5netcdf')
        return geo

    def load_met(self):
        met_path = os.path.join(self.path, 'met_tx.nc')
        met = xr.open_dataset(met_path, decode_times=False, engine='h5netcdf')
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
        flags = xr.open_dataset(flags_path, decode_times=False, engine='h5netcdf',
                                drop_variables=excluded)
        return flags
