import h5py
import tensorflow as tf
import numpy as np
from pathlib import Path
import horovod.tensorflow as hvd

import sciml_bench.mark
from sciml_bench.core.data_loader import DataLoader
from sciml_bench.benchmarks.em_denoise.constants import IMG_SIZE


@sciml_bench.mark.data_loader('em_denoise')
@sciml_bench.mark.validation_data_loader('em_denoise')
class EMGrapheneDataset(DataLoader):

    def __init__(self, data_dir, seed=None, batch_size=10, **kwargs):
        self._seed = seed
        self._data_dir = Path(data_dir)
        self._batch_size = 10

    def _load_data(self, path):
        path = path.decode()
        with h5py.File(path, "r") as hdf5_file:
            for i in range(len(hdf5_file['images'])):
                images = np.array(hdf5_file["images"][i])
                yield images

    @property
    def input_shape(self):
        return (IMG_SIZE, IMG_SIZE, 1)

    @property
    def output_shape(self):
        return (IMG_SIZE, IMG_SIZE, 1)

    def to_dataset(self):
        types = tf.float32
        shapes = tf.TensorShape([IMG_SIZE, IMG_SIZE, 1])

        path = str(self._data_dir / 'graphene_img_noise.h5')
        noise_dataset = tf.data.Dataset.from_generator(self._load_data,
                                                 output_types=types,
                                                 output_shapes=shapes,
                                                 args=(path, ))

        path = str(self._data_dir / 'graphene_img_clean.h5')
        clean_dataset = tf.data.Dataset.from_generator(self._load_data,
                                                 output_types=types,
                                                 output_shapes=shapes,
                                                 args=(path, ))

        dataset = tf.data.Dataset.zip((noise_dataset, clean_dataset))
        dataset = dataset.shard(hvd.size(), hvd.rank())
        dataset = dataset.shuffle(1000)
        dataset = dataset.batch(self._batch_size)
        return dataset
