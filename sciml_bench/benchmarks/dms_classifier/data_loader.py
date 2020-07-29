import h5py
import tensorflow as tf
import numpy as np
from pathlib import Path
import horovod.tensorflow as hvd

from sciml_bench.core.data_loader import DataLoader
from sciml_bench.benchmarks.dms_classifier.constants import IMG_HEIGHT, IMG_WIDTH, N_CHANNELS


class DMSDataset(DataLoader):

    def __init__(self, data_dir, seed=None, batch_size: int=10, is_training_data: bool=True, **kwargs):
        self._seed = seed
        self._data_dir = Path(data_dir)
        self._batch_size = batch_size

        if is_training_data:
            self._image_name = 'data/train'
            self._label_name = 'labels/train'
        else:
            self._image_name = 'data/test'
            self._label_name = 'labels/test'

    def _load_data(self, path):
        path = path.decode()

        with h5py.File(path, "r") as hdf5_file:
            for i in range(0, len(hdf5_file[self._image_name]), self._batch_size):
                images = np.array(hdf5_file[self._image_name][i:i + self._batch_size])
                labels = np.array(hdf5_file[self._label_name][i:i + self._batch_size, 0])
                yield images, labels

    @property
    def input_shape(self):
        return (IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)

    @property
    def output_shape(self):
        return (1,)

    def to_dataset(self):
        path = str(self._data_dir / 'dxs-data.hdf5')
        types = (tf.float32, tf.float32)
        shapes = (tf.TensorShape([None, IMG_HEIGHT, IMG_WIDTH, N_CHANNELS]),
                  tf.TensorShape([None]))
        dataset = tf.data.Dataset.from_generator(self._load_data,
                                                 output_types=types,
                                                 output_shapes=shapes,
                                                 args=(path, ))
        dataset = dataset.unbatch()
        dataset = dataset.shard(hvd.size(), hvd.rank())
        dataset = dataset.shuffle(2000)
        dataset = dataset.batch(self._batch_size)
        return dataset
