import h5py
import tensorflow as tf
import numpy as np
from pathlib import Path
import horovod.tensorflow as hvd

from sciml_bench.core.data_loader import DataLoader
from sciml_bench.dms_classifier.constants import IMG_HEIGHT, IMG_WIDTH, N_CHANNELS

class DMSDataset(DataLoader):

    def __init__(self, data_dir, seed=None):
        self._seed = seed

        data_dir = Path(data_dir)
        self._train_dir = data_dir / 'train'
        self._test_dir = data_dir / 'test'

    def _load_data(self, path):
        path = path.decode()
        with h5py.File(path, "r") as hdf5_file:
            for i in range(len(hdf5_file['images'])):
                images = np.array(hdf5_file["images"][i])
                labels = np.array(hdf5_file['labels'][i])
                yield images, labels

    @property
    def dimensions(self):
        return (IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)

    @property
    def train_size(self):
        return 6448

    @property
    def test_size(self):
        return 1612

    def train_fn(self, batch_size=10):
        path = str(self._train_dir / 'dms_phases.h5')
        types = (tf.float32, tf.float32)
        shapes = (tf.TensorShape([IMG_HEIGHT, IMG_WIDTH, 3]),
                  tf.TensorShape([]))
        dataset = tf.data.Dataset.from_generator(self._load_data,
                                                 output_types=types,
                                                 output_shapes=shapes,
                                                 args=(path, ))
        dataset = dataset.shard(hvd.size(), hvd.rank())
        dataset = dataset.map(lambda x, y: (x[:, :, :1], y))
        dataset = dataset.shuffle(1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
        return dataset

    def test_fn(self, batch_size=10):
        path = str(self._test_dir / 'dms_phases.h5')

        types = (tf.float32, tf.float32)
        shapes = (tf.TensorShape([IMG_HEIGHT, IMG_WIDTH, 3]),
                  tf.TensorShape([]))
        dataset = tf.data.Dataset.from_generator(self._load_data,
                                                 output_types=types,
                                                 output_shapes=shapes,
                                                 args=(path, ))
        dataset = dataset.map(lambda x, y: (x[:, :, :1], y))
        dataset = dataset.batch(batch_size)
        return dataset
