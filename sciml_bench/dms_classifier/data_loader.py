import h5py
import tensorflow as tf
import numpy as np
from pathlib import Path

from sciml_bench.core.data_loader import DataLoader
from sciml_bench.dms_classifier.constants import IMG_HEIGHT, IMG_WIDTH, N_CHANNELS

class DMSDataset(DataLoader):

    def __init__(self, data_dir, seed=None):
        self._seed = seed

        data_dir = Path(data_dir)
        train_dir = data_dir / 'train'
        test_dir = data_dir / 'test'

        hdf5_file = h5py.File(train_dir / 'dms_phases.h5', "r")
        self._train_images = np.array(hdf5_file["images"])
        self._train_labels = np.array(hdf5_file['labels'])

        hdf5_file = h5py.File(test_dir / 'dms_phases.h5', "r")
        self._test_images = np.array(hdf5_file["images"])
        self._test_labels = np.array(hdf5_file['labels'])

    @property
    def dimensions(self):
        return (IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)

    @property
    def train_size(self):
        return len(self._train_images)

    @property
    def test_size(self):
        return len(self._test_images)

    def train_fn(self, batch_size=10):
        dataset = tf.data.Dataset.from_tensor_slices((self._train_images, self._train_labels))
        dataset = dataset.shuffle(self.train_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
        return dataset

    def test_fn(self, batch_size=10):
        dataset = tf.data.Dataset.from_tensor_slices((self._test_images, self._test_labels))
        dataset = dataset.batch(batch_size)
        return dataset
