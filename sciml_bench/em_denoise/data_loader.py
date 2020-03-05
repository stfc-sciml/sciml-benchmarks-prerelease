import tensorflow as tf
import numpy as np
from pathlib import Path
from sciml_bench.em_denoise.constants import IMG_SIZE

class EMGrapheneDataset:

    def __init__(self, data_dir, batch_size=10, seed=None):
        self._batch_size = batch_size
        self._seed = seed

        data_dir = Path(data_dir)
        train_dir = data_dir / 'train'
        test_dir = data_dir / 'test'

        self._clean_train = np.load(train_dir / 'graphene_img_clean.npy')
        self._clean_test = np.load(test_dir / 'graphene_img_clean.npy')

        self._noise_train = np.load(train_dir / 'graphene_img_noise.npy')
        self._noise_test = np.load(test_dir / 'graphene_img_noise.npy')

    @property
    def dimensions(self):
        return (IMG_SIZE, IMG_SIZE, 1)

    @property
    def train_size(self):
        return len(self._clean_train)

    @property
    def test_size(self):
        return len(self._clean_test)

    def train_fn(self):
        dataset = tf.data.Dataset.from_tensor_slices((self._noise_train, self._clean_train))
        dataset = dataset.shuffle(self.train_size)
        dataset = dataset.batch(self._batch_size)
        return dataset

    def test_fn(self):
        dataset = tf.data.Dataset.from_tensor_slices((self._noise_test, self._clean_test))
        dataset = dataset.batch(self._batch_size)
        return dataset
