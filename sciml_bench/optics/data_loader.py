import pandas as pd
import numpy as np
import tensorflow as tf

from sciml_tools.image import load_tiff
from sciml_bench.core.data_loader import DataLoader

class OpticsDataLoader(DataLoader):

    def __init__(self, data_dir, seed=None):
        self._seed = seed

        train_df = pd.read_csv(data_dir / 'train/meta.csv')
        test_df = pd.read_csv(data_dir / 'test/meta.csv')
        valid_df = pd.read_csv(data_dir / 'valid/meta.csv')

        # Fix paths in dataset
        train_df.path = train_df.path.str.replace('data/processed/schlieren', str(data_dir))
        test_df.path = test_df.path.str.replace('data/processed/schlieren', str(data_dir))
        valid_df.path = valid_df.path.str.replace('data/processed/schlieren', str(data_dir))

        self._train_labels = train_df['class'].astype(np.float32)
        self._test_labels = test_df['class'].astype(np.float32)
        self._valid_labels = valid_df['class'].astype(np.float32)

        self._train_images = np.array([load_tiff(p) for p in train_df.path]).astype(np.float32)
        self._test_images = np.array([load_tiff(p) for p in test_df.path]).astype(np.float32)
        self._valid_images = np.array([load_tiff(p) for p in valid_df.path]).astype(np.float32)

        # Add channel dimension
        self._train_images = np.expand_dims(self._train_images, axis=-1)
        self._test_images = np.expand_dims(self._test_images, axis=-1)
        self._valid_images = np.expand_dims(self._valid_images, axis=-1)

        self._min, self._max = self._train_images.min(), self._train_images.max()

    @property
    def dimensions(self):
        return (296, 394, 1)

    @property
    def train_size(self):
        return len(self._train_labels)

    @property
    def test_size(self):
        return len(self._test_labels)

    @property
    def valid_size(self):
        return len(self._valid_labels)

    def _crop_image(self, x):
        x = tf.image.central_crop(x, .6)
        return x

    def _normalize(self, x):
        x = (x - self._min) / (self._max - self._min)
        return x

    def train_dataset(self, batch_size=10, augment=None, **kwargs):
        dataset = tf.data.Dataset.from_tensor_slices((self._train_images, self._train_labels))
        dataset = dataset.map(lambda x, y: (self._crop_image(x), y))
        dataset = dataset.map(lambda x, y: (self._normalize(x), y))
        dataset = dataset.shuffle(2000)
        dataset = dataset.batch(batch_size)
        return dataset

    def validation_dataset(self, batch_size=10, **kwargs):
        dataset = tf.data.Dataset.from_tensor_slices((self._valid_images, self._valid_labels))
        dataset = dataset.map(lambda x, y: (self._crop_image(x), y))
        dataset = dataset.map(lambda x, y: (self._normalize(x), y))
        dataset = dataset.batch(batch_size)
        return dataset

    def test_dataset(self, batch_size=10, **kwargs):
        dataset = tf.data.Dataset.from_tensor_slices((self._test_images, self._test_labels))
        dataset = dataset.map(lambda x, y: (self._crop_image(x), y))
        dataset = dataset.map(lambda x, y: (self._normalize(x), y))
        dataset = dataset.batch(batch_size)
        return dataset
