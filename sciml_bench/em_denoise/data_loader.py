import h5py
import tensorflow as tf
import numpy as np
from pathlib import Path
import horovod.tensorflow as hvd

from sciml_bench.core.data_loader import DataLoader
from sciml_bench.em_denoise.constants import IMG_SIZE

class EMGrapheneDataset(DataLoader):

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
                yield images

    @property
    def dimensions(self):
        return (IMG_SIZE, IMG_SIZE, 1)

    @property
    def train_size(self):
        return 8000

    @property
    def test_size(self):
        return 2000

    def train_fn(self, batch_size=10):
        types = tf.float32
        shapes = tf.TensorShape([IMG_SIZE, IMG_SIZE, 1])

        path = str(self._train_dir / 'graphene_img_noise.h5')
        noise_dataset = tf.data.Dataset.from_generator(self._load_data,
                                                 output_types=types,
                                                 output_shapes=shapes,
                                                 args=(path, ))

        path = str(self._test_dir / 'graphene_img_clean.h5')
        clean_dataset = tf.data.Dataset.from_generator(self._load_data,
                                                 output_types=types,
                                                 output_shapes=shapes,
                                                 args=(path, ))

        dataset = tf.data.Dataset.zip((noise_dataset, clean_dataset))
        dataset = dataset.shard(hvd.size(), hvd.rank())
        dataset = dataset.shuffle(self.train_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
        return dataset

    def test_fn(self, batch_size=10):
        types = tf.float32
        shapes = tf.TensorShape([IMG_SIZE, IMG_SIZE, 1])

        path = str(self._train_dir / 'graphene_img_noise.h5')
        noise_dataset = tf.data.Dataset.from_generator(self._load_data,
                                                 output_types=types,
                                                 output_shapes=shapes,
                                                 args=(path, ))

        path = str(self._test_dir / 'graphene_img_clean.h5')
        clean_dataset = tf.data.Dataset.from_generator(self._load_data,
                                                 output_types=types,
                                                 output_shapes=shapes,
                                                 args=(path, ))

        dataset = tf.data.Dataset.zip((noise_dataset, clean_dataset))
        dataset = dataset.shard(hvd.size(), hvd.rank())
        dataset = dataset.batch(batch_size)
        return dataset
