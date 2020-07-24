import h5py
import tensorflow as tf
import numpy as np
from pathlib import Path
import horovod.tensorflow as hvd

from sciml_bench.core.data_loader import DataLoader
from sciml_bench.dms_classifier.constants import IMG_HEIGHT, IMG_WIDTH, N_CHANNELS


class DMSDataset(DataLoader):

    def __init__(self, data_dir, seed=None, batch_size: int=10, **kwargs):
        self._seed = seed
        self._data_dir = Path(data_dir)
        self._batch_size = batch_size

    def _load_data(self, path):
        path = path.decode()
        with h5py.File(path, "r") as hdf5_file:
            for i in range(len(hdf5_file['images'])):
                images = np.array(hdf5_file["images"][i])
                labels = np.array(hdf5_file['labels'][i])
                yield images, labels

    @property
    def input_shape(self):
        return (IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)

    @property
    def output_shape(self):
        return (1,)

    def to_dataset(self):
        path = str(self._data_dir / 'dms_phases.h5')
        types = (tf.float32, tf.float32)
        shapes = (tf.TensorShape([IMG_HEIGHT, IMG_WIDTH, 3]),
                  tf.TensorShape([]))
        dataset = tf.data.Dataset.from_generator(self._load_data,
                                                 output_types=types,
                                                 output_shapes=shapes,
                                                 args=(path, ))
        dataset = dataset.shard(hvd.size(), hvd.rank())
        dataset = dataset.map(lambda x, y: (x[:, :, :1], y))
        dataset = dataset.shuffle(500)
        dataset = dataset.batch(self._batch_size)
        return dataset
