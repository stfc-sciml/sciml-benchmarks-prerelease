import pytest
import numpy as np
import tensorflow as tf

class FakeDataLoader():

    def __init__(self, input_dims, output_dims, train_size=10, test_size=10):
        self._input_dims = input_dims
        self._output_dims = output_dims
        self._train_size = train_size
        self._test_size = test_size

    @property
    def dimensions(self):
        return self._input_dims

    @property
    def train_size(self):
        return self._train_size

    @property
    def test_size(self):
        return self._test_size

    def train_fn(self, batch_size=10):
        X = np.random.random((self._train_size, ) + self._input_dims)
        y = np.random.random((self._train_size, ) + self._output_dims)
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        return dataset.batch(batch_size)

    def test_fn(self, batch_size=10):
        X = np.random.random((self._test_size, ) + self._input_dims)
        y = np.random.random((self._test_size, ) + self._output_dims)
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        return dataset.batch(batch_size)

def fake_model_fn(input_shape, **params):
    inputs = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs, x)
    model.compile('sgd', loss='mse')
    return model

