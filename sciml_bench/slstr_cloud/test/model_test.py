import tensorflow as tf
import numpy as np
from sciml_bench.slstr_cloud.model import unet


def test_unet():
    model = unet((128, 128, 2))

    assert isinstance(model, tf.keras.Model)
    assert model.input_shape == (None, 128, 128, 2)
    assert model.output_shape == (None, 128, 128, 1)


def test_unet_predict():
    model = unet((128, 128, 2))
    output = model.predict(np.random.random((1, 128, 128, 2)))
    assert output.shape == (1, 128, 128, 1)


def test_unet_fit():
    X = np.random.random((1, 128, 128, 2))
    Y = np.random.randint(0, 1, size=(1, 128, 128, 1))
    model = unet((128, 128, 2))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    history = model.fit(X, Y)
    assert isinstance(history, tf.keras.callbacks.History)
