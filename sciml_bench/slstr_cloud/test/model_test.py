import tensorflow as tf
import numpy as np
from sciml_bench.slstr_cloud.model.unet import unet_v1

def test_unet_v1():
    model = unet_v1((128, 128, 2))

    assert isinstance(model, tf.keras.Model)
    assert model.input_shape == (None, 128, 128, 2)
    assert model.output_shape == (None, 128, 128, 2)

def test_unet_v1_feed_forward():
    model = unet_v1((128, 128, 2))
    output = model.predict(np.random.random((1, 128, 128, 2)))
    assert output.shape == (1, 128, 128, 2)

def test_unet_v1_backprop():
    X = np.random.random((1, 128, 128, 2))
    Y = np.random.randint(0, 1, size=(1, 128, 128, 2))
    model = unet_v1((128, 128, 2), learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    history = model.fit(X, Y)
    assert isinstance(history, tf.keras.callbacks.History)
