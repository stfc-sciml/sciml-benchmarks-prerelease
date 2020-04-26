import tensorflow as tf
import numpy as np
from sciml_bench.em_denoise.model import autoencoder

def test_autoencoder():
    model = autoencoder((128, 128, 1))

    assert isinstance(model, tf.keras.Model)
    assert model.input_shape == (None, 128, 128, 1)
    assert model.output_shape == (None, 128, 128, 1)

def test_autoencoder_feed_forward():
    model = autoencoder((128, 128, 1))
    output = model.predict(np.random.random((1, 128, 128, 1)))
    assert output.shape == (1, 128, 128, 1)

def test_autoencoder_backprop():
    X = np.random.random((1, 128, 128, 1))
    model = autoencoder((128, 128, 1), learning_rate=0.001)
    model.compile(loss='mse', optimizer='adam')
    history = model.fit(X, X)
    assert isinstance(history, tf.keras.callbacks.History)
