import tensorflow as tf
import numpy as np
from sciml_bench.dms_classifier.model import small_cnn_classifier

def test_small_cnn_classifier():
    model = small_cnn_classifier((128, 128, 1))

    assert isinstance(model, tf.keras.Model)
    assert model.input_shape == (None, 128, 128, 1)
    assert model.output_shape == (None, 1)

def test_small_cnn_classifier_feed_forward():
    model = small_cnn_classifier((128, 128, 1))
    output = model.predict(np.random.random((1, 128, 128, 1)))
    assert output.shape == (1, 1)

def test_small_cnn_classifier_backprop():
    X = np.random.random((1, 128, 128, 1))
    Y = np.random.random((1, 1))
    model = small_cnn_classifier((128, 128, 1), learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    history = model.fit(X, Y)
    assert isinstance(history, tf.keras.callbacks.History)
