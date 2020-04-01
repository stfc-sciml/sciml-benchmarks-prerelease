import tensorflow as tf
from sciml_bench.optics.model import cnn_model

def test_cnn_model():
    model = cnn_model()

    assert isinstance(model, tf.keras.Model)
    assert model.input_shape == (None, 150, 150, 3)
    assert model.output_shape == (None, 1,)

def test_cnn_model_batch_norm():
    model = cnn_model(batch_norm=True)

    assert isinstance(model, tf.keras.Model)
    assert model.input_shape == (None, 150, 150, 3)
    assert model.output_shape == (None, 1,)

def test_cnn_model_kernel_reg():
    model = cnn_model(batch_norm=False, kernel_reg=0.01)

    assert isinstance(model, tf.keras.Model)
    assert model.input_shape == (None, 150, 150, 3)
    assert model.output_shape == (None, 1,)

