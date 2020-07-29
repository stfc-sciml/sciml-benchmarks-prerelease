import tensorflow as tf
from tensorflow.keras import layers


def dms_classifier(input_shape, **kwargs) -> tf.keras.Model:
    input_layer = layers.Input(input_shape)
    x = input_layer

    x = layers.Conv2D(32, kernel_size=3)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(32, kernel_size=3)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(32, kernel_size=3)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(10)(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.models.Model(input_layer, x)
    return model
