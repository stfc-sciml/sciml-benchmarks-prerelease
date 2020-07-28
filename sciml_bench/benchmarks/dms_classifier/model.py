import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import Normalization


def dms_classifier(input_shape, **kwargs) -> tf.keras.Model:
    input_layer = layers.Input(input_shape)
    x = input_layer

    x = Normalization()(x)
    x = layers.Conv2D(32, kernel_size=3, kernel_initializer='he_normal')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, kernel_size=3, kernel_initializer='he_normal')(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(32, kernel_size=3, kernel_initializer='he_normal')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, kernel_size=3, kernel_initializer='he_normal')(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dropout(.2)(x)
    x = layers.Dense(7, activation='sigmoid')(x)

    model = tf.keras.models.Model(input_layer, x)
    return model
