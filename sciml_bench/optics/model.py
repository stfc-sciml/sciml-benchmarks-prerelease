import tensorflow as tf
from tensorflow.keras.regularizers import l2

def cnn_model(input_shape=(150, 150, 3), kernel_reg=None, batch_norm=False, **params):
    """ Simple two layer CNN model for binary classification

    This will classify images into damaged and undamaged.

    """
    if kernel_reg is not None:
        kernel_reg = l2(kernel_reg)

    # Create a Keras model.
    model = tf.keras.Sequential()

    #Add layers to the model
    # First layer of the CNN: Convolutional layer
    model.add(
      tf.keras.layers.Conv2D(
        input_shape=input_shape, # (height,width,channels) with channels = 1 indicating greyscale
        filters=32,
        kernel_size=(3, 3), # determine the width and height of the filter matrix / 2D convolution window
        kernel_regularizer=kernel_reg,
        kernel_initializer='he_normal',
        activation='relu',
      )
    )

    if batch_norm:
        model.add(tf.keras.layers.BatchNormalization())

    model.add(
      tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3), # determine the width and height of the filter matrix / 2D convolution window
        kernel_regularizer=kernel_reg,
        kernel_initializer='he_normal',
        activation='relu',
      )
    )

    if batch_norm:
        model.add(tf.keras.layers.BatchNormalization())

    #Pooling layer performs downsampling, reducing dimensions of image to save computing power
    model.add(
      tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
      )
    )


    # Second layer of the CNN
    # Convolutional layer
    model.add(
      tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        kernel_regularizer=kernel_reg,
        kernel_initializer='he_normal',
        activation='relu',
      )
    )

    if batch_norm:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(
      tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        kernel_regularizer=kernel_reg,
        kernel_initializer='he_normal',
        activation='relu',
      )
    )
    if batch_norm:
        model.add(tf.keras.layers.BatchNormalization())
    #Another pooling layer
    model.add(
      tf.keras.layers.MaxPooling2D( # Numbers 2, 2 denote the pool size, which halves the input in both spatial dimension.
        pool_size=(2, 2),
      )
    )

    # Dense layers of the CNN.
    #'Flatten' layer converts 2D arrays produced by previous layers into a column vector
    model.add(tf.keras.layers.Flatten()) # 'flatten' performs the input role
    model.add(
      tf.keras.layers.Dense(
        units=64,
        kernel_initializer='he_normal',
        activation='relu',
      )
    )

    model.add(tf.keras.layers.Dropout(0.2)) # 'dropout' prevents overfitting

    model.add(
      tf.keras.layers.Dense(
        units=32,
        activation='relu',
        kernel_initializer='he_normal'
      )
    )

    # Output layer of the CNN.
    model.add(
      tf.keras.layers.Dense(
        units=1,
        kernel_initializer='he_normal',
        activation='sigmoid',
      ))

    return model
