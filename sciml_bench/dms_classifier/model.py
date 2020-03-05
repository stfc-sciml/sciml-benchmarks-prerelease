import tensorflow as tf

def small_cnn_classifier(img_height, img_width, n_channels=3, n_classes=2, dropout=0.):
    """
    A very basic setup of a small CNN for testing on classification problems
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(8, kernel_size=(4, 4),
                     activation='relu',
                     input_shape=(img_width, img_height, 3)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(16, kernel_size=(2, 2), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    if dropout > 0.:
        model.add(tf.keras.layers.Dropout(dropout))
    else:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(8, activation='relu'))
    if dropout > 0.:
        model.add(tf.keras.layers.Dropout(dropout))
    else:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(n_classes, activation='sigmoid'))

    return model

