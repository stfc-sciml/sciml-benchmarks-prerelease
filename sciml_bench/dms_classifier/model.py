import tensorflow as tf

def small_cnn_classifier(input_shape, dropout=0., learning_rate=0.001, **params):
    """
    A very basic setup of a small CNN for testing on classification problems
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(4, kernel_size=(4, 4),
                     activation='relu',
                     input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(8, kernel_size=(2, 2), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(8, activation='relu'))
    if dropout > 0.:
        model.add(tf.keras.layers.Dropout(dropout))
    else:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(4, activation='relu'))
    if dropout > 0.:
        model.add(tf.keras.layers.Dropout(dropout))
    else:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    return model

