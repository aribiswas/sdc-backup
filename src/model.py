import tensorflow as tf


def basicnet(input_size):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=input_size),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(120)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    return model


def alexnet(input_size):
    # AlexNet-like model architecture:

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, 11, activation='relu', padding='same', input_shape=input_size),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3)),
        tf.keras.layers.Conv2D(32, 4, activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3)),
        tf.keras.layers.Conv2D(64, 4, activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(120)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    return model
