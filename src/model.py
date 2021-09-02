import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50


def h5_to_savedmodel(h5_path, save_path):
    model = tf.keras.models.load_model(h5_path)
    tf.saved_model.save(model, save_path + "saved_model1")


def load_model(load_path):
    model = tf.keras.models.load_model(load_path)
    return model


def basicnet(image_shape, num_classes, lr=0.001):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=image_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    return model


def alexnet(image_shape, num_classes, lr=0.001):
    # AlexNet-like model architecture:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(96, 11, activation='relu', strides=(4, 4), padding='valid', input_shape=image_shape),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Conv2D(256, 5, activation='relu', strides=(1, 1), padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Conv2D(384, 3, activation='relu', strides=(1, 1), padding='same'),
        tf.keras.layers.Conv2D(384, 3, activation='relu', strides=(1, 1), padding='same'),
        tf.keras.layers.Conv2D(384, 3, activation='relu', strides=(1, 1), padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    return model


def vgg16(image_shape, num_classes, lr=0.001):
    base_model = VGG16(input_shape=image_shape,
                       include_top=False,
                       weights='imagenet',
                       pooling='avg')
    base_model.trainable = False
    model = tf.keras.models.Sequential([
        base_model,
        # tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation='softmax')])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    return model


def resnet50(image_shape, num_classes, lr=0.001):
    base_model = ResNet50(input_shape=image_shape,
                          include_top=False,
                          weights='imagenet',
                          pooling='avg')
    base_model.trainable = False
    model = tf.keras.models.Sequential([
        base_model,
        # tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation='softmax')])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss='categorical_crossentropy',
        metrics='accuracy'
    )
    return model


def mobilenet(image_shape, num_classes, lr=0.001):
    base_model = tf.keras.applications.MobileNetV2(input_shape=image_shape,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    return model
