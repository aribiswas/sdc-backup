# Utilities for loading data from Stanford Dogs Dataset.
#
# Ari Biswas, 08/03/2021

import tensorflow as tf
import tensorflow_datasets as tfds
import xml.etree.cElementTree as et
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from PIL import Image, ImageFilter


def load_dataset():
    """
    Load stanford_dogs tensorflow dataset.
    :return:
    ds_train (tf.data.DataSet) The requested training dataset.
    ds_test (tf.data.DataSet) The requested test dataset.
    ds_info (tfds.core.DatasetInfo) The requested dataset info.
    """
    (ds_train, ds_test), ds_info = tfds.load('stanford_dogs',
                                             split=['train', 'test'],
                                             shuffle_files=True,
                                             as_supervised=False,
                                             with_info=True,
                                             data_dir='../data/tfds')
    return ds_train, ds_test, ds_info


def preprocess(data, image_size, num_labels, cast=True, resize=True, normalize=True, one_hot=True):
    """
    Process an image.
    :param data: Tensorflow datset containing an image and label.
    :param image_size: Size of the image. Images may be resized to this size. E.g. (224, 224)
    :param num_labels: Number of labels for prediction.
    :param cast: Flag for casting to float32. True or False.
    :param resize: Flag for resizing the image. True or False.
    :param normalize: Flag for normalizing the image pixel values from 0-1. True or False.
    :param one_hot: Flag for one hot encoding the labels. True or False.
    :return: Processed image and encoded label.
    """
    # processed_image = tf.keras.applications.resnet.preprocess_input(data['image'])
    processed_image = data['image']
    label = data['label']
    if cast:
        processed_image = tf.cast(processed_image, tf.float32)
    if resize:
        processed_image = tf.image.resize(processed_image, image_size, method='nearest')
    if normalize:
        processed_image = processed_image / 255.
    if one_hot:
        label = tf.one_hot(label, num_labels)
    return processed_image, label


def prepare(dataset, image_shape, num_classes, batch_size=None):
    """
    Prepare an input pipeline for training a dataset.
    :param dataset: The dataset containing training data.
    :param image_shape: A common shape of the input image. Images with different sizes will be resized. E.g. (80, 80, 3)
    :param num_classes: Number of prediction classes.
    :param batch_size: Batch size for training.
    :return: Prepared dataset.
    """
    dataset = dataset.map(lambda x: preprocess(x, image_shape[0:-1], num_classes),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def analyze():
    ds_train, ds_test, ds_info = load_dataset()
    img_width = []
    img_height = []
    for row in ds_train:
        img_width.append(row['image'].shape[0])
        img_height.append(row['image'].shape[1])
    for row in ds_test:
        img_width.append(row['image'].shape[0])
        img_height.append(row['image'].shape[1])

    plt.subplot(211)
    plt.hist(x=img_width, bins=20, alpha=0.7, density=True)
    plt.axvline(max(set(img_width), key=img_width.count), color='r')
    plt.ylabel('Pixels')
    plt.ylabel('Frequency')
    plt.title('Image width distribution')
    plt.grid(axis='y', alpha=0.75)

    plt.subplot(212)
    plt.hist(x=img_height, bins=20, alpha=0.7)
    plt.axvline(max(set(img_height), key=img_height.count), color='r')
    plt.xlabel('Pixels')
    plt.ylabel('Frequency')
    plt.title('Image height distribution')
    plt.grid(axis='y', alpha=0.75)

    plt.subplots_adjust(hspace=0.5)
    plt.show()


def decode(label_index):
    ds_train, _, _ = load_dataset()
    for row in ds_train.take(5):
        label = row['label']
        # print(label)
        print(label.decode_example())


def show_examples():
    """
    Display a random set of examples in the stanford_dogs dataset
    :return: None
    """
    ds_train, _, ds_info = load_dataset()
    ds_train = ds_train.map(lambda x: (preprocess(x, (80, 80), 120, cast=False, resize=True, one_hot=False)))
    tfds.show_examples(ds_train, ds_info)


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


def load_image(url_list, output_shape=(224, 224, 3)):
    cwd = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    img_list = []
    file_list = []
    for idx, url in enumerate(url_list):
        file_name = f"{cwd}/../data/downloads/image-{timestamp}_{idx+1}.jpg"
        image_file = tf.keras.utils.get_file(file_name, url, extract=True)
        img = tf.keras.preprocessing.image.load_img(image_file).resize(output_shape[:-1])
        img_list.append(tf.keras.preprocessing.image.img_to_array(img) / 255.)
        file_list.append(file_name)
    return img_list, file_list


def load_labels():
    breed_names = []
    with open("../data/breed_names.txt", 'r') as f:
        for line in f:
            line = line.rstrip()
            line = line.split('-')[1:]
            line = ''.join(line)
            breed_names.append(line)
    return breed_names


def load_labels_from_dataset():
    ds_train, ds_test, ds_info = load_dataset()
    unformatted_labels = ds_info.features['label'].names
    breed_names = [''.join(item.split('-')[1:]) for item in unformatted_labels]
    return breed_names


def main():
    show_examples()
    # analyze()
    # print(load_labels_from_dataset())


if __name__ == "__main__":
    main()
