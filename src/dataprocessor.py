# Utilities for loading data from Stanford Dogs Dataset.
#
# Ari Biswas, 08/03/2021

import tensorflow as tf
import tensorflow_datasets as tfds
import xml.etree.cElementTree as et
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter


def load_annotations(filename):
    csv_df = pd.read_csv(filename, delimiter=",")
    tags = ('name', 'width', 'height', 'depth', 'xmin', 'ymin', 'xmax', 'ymax', 'pose', 'truncated', 'difficult',
            'folder', 'filename')

    info_list = []

    for _, row in csv_df.iterrows():
        file = "../data/Annotation/" + row['annotation']
        tree = et.parse(file)
        root = tree.getroot()
        info = dict()
        for elem in root.iter():
            if elem.tag in tags:
                info[elem.tag] = elem.text
            if elem.tag == 'filename':
                info[elem.tag] = row['filename']
        info_list.append(info)

    df = pd.DataFrame(info_list, columns=tags)
    df = df.astype({'width': 'float', 'height': 'float', 'depth': 'float', 'xmin': 'float', 'ymin': 'float',
                    'xmax': 'float', 'ymax': 'float'})
    return df


def preprocess_image(img_path, bounding_box=None, output_size=None, smooth=True, show=False):
    im = Image.open(img_path)
    if bounding_box is not None:
        im = im.crop(bounding_box)
    if output_size is not None:
        im = im.resize(output_size)
    if smooth:
        im = im.filter(ImageFilter.GaussianBlur)
    if show:
        im.show()
    return np.asarray(im) / 255.0


def one_hot_encode(label, labels_list):
    num_labels = len(labels_list)
    encoded = []
    idx = int(labels_list.index(label))
    enc = np.zeros(num_labels)
    enc[idx] = 1
    encoded.append(enc)
    return encoded


def load_data(filename):
    print('Reading annotations...')
    annotations = load_annotations(filename)

    labels = list(annotations['name'].unique())

    x = []
    y = []

    print("Reading images. This may take a while...")
    for idx, row in annotations.iterrows():
        img_path = f"../data/Images/{row['filename']}"
        bounding_box = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
        output_size = (80, 60)
        image = preprocess_image(img_path, bounding_box, output_size, smooth=True)
        encoded_name = one_hot_encode(row['name'], labels)
        x.append(image)
        y.append(encoded_name)
    print(f"Imported {len(annotations)} files.")

    return x, y


def load_dataset():
    """
    Load training dataset from stanford_dogs tfds
    :return:
    ds_train (tf.data.DataSet) The requested training dataset.
    ds_test (tf.data.DataSet) The requested test dataset.
    ds_info (tfds.core.DatasetInfo) The requested dataset info.
    """
    (ds_train, ds_test), ds_info = tfds.load('stanford_dogs',
                                             split=['train', 'test'],
                                             shuffle_files=True,
                                             as_supervised=True,
                                             with_info=True,
                                             data_dir='../data/tfds')
    return ds_train, ds_test, ds_info


def preprocess_img(image, label, size=(120, 120)):
    """
    Preprocess image by resizing to 80x60, casting data type `uint8` -> `float32`2 and normalizing values between 0-1.
    :param image: The image to be processed.
    :param label: The associated label.
    :param size: Output image size.
    :return:
    processed_image, label
    """
    resized_image = tf.image.resize(image, size)
    cast_img = tf.cast(resized_image, tf.float32)
    processed_img = cast_img / 255.
    return processed_img, label
