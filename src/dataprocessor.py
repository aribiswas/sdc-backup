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


def load_image(url, output_shape=(224, 224, 3)):
    cwd = os.path.dirname(os.path.abspath(__file__))
    file_name = cwd + "/../data/downloads/image-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".jpg"
    image_file = tf.keras.utils.get_file(file_name, url, extract=True)
    img = tf.keras.preprocessing.image.load_img(image_file).resize(output_shape[:-1])
    return tf.keras.preprocessing.image.img_to_array(img) / 255.


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
    # show_examples()
    # analyze()
    print(load_labels_from_dataset())


if __name__ == "__main__":
    main()
