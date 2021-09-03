import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import math
import numpy
import argparse
import tensorflow as tf
import dataprocessor as proc
import matplotlib.pyplot as plt
from model import load_model
from PIL import Image


model = load_model('../trained_models/model_1.h5')
input_shape = model.layers[0].input_shape[1:]
num_breeds = model.layers[-1].output_shape[1]
ds_train, ds_test, ds_info = proc.load_dataset()
test_url = [
        'https://media.nature.com/lw800/magazine-assets/d41586-020-03053-2/d41586-020-03053-2_18533904.jpg',
        'https://static01.nyt.com/images/2021/05/11/science/04TB-DOGS/04TB-DOGS-superJumbo.jpg',
        'https://images.theconversation.com/files/319652/original/file-20200310-61148-vllmgm.jpg',
        'https://s01.sgp1.cdn.digitaloceanspaces.com/article/131928-mxiccwtarv-1575034997.jpg'
    ]


def predict(x, top_k=5, verbose=True):
    """
    Predict the top k breeds for an image.
    :param x: Image of the dog. Must be a numpy array or tensor.
    :param top_k: Scalar specifying the top k breeds for prediction.
    :param verbose: Flag to print results on the command line. True or False.
    :return: A dictionary object containing the predictions.
    """
    if tf.is_tensor(x):
        x = tf.reshape(x[0], [1] + list(input_shape))
    elif isinstance(x, numpy.ndarray):
        assert x.shape == input_shape
        x = tf.reshape(x, [1] + list(input_shape))

    # predict
    pred = model.predict(x)
    top_k_pred, top_k_indices = tf.math.top_k(pred, k=top_k)

    # display the prediction
    predictions = dict()
    for ct in range(top_k):
        name = ds_info.features['label'].int2str(top_k_indices[0][ct])
        name = "".join(name.split('-')[1:])
        value = top_k_pred.numpy()[0][ct]
        predictions[name] = value
        if verbose:
            print(name + " : {:.2f}%".format(value*100))
    return predictions


def main(url=None):
    if url is None:
        url = []
        while True:
            url_input = input('Enter an url for the image, 0 to quit: ')
            if url_input == '0':
                break
            url.append(url_input)

    # download the images
    img_list, file_list = proc.load_image(url)

    # predict
    print("")
    result = []
    for idx, val in enumerate(url):
        print(f"Image URL: {val}")
        result.append(predict(img_list[idx], top_k=3))
        print("")

    num_images = len(img_list)
    if num_images == 1:
        num_rows = 1
        num_cols = 1
    else:
        num_rows = 2
        num_cols = math.ceil(num_images/2)

    for idx, val in enumerate(file_list):
        plt.subplot(num_rows, num_cols, idx+1)
        img = Image.open(val)
        img.thumbnail((120, 120), Image.ANTIALIAS)  # resizes image in-place
        plt.imshow(img)
        title_str = ""
        for k, v in result[idx].items():
            title_str += f'{k} - {100*v:.2f}%\n'
        plt.title(title_str)
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout(pad=2.0)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dog breed classifier.')
    parser.add_argument('--url', metavar='image_url', type=str, nargs='+', default=None, help='URL of the image.')
    args = parser.parse_args()
    # main(args.url)
    main(test_url)
