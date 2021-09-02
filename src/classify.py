import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy
import tensorflow as tf
import dataprocessor as proc
import argparse
from model import load_model


model = load_model('../trained_models/model_1.h5')
input_shape = model.layers[0].input_shape[1:]
num_breeds = model.layers[-1].output_shape[1]
# breed_names = proc.load_labels_from_dataset()
ds_train, ds_test, ds_info = proc.load_dataset()


def predict(x, top_k=5, verbose=True):
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
        value = top_k_pred.numpy()[0][ct]
        predictions[name] = value
        if verbose:
            print(name + " : {:.2f}%".format(value*100))

    return predictions


def main(url=None):
    if url is None:
        url = input('Enter an url for the image: ')
        url = [url]

    # download the images
    img_numpy = []
    for item in url:
        img_numpy.append(proc.load_image(item))

    # predict
    print("")
    for idx, val in enumerate(url):
        print(f"Image URL: {val}")
        predict(img_numpy[idx], top_k=3)
        print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dog breed classifier.')
    parser.add_argument('--url', metavar='image_url', type=str, nargs=1, default=None, help='URL of the image.')
    args = parser.parse_args()
    main(args.url)
