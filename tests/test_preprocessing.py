import unittest
import tensorflow_datasets as tfds
from src import dataprocessor as proc
from PIL import Image


class PreprocessingTestCase(unittest.TestCase):
    def test_load_data(self):
        train_file = '../data/train_annotations.csv'
        x_train, y_train = proc.load_data(train_file)

        # 1. There are 12000 training samples
        self.assertEqual(len(x_train), 12000)
        self.assertEqual(len(y_train), 12000)

    def test_preprocess_img(self):
        fcn = lambda im, lab: (proc.preprocess_img(im, (120, 120)), lab)
        ds_train, _, ds_info = proc.load_dataset()
        ds_sample = ds_train.take(1)
        for image, label in tfds.as_numpy(ds_sample):
            resized_img, _ = fcn(image, label)
            img = proc.tensor_to_image(resized_img)
            img.show()
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
