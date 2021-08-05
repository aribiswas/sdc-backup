import unittest
import pandas as pd
from src import dataprocessor as prep


class PreprocessingTestCase(unittest.TestCase):
    def test_load_data(self):
        train_file = '../data/train_annotations.csv'
        x_train, y_train = prep.load_data(train_file)

        # 1. There are 12000 training samples
        self.assertEqual(len(x_train), 12000)
        self.assertEqual(len(y_train), 12000)


if __name__ == '__main__':
    unittest.main()
