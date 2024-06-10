import os
import sys
import torch
import unittest

sys.path.append("src/")

from dataloader import Loader
from utils import load, config


class UnitTest(unittest.TestCase):
    def setUp(self):

        self.CONFIG = config()

        self.train_dataloader = load(
            filename=os.path.join(
                self.CONFIG["path"]["PROCESSED_DATA_PATH"], "train_dataloader.pkl"
            )
        )
        self.valid_dataloader = load(
            filename=os.path.join(
                self.CONFIG["path"]["PROCESSED_DATA_PATH"], "valid_dataloader.pkl"
            )
        )

    def tearDown(self):
        pass

    def test_total_dataset(self):
        self.assertEqual(
            (
                sum(X.size(0) for X, _ in self.train_dataloader)
                + sum(X.size(0) for X, _ in self.valid_dataloader)
            ),
            18,
        )

    def test_train_data_shape(self):
        X, _ = next(iter(self.train_dataloader))

        self.assertEqual(X.size(), torch.Size([1, 3, 64, 64]))

    def test_valid_data_shape(self):
        _, y = next(iter(self.valid_dataloader))

        self.assertEqual(y.size(), torch.Size([8, 3, 256, 256]))

    def test_train_batch_size(self):
        self.assertEqual(self.train_dataloader.batch_size, 1)

    def test_valid_batch_size(self):
        self.assertEqual(self.valid_dataloader.batch_size, 1 * 8)


if __name__ == "__main__":
    unittest.main()
