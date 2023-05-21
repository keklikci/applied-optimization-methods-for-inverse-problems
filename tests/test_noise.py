# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 21.05.2023                   #
# ********************************** #

import unittest
import tifffile
import os
import sys
sys.path.append("homework/hw03")
from noise import *

INPUT_PATH = "homework/hw03/input/htc2022_02c_recon.tif"
OUTPUT_PATH = "homework/hw3/output"

class TestNoise(unittest.TestCase):

    def setUp(self):
        self.noise = Noise()

    def test_string(self):
        self.assertEqual(str(self.noise), "Gaussian")

    def test_update_poisson(self):
        self.noise.update(method = "Poisson")
        self.assertEqual(str(self.noise), "Poisson")

    def test_update_salt_pepper(self):
        self.noise.update(method = "Salt-Pepper")
        self.assertEqual(str(self.noise), "Salt-Pepper")

    def test_validity(self):
        with self.assertRaises(ValueError) as context:
            self.noise.update(method = "invalid key")
        self.assertEqual(str(context.exception), f"Method must be an element of {self.noise.validity}")

    def test_gaussian_transform(self):
        # TODO: add transformation unit tests

if __name__ == "__main__":
    unittest.main()