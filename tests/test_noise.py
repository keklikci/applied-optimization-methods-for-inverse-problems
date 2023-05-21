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

INPUT_PATH = "homework/hw03/images/input/htc2022_04b_recon.tif"
OUTPUT_PATH = "homework/hw03/images/output"

# ensure correct range
func = lambda image: image.astype(np.uint8)

class TestNoise(unittest.TestCase):

    def setUp(self):
        self.noise = Noise()

    def test_string(self):
        self.assertEqual(str(self.noise), "Gaussian")

    def test_seed(self):
        self.assertEqual(self.noise.seed, 42)

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
        image = tifffile.imread(INPUT_PATH)
        noisy_image = func(self.noise.transform(image))
        tifffile.imsave(OUTPUT_PATH + "/" + f"{str(self.noise).lower()}_noise.tif", noisy_image)
        self.assertEqual(str(self.noise).lower(), "gaussian")

    def test_poisson_transform(self):
        image = tifffile.imread(INPUT_PATH)
        self.noise.update(method = "Poisson")
        noisy_image = self.noise.transform(image)
        tifffile.imsave(OUTPUT_PATH + "/" + f"{str(self.noise).lower()}_noise.tif", noisy_image)
        self.assertEqual(str(self.noise).lower(), "poisson")

    def test_salt_pepper_transform(self):
        image = tifffile.imread(INPUT_PATH)
        self.noise.update(method = "Salt-Pepper")
        noisy_image = func(self.noise.transform(image))
        tifffile.imsave(OUTPUT_PATH + "/" + f"{str(self.noise).lower().replace('-', '_')}_noise.tif", noisy_image)
        self.assertEqual(str(self.noise).lower(), "salt-pepper")

if __name__ == "__main__":
    unittest.main()