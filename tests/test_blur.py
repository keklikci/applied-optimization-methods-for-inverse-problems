# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 22.05.2023                   #
# ********************************** #

import unittest
import matplotlib.pyplot as plt
import tifffile
import os
import sys

sys.path.append("homework/hw03")
from PIL import Image
from blur import *

INPUT_PATH = "homework/hw03/images/input/htc2022_04b_recon.tif"
OUTPUT_PATH = "homework/hw03/images/output"

# read input image
image = tifffile.imread(INPUT_PATH)

# ensure correct range
func = lambda image: image.astype(np.uint8)


class TestBlur(unittest.TestCase):
    def setUp(self):
        self.blur = Blur(shape=image.shape, kernel_stride=7, scale=image.std())

    def test_string(self):
        self.assertEqual(str(self.blur), "blurred")

    # will never fail, execute to save the blurred image
    # multiplied by a factor of 100 to decrease contrast
    def test_convolution(self):
        blurred = self.blur.transform(image)
        tifffile.imsave(
            OUTPUT_PATH + "/" + f"{str(self.blur)}.tif", func(blurred * 100)
        )

    # will never fail, execute to save the deblurred image
    def test_deconvolution(self):
        blurred = self.blur.transform(image)
        self.blur.update(_mode="deblurred")
        deblurred = self.blur.transform(blurred)
        tifffile.imsave(OUTPUT_PATH + "/" + f"{str(self.blur)}.tif", func(deblurred))


if __name__ == "__main__":
    unittest.main()
