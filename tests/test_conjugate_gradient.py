# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 22.05.2023                   #
# ********************************** #

import os
import sys
import unittest

sys.path.append("homework/hw03")
from conjugate_gradient import *
from PIL import Image

OUTPUT_PATH = "homework/hw03/images/output"

# X-ray operator arguments
thetas = np.arange(360)
volume_shape = [128, 128]
sinogram_shape = [128]
d2c = volume_shape[0] * 100.0
c2d = volume_shape[0] * 5.0


class ConjugateGradientTest(unittest.TestCase):
    def setUp(self):
        self.algorithm = ConjugateGradient()

    def test_string(self):
        self.assertEqual(str(self.algorithm), "Conjugate Gradient")

    def test_n_iterations(self):
        self.assertEqual(self.algorithm.n_iterations, 10)

    # will never fail, execute to save output image
    def test_algorithm(self) -> None:
        try:
            out = self.algorithm.optimize().reshape(volume_shape)
            out = Image.fromarray(out)
            out.save(os.path.join(OUTPUT_PATH, "conjugate_gradient.tif"))
        except:
            self.fail("Encountered runtime exception!")


if __name__ == "__main__":
    unittest.main()
