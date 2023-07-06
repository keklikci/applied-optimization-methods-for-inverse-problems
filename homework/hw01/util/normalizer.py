# !/usr/bin/env python
# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 30.04.2023                   #
# ********************************** #

import numpy as np


def normalize(image: np.ndarray = np.array([])) -> np.ndarray:
    """
    Min-max scales the values of image data for plotting.
    :param:
        image: np.ndarray containing pixel values
    :return:
        normalized: np.ndarray of normalized image
    """
    normalized = (image - image.min()) / (image.max() - image.min())
    return normalized
