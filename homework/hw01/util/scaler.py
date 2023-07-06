# !/usr/bin/env python
# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 01.05.2023                   #
# ********************************** #

import numpy as np


def uint_mapper(image: np.ndarray = np.array([]), output_mode: str = "uint16"):
    """
    Maps unsigned integer values to the correct range, given the image representation.
    Implementation is referenced by this link, https://scikit-image.org/docs/stable/user_guide/data_types.html
    :param:
        image: np.ndarray of a normalized image
        output_mode: str, data type
    :return:
        mapped: mapped representation of the image
    """
    mapped = np.array([])
    if output_mode == "uint16":
        image *= 65535
        mapped = image.astype(np.uint16)
    elif output_mode == "uint8":
        image *= 255
        mapped = image.astype(np.uint8)
    return mapped
