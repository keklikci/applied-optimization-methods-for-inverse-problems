# !/usr/bin/env python
# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 01.05.2023                   #
# ********************************** #

import numpy as np

def average_samples(sliced_vector: list) -> float:
    """
    Returns averaged samples of a pair-wise row vector.
    :param:
        sliced_vector: np.ndarray, sliced row vector of a 2D np.ndarray image
    :return:
        averages: list of floats, including average for non-overlapping pixel values
    """
    averages = [vector.mean() for vector in sliced_vector]
    return averages

def bin(image: np.ndarray = np.array([]), factor: int = 2) -> np.ndarray:
    """
    Inputs and returns a 2D binned signal/image.
    :param:
        image: 2D np.ndarray signal/image
        factor: binning factor
    :return:
        binned_image: binned 2D np.ndarray of signal/image
    """
    if factor % 2 != 0:
        raise ValueError("Binning factor must be a power of 2!")
    binned_image = []
    # dimensionality of the resulting image
    ndim = image[0].shape[0] // factor
    nrows, ncols = image.shape
    for index in range(nrows):
        row_vector = image[index].reshape(-1, )
        split_value = row_vector.size // factor
        sliced_vector = np.array_split(row_vector, split_value)
        averages = average_samples(sliced_vector)
        binned_image += averages
    binned_image = np.array(binned_image)
    binned_image = binned_image.reshape(-1, ndim)
    return binned_image
