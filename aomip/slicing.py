# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 07.05.2023                   #
# ********************************** #

import numpy as np

def stack_slice(volume: np.ndarray, image: np.ndarray, slice_idx: int) -> np.ndarray:
    """
    Implements and outputs a sliced sinogram, parametrized by the row slicing row index.
    :param:
        volume: np.ndarray of np.ndarrays, stack of projections
        image: np.ndarray, array representation of the image
        slice_idx: int, slicing row index
    :return:
        sliced_image: np.ndarray, sliced image given the slicing position
    """
    sliced_row = image[:slice_idx, :]
    volumne = np.vstack((volume, sliced_row))
    return volume