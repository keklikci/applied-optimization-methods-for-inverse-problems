# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 07.05.2023                   #
# ********************************** #

import numpy as np


def slicing(projections: list, slice_idx: int) -> np.ndarray:
    """
    Implements and outputs a sliced sinogram, parametrized by the row slicing row index.
    :param:
        projections: list of np.ndarrays, projection filenames
        slice_idx: int, slicing row index
    :return:
        sliced: np.ndarray, slice of volume given the slicing position
    """
    sliced = []
    for projection in projections:
        s = projection[slice_idx, :]
        sliced.append(s)
    sliced = np.transpose(sliced)
    return sliced
