# !/usr/bin/env python
# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 30.04.2023                   #
# ********************************** #

import numpy as np


def apply_correction(
    scan: np.ndarray = np.array([]),
    dark_frame: np.ndarray = np.array([]),
    flat_fields: list = [],
):
    """
    :param:
        scan: np.ndarray of flat-field detector output, i.e., CT scan
        dark_frame: np.ndarray, dark frame
        flat_fields: list of np.ndarrays for flat field images
    :return:
        b: np.ndarray of flat-field corrected output image
    """
    # average flat-fields element-wise
    f = np.mean(flat_fields, axis=0)
    # corrected scan output
    b = (scan - dark_frame) / (f - dark_frame)
    return b
