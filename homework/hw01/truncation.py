# !/usr/bin/env python
# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 01.05.2023                   #
# ********************************** #

import numpy as np

def truncate(image: np.ndarray, I0: int, input_mode: str = "absorption") -> np.ndarray:
    """
    Truncates image based on the supported modes, i.e., absorption and transmission.
    :param:
        image: np.ndarray of image
        I0: float, initial density
        input_mode: str, type of input image
    :return:
        image: np.ndarray of truncated image
    """
    if input_mode == "absorption":
        image[image < 0] = 0
    elif input_mode == "transmission":
        image[image > I0] = I0
    else:
        raise NotImplementedError("Supported modes are absorption and transmission.")
    return image