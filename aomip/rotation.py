# !/usr/bin/env python
# -*- coding: utf-8 -*-
# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 01.05.2023                   #
# ********************************** #

import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from dotenv import load_dotenv

# source configuration file
load_dotenv()


def rotate(image: np.ndarray, offset: int) -> np.ndarray:
    """
    Shifts object over the last axis.
    :param:
        image: np.ndarray of image
        offset: shift value, sourced by configuration
    :return:
        image: np.ndarray of rotated image
    """
    # apply rotational shift alongside the last axis
    last_axis = len(image.shape) - 1
    image = np.roll(image, shift=offset, axis=last_axis)
    return image
