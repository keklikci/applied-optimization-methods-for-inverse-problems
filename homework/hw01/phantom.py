# !/usr/bin/env python
# -*- coding: utf-8 -*-
# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 01.05.2023                   #
# ********************************** #

import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom

def shepp_logan() -> np.ndarray:
    """
    Creates a shepp logan phantom using scikit-image.
    :param:
        None
    :return:
        image: np.ndarray, 2D image
    """
    image = shepp_logan_phantom()
    return image

def plot_phantom(phantom: np.ndarray) -> None:
    """
    Plots the parametrized phantom.
    :param:
        phantom: np.ndarray of phantom image
    :return:
        None
    """ 
    out_dir = os.path.join(os.getcwd(), "homework", "hw01", "output", "phantom", "raw")
    os.makedirs(out_dir, exist_ok=True)
    plt.axis("off")
    plt.imshow(phantom, cmap="gray")
    plt.savefig(out_dir + "/phantom.png", transparent=True)
    plt.close()