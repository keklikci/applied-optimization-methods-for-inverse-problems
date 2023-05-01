# !/usr/bin/env python
# -*- coding: utf-8 -*-
# ************************************************************************************************** #
# Author: kaanguney.keklikci@tum.de                                                                  #
# Date: 01.05.2023                                                                                   #
# Reference: https://scikit-image.org/docs/stable/auto_examples/transform/plot_radon_transform.html  #
# ************************************************************************************************** #

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon
from phantom import *

def transform(phantom: np.ndarray) -> np.ndarray:
    """
    Applies a radon transformation to the phantom.
    :param:
        phantom: np.ndarray of phantom image
    :return:
        sinogram: np.ndarray, the corresponding radon transform
    """
    theta = np.linspace(0., 180., max(phantom.shape), endpoint=False)
    sinogram = radon(phantom, theta=theta)
    return sinogram

def plot_sinogram(sinogram: np.ndarray) -> None:
    """
    Plots the parametrized sinogram.
    :param:
        sinogram: np.ndarray of sinogram
    :return:
        None
    """ 
    out_dir = os.path.join(os.getcwd(), "homework", "hw01", "output", "phantom", "sinogram")
    os.makedirs(out_dir, exist_ok=True)
    plt.axis("off")
    dx, dy = 0.5 * 180.0 / max(phantom.shape), 0.5 / sinogram.shape[0]
    plt.imshow(sinogram, extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy), aspect="auto", cmap="gray")
    plt.savefig(out_dir + "/sinogram.png", transparent=True)
    plt.close()

if __name__ == "__main__":
    phantom = shepp_logan()
    plot_phantom(phantom)
    sinogram = transform(phantom)
    plot_sinogram(sinogram)
