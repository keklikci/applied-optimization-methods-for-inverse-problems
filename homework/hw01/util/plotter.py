# !/usr/bin/env python
# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 30.04.2023                   #
# ********************************** #

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def plot_figure(
    image: np.ndarray = np.array([]),
    save: bool = True,
    save_dir: str = "",
    tag: int = 0,
) -> None:
    """
    Plots and saves the parametrized data.
    :param:
        image: np.ndarray of image data
        save: boolean to save the figure or not
        tag: suffix string, associated file tag
    :return:
        None
    """
    plt.axis("off")
    # convert back to the loaded format
    image = Image.fromarray(image).convert("I;16")
    if save:
        output_file = save_dir + "/000{}.png".format(tag + 1)
        image.save(output_file)
    plt.close()
