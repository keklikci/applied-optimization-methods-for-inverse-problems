# !/usr/bin/env python

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 30.04.2023                   #
# ********************************** #

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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
        output_file = save_dir + f"/000{tag + 1}.png"
        image.save(output_file)
    plt.close()
