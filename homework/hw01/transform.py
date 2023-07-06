# !/usr/bin/env python
# -*- coding: utf-8 -*-
# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 01.05.2023                   #
# ********************************** #

import numpy as np
import os
import matplotlib.pyplot as plt


def estimate_initial_density(image: np.ndarray, window: int = 10) -> float:
    """
    Estimates initial density.
    :param:
        image: np.ndarray of 2D image
        window: small window size such that object is invisible
    :return:
        estimate: float, estimated initial density
    """
    pixels = image[:window, :window]
    estimate = np.mean(pixels)
    return estimate


def transmission_to_absorption(x, IO) -> np.ndarray:
    """
    Takes an estimate and converts transmission image to absorption image.
    :param:
        x: np.ndarray of 2D image
        I0: float, initial estimated density
    :return:
        absorption_image: np.ndarray of absorbed image
    """
    absorption_image = -np.log(x / I0)
    return absorption_image


def absorption_to_transmission(x, IO) -> np.ndarray:
    """
    Takes an estimate and converts absorption image to transmission image.
    :param:
        x: np.ndarray of 2D image
        I0: float, initial estimated density
    :return:
        transmission: np.ndarray of transmitted image
    """
    transmission_image = np.exp(-x) * I0
    return transmission_image


if __name__ == "__main__":
    # source raw files
    input_dir = os.path.join(os.getcwd(), "homework", "hw01", "output", "scan", "raw")
    # create output directory
    output_absorption_dir = os.path.join(
        input_dir.replace("raw", "transformed"), "absorption"
    )
    output_transmission_dir = os.path.join(
        input_dir.replace("raw", "transformed"), "transmission"
    )
    os.makedirs(output_absorption_dir, exist_ok=True)
    os.makedirs(output_transmission_dir, exist_ok=True)
    for tag, f in enumerate(os.listdir(input_dir)):
        filename = os.path.join(input_dir, f)
        image = plt.imread(filename)
        I0 = estimate_initial_density(image)
        absorption_image = transmission_to_absorption(image, I0)
        # save forward, i.e., transmission to absorption
        plt.axis("off")
        plt.imshow(absorption_image, cmap="gray")
        plt.colorbar()
        save_path = os.path.join(output_absorption_dir, f"000{tag + 1}.png")
        plt.savefig(save_path, transparent=True)
        plt.close()
        # save inverse, i.e., absorption to transmission
        plt.axis("off")
        transmission_image = absorption_to_transmission(absorption_image, I0)
        plt.imshow(transmission_image, cmap="gray")
        plt.colorbar()
        save_path = os.path.join(output_transmission_dir, f"000{tag + 1}.png")
        plt.savefig(save_path, transparent=True)
        plt.close()
