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