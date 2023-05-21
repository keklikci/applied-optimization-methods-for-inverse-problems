# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 21.05.2023                   #
# ********************************** #

import numpy as np

class Noise:
    """
    Sample and add noise to an image/signal

    Parameters
    ----------
    method : :obj:`str`
        Type of noise to sample and add to the image
    """

    def __init__(self, method):
        self.validity = {"Poisson", "Gaussian", "Salt-Pepper"}
        if self.__isinstance(method) and self.__isvalid(method):
            method = str.capitalize(method)
            self.method = method

    def __isinstance(self, method):
        if not isinstance(method, str):
            raise TypeError("Method must be of type str!")
        return True

    def __isvalid(self, method):
        if not method in self.validity:
            raise ValueError(f"Method must be an element of {self.validity}!")

    def __str__(self):
        return self.method

    def update(self, method):
        self.method = method

    def transform(self, image):
        noise, noise_shape = None, image.shape
        if self.method == "Gaussian":
            noise = np.random.normal(loc = 0., scale = 0.1, size = noise_shape)
            return image + noise
        elif self.method == "Poisson":
            image = np.random.poisson(image)
            image = np.clip(a = image, a_min = 0, a_max = 255).astype(np.uint8)
            return image
        else:
            noise = np.random.randint(low = 0, high = 2, size = image.shape)
            img = image.copy()
            img[noise == 0] = 0
            img[noise == 1] = 1
            return img