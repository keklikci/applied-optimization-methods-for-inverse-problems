# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 12.07.2023                   #
# ********************************** #

import numpy as np


def mask(image, radius=170) -> np.ndarray:
    height, width = image.shape
    blacked_out_image = np.zeros_like(image)
    center_x = width // 2
    center_y = height // 2
    y_indices, x_indices = np.ogrid[:height, :width]
    mask = (x_indices - center_x) ** 2 + (y_indices - center_y) ** 2 <= radius**2
    blacked_out_image[mask] = 1
    masked = blacked_out_image * image
    return masked
