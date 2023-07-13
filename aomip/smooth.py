# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 06.07.2023                   #
# ********************************** #

import numpy as np


def smooth(N):
    I, J = np.meshgrid(np.arange(N), np.arange(N))
    sigma = 0.25 * N
    c = np.array(
        [[0.6 * N, 0.6 * N], [0.5 * N, 0.3 * N], [0.2 * N, 0.7 * N], [0.8 * N, 0.2 * N]]
    )
    a = np.array([1, 0.5, 0.7, 0.9])
    img = np.zeros((N, N))
    for i in range(4):
        term1 = (I - c[i, 0]) ** 2 / (1.2 * sigma) ** 2
        term2 = (J - c[i, 1]) ** 2 / sigma**2
        img += a[i] * np.exp(-term1 - term2)
    return img
