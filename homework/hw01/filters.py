# !/usr/bin/env python
# -*- coding: utf-8 -*-
# ******************************************************************************************************* #
# Author: kaanguney.keklikci@tum.de                                                                       #
# Date: 01.05.2023                                                                                        #
# Reference: https://github.com/scikit-image/scikit-image/blob/main/skimage/transform/radon_transform.py  #
# ******************************************************************************************************* #

import numpy as np
from scipy.fft import fft, ifft, fftshift, fftfreq
from phantom import shepp_logan


def add_axis(image_filter: np.ndarray) -> np.ndarray:
    return image_filter[:, np.newaxis]


def ramp(size: int) -> np.ndarray:
    n = np.concatenate(
        (
            np.arange(1, size / 2 + 1, 2, dtype=int),
            np.arange(size / 2 - 1, 0, -2, dtype=int),
        )
    )
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2
    fourier = 2 * np.real(fft(f))
    return fourier[:, np.newaxis]


def cosine(size: int) -> np.ndarray:
    fourier_filter = ramp().reshape(
        -1,
    )
    freq = np.linspace(0, np.pi, size, endpoint=False)
    fourier_filter *= fftshift(np.sin(freq))
    return fourier_filter[:, np.newaxis]


def shepp_logan(size: int) -> np.ndarray:
    fourier_filter = ramp().reshape(
        -1,
    )
    omega = np.pi * fftfreq(size)[1:]
    fourier_filter[1:] *= np.sin(omega) / omega
    return fourier_filter[:, np.newaxis]
