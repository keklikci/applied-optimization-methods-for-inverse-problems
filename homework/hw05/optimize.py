# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 14.06.2023                   #
# ********************************** #

import aomip
import numpy as np
import tifffile
from abc import ABC, abstractmethod


class Optimization(object):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vol_shape = [512, 512]
        self.sino_shape = [512]
        self.d2c = self.vol_shape[0] * 100.0
        self.c2d = self.vol_shape[0] * 5.0
        self.thetas = np.arange(360)
        self.x0 = np.zeros(self.vol_shape)
        self.operator = aomip.XrayOperator(
            self.vol_shape, self.sino_shape, self.thetas, self.d2c, self.c2d
        )
        self.target = tifffile.imread("images/htc2022_05c_recon.tif")
        self.sino = aomip.radon(
            self.target, self.sino_shape, self.thetas, self.d2c, self.c2d
        )
        # apply ram-lak filter
        self.sino = self.apply_filter()

    @property
    def operator(self) -> aomip.XrayOperator:
        return self._operator

    @operator.setter
    def operator(self, operator) -> None:
        self._operator = operator

    def calculate_error(self, x) -> float:
        error = self.operator.apply(x) - self.sino
        return error

    def calculate_norm(self, x) -> float:
        error = self.calculate_error(x)
        return np.linalg.norm(error)

    def calculate_gradient(self, x) -> float:
        error = self.calculate_error(x)
        gradient = self.operator.applyAdjoint(error)
        return gradient

    def decay(self, step, beta) -> float:
        step *= beta
        return step

    def apply_filter(self) -> np.ndarray:
        H = np.linspace(-1, 1, self.sino_shape[0])
        ram_lak = np.abs(H)
        h = np.tile(ram_lak, (360, 1)).T
        fftsino = np.fft.fft(self.sino, axis=0)
        projection = np.fft.fftshift(fftsino, axes=1) * np.fft.fftshift(h, axes=0)
        sino = np.real(np.fft.ifft(np.fft.ifftshift(projection, axes=1), axis=0))
        return sino

    @abstractmethod
    def optimize(self) -> float:
        raise NotImplementedError
