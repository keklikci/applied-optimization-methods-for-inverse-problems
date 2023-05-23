# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 22.05.2023                   #
# ********************************** #

import numpy as np

try:
    import aomip
except:
    import sys

    sys.path.append(os.getcwd())
    import aomip


class ConjugateGradient:
    def __init__(self, n_iterations: int = 10) -> None:
        self.thetas = np.arange(360)
        self.volume_shape = [128, 128]
        self.sinogram_shape = [128]
        self.d2c = self.volume_shape[0] * 100.0
        self.c2d = self.volume_shape[0] * 5.0
        self.operator = self.__create_operator()
        self.phantom = self.__create_phantom()
        self.sinogram = self.__create_sinogram()
        self.n_iterations = n_iterations

    def __str__(self) -> str:
        return "Conjugate Gradient"

    def __create_phantom(self) -> type:
        phantom = aomip.shepp_logan(self.volume_shape)
        return phantom

    def __create_operator(self) -> type:
        operator = aomip.XrayOperator(
            self.volume_shape, self.sinogram_shape, self.thetas, self.d2c, self.c2d
        )
        return operator

    def __create_sinogram(self) -> type:
        sinogram = aomip.radon(
            self.phantom, self.sinogram_shape, self.thetas, self.d2c, self.c2d
        )
        return sinogram

    def optimize(self) -> np.ndarray:
        x = np.zeros(self.volume_shape).flatten()
        op = self.operator.T @ self.operator
        b = self.operator.T @ self.sinogram.flatten()
        r = (b - op @ x).flatten()
        p = (b - op @ x).flatten()
        for i in range(self.n_iterations):
            alpha = np.dot(r, r) / np.dot(p, op @ p)
            x += alpha * p
            r_updated = r - alpha * op @ p
            beta = np.dot(r_updated, r_updated) / np.dot(r, r)
            p = r_updated + beta * p
            r = r_updated
        x = x.reshape(self.volume_shape)
        return x
