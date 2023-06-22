# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 22.06.2023                   #
# ********************************** #

import aomip
import numpy as np
import tifffile
import os
import matplotlib.pyplot as plt
from optimize import Optimization
from proximal_operators.l1 import L1


class ISTA(Optimization):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.step = 1e-3
        self.beta = 1e-6

    @property
    def step(self) -> float:
        return self._step

    @step.setter
    def step(self, step) -> float:
        self._step = step

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, beta) -> float:
        self._beta = beta

    def optimize(self, num_iterations=100, callback=None) -> None:
        x = self.x0
        for i in range(num_iterations):
            gradient = self.calculate_gradient(x)
            x = L1().proximal(x - self.step * gradient, self.beta)
            if callback is not None and i % 2 == 0:
                error = self.calculate_norm(x)
                callback.append(error)
        return x, callback


def main():
    ista = ISTA()
    alphas = np.linspace(1e-5, 1e-3, num=5)
    betas = np.linspace(1e-6, 1e-3, num=5)
    callback = []
    for i, params in enumerate(zip(alphas, betas)):
        ista.step = params[0]
        ista.beta = params[1]
        x, callback = ista.optimize(callback=callback)
        os.makedirs("images", exist_ok=True)
        tifffile.imsave(f"images/ista_proximal_{i + 1}.tif", x.astype(np.uint8))
        plt.plot(np.arange(len(callback)), callback)
        plt.ylabel(f"Reconstruction error, alpha = {params[0]}, beta = {params[1]}")
        plt.xlabel(f"# of iterations")
        plt.savefig(f"images/ista_proximal_callback_{i + 1}")


if __name__ == "__main__":
    main()
