# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 14.06.2023                   #
# ********************************** #

import aomip
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import os
from optimize import Optimization


class ProjectedGradientDescent(Optimization):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.step = 1e-5

    @property
    def step(self) -> float:
        return self._step

    @step.setter
    def step(self, step) -> float:
        self._step = step

    def project(self, x, gradient) -> np.ndarray:
        xproj = np.maximum(x - self.step * gradient, 0)
        return xproj

    def optimize(self, alpha=1e-5, num_iterations=100, callback=None) -> None:
        x = self.x0
        self.step = alpha
        for i in range(num_iterations):
            gradient = self.calculate_gradient(x)
            x = self.project(x, gradient)
            if callback is not None and i % 2 == 0:
                error = self.calculate_norm(x)
                callback.append(error)
        return x, callback


def main():
    descent = ProjectedGradientDescent()
    callback = []
    alphas = np.linspace(1-3, 1e-8, num=5)
    for i, alpha in enumerate(alphas):
        x, callback = descent.optimize(alpha=alpha, callback=callback)
        os.makedirs("images", exist_ok=True)
        tifffile.imsave(f"images/proj_grad_descent_{i + 1}.tif", x.astype(np.uint8))
        plt.plot(np.arange(len(callback)), callback)
        plt.ylabel(f"Reconstruction error, alpha = {alpha}")
        plt.xlabel(f"# of iterations")
        plt.savefig(f"images/proj_grad_descent_callback_{i + 1}")


if __name__ == "__main__":
    main()
