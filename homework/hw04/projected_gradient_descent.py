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

    def optimize(self, num_iterations=100) -> None:
        x = self.x0
        for i in range(num_iterations):
            gradient = self.calculate_gradient(x)
            x = self.project(x, gradient)
        return x


def main():
    descent = ProjectedGradientDescent()
    output = descent.optimize()
    os.makedirs("images", exist_ok=True)
    plt.imshow(output, cmap="gray")
    plt.axis("off")
    plt.savefig("images/proj_grad_descent.tif", transparent=True)


if __name__ == "__main__":
    main()
