# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 22.06.2023                   #
# ********************************** #

import aomip
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import os
from optimize import Optimization
from proximal_operators.nonnegativity import Nonnegativity


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

    def optimize(self, alpha=1e-5, num_iterations=100, callback=None) -> None:
        x = self.x0
        for i in range(num_iterations):
            gradient = self.calculate_gradient(x)
            x = Nonnegativity().proximal(x, alpha, gradient)
            if callback is not None and i % 2 == 0:
                error = self.calculate_norm(x)
                callback.append(error)
        return x, callback


def main():
    descent = ProjectedGradientDescent()
    alphas = np.linspace(1e-10, 1e-6, num=5)
    for i, alpha in enumerate(alphas):
        callback = []
        x, callback = descent.optimize(alpha=alpha, callback=callback)
        os.makedirs("images", exist_ok=True)
        # tifffile.imsave(f"images/projected_gradient_descent_proximal_{i + 1}.tif", x.astype(np.uint8))
        plt.imshow(x, cmap="gray")
        plt.savefig(f"images/projected_gradient_descent_proximal_{i + 1}.tif", transparent=True)
        plt.clf()
        plt.plot(callback)
        plt.ylabel(f"Reconstruction error, alpha = {alpha}")
        plt.xlabel(f"# of iterations")
        plt.savefig(f"images/projected_gradient_descent_proximal_callback_{i + 1}")
        plt.clf()


if __name__ == "__main__":
    main()
