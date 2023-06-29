# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 29.06.2023                   #
# ********************************** #

import aomip
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import os
from optimize import Optimization
from proximal_operators.nonnegativity import Nonnegativity


class POGM(Optimization):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.step = 1e-3

    @property
    def step(self) -> float:
        return self._step

    @step.setter
    def step(self, step) -> float:
        self._step = step

    def optimize(self, num_iterations=100, callback=None) -> None:
        w, x, z = self.x0, self.x0, self.x0
        theta, gamma = 1, 1
        for k in range(num_iterations):
            wprev, xprev, zprev = w, x, z
            thetaprev, gammaprev = theta, gamma
            gradient = self.calculate_gradient(z)
            L = np.linalg.norm(gradient, ord=2) ** 2
            self.step = 1.0 / L
            if 2 <= k < num_iterations - 1:
                theta = 0.5 * (1 + np.sqrt(4 * thetaprev**2 + 1))
            if k == num_iterations - 1:
                theta = 0.5 * (1 + np.sqrt(8 * theta**2 + 1))
            gamma = self.step * (2 * thetaprev + theta - 1) / theta
            w = xprev - self.step * gradient
            nesterov = (thetaprev - 1) / theta * (w * wprev)
            ogm = thetaprev / theta * (w - xprev)
            pogm = (thetaprev - 1) / (L * gammaprev * theta) * (zprev - xprev)
            z = w + nesterov + ogm + pogm
            x = Nonnegativity().proximal(z, gamma, gradient)
            if callback is not None and k % 2 == 0:
                error = self.calculate_norm(x)
                callback.append(error)
        return x, callback


def main():
    factor = 1000
    pogm = POGM()
    callback = []
    x, callback = pogm.optimize(callback=callback)
    # rescale by the factor
    x *= factor
    os.makedirs("images", exist_ok=True)
    # plot the callback
    plt.plot(callback)
    plt.ylabel(f"Reconstruction error")
    plt.xlabel(f"# of iterations")
    plt.savefig(f"images/convergence.png")
    tifffile.imwrite(f"images/pogm.tif", x)


if __name__ == "__main__":
    main()
