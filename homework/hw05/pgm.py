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
from proximal_operators.nonnegativity import Nonnegativity
from proximal_operators.l2 import L2


class PGM(Optimization):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.step = 1e-3

    @property
    def step(self) -> float:
        return self._step

    @step.setter
    def step(self, step) -> float:
        self._step = step
    
    def optimize(self, alpha=1e-3, num_iterations=100, callback=None, variant="nonnegative", beta=None) -> None:
        x, z = self.x0, self.x0
        self.step = alpha
        for i in range(num_iterations):
            xprev, zprev = x, z
            gradient = self.calculate_gradient(z)
            if variant == "l2":
                z = L2().proximal(xprev, self.step, gradient, beta)
                x = z + self.step * (z - zprev)
            else:
                z = Nonnegativity().proximal(xprev, self.step, gradient)
                x = z + self.step * (z - zprev)
            if callback is not None and i % 2 == 0:
                error = self.calculate_norm(x)
                callback.append(error)
        return x, callback


def main():
    pgm = PGM()
    alphas = np.linspace(1e-5, 1e-3, num=5)
    betas = np.linspace(1e-6, 1e-3, num=5)
    variants = ["nonnegative", "l2"]
    for variant in variants:
        for i, params in enumerate(zip(alphas, betas)):
            alpha = params[0]
            beta = params[1]
            callback = []
            pgm.step = alpha
            x, callback = pgm.optimize(callback=callback, variant=variant, beta=beta)
            os.makedirs("images", exist_ok=True)
            # tifffile.imsave(f"images/pgm_proximal_variant_{variant}_{i + 1}.tif", x.astype(np.uint8))
            plt.imshow(x, cmap="gray")
            plt.savefig(f"images/pgm_proximal_{variant}_{i + 1}.tif", transparent=True)
            plt.clf()
            plt.plot(callback)
            plt.ylabel(f"Reconstruction error, alpha = {alpha}, beta = {beta}")
            plt.xlabel(f"# of iterations")
            plt.savefig(f"images/pgm_proximal_{variant}_callback_{i + 1}")
            plt.clf()


if __name__ == "__main__":
    main()
