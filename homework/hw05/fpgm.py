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


class FPGM(Optimization):
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
        x, z = self.x0, self.x0
        t = 1
        for i in range(num_iterations):
            xprev, zprev = x, z
            gradient = self.calculate_gradient(z)
            tprev = t
            t = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
            # calculate momentum parameter 
            self.step = (tprev - 1) / t
            z = Nonnegativity().proximal(xprev, self.step, gradient)
            x = z + self.step * (z - zprev)
            if callback is not None and i % 2 == 0:
                error = self.calculate_norm(x)
                callback.append(error)
        return x, callback


def main():
    fpgm = FPGM()
    alphas = np.linspace(1e-5, 1e-3, num=5)
    callback = []
    for i, alpha in enumerate(alphas):
        fpgm.step = alpha
        x, callback = fpgm.optimize(callback=callback)
        os.makedirs("images", exist_ok=True)
        tifffile.imsave(f"images/fpgm_proximal_{i + 1}.tif", x.astype(np.uint8))
        plt.plot(np.arange(len(callback)), callback)
        plt.ylabel(f"Reconstruction error, alpha = {alpha}")
        plt.xlabel(f"# of iterations")
        plt.savefig(f"images/fpgm_proximal_callback_{i + 1}")


if __name__ == "__main__":
    main()
