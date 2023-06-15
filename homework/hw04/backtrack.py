# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 14.06.2023                   #
# ********************************** #

import numpy as np
import os
import matplotlib.pyplot as plt
import tifffile
from optimize import Optimization


class Backtracking(Optimization):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def optimize(self, alpha=1e-5, beta=0.9, num_iterations=100, callback=None) -> None:
        x = self.x0
        step = 1.0
        gradient = self.calculate_gradient(x)
        norm = np.linalg.norm(gradient)
        for i in range(num_iterations):
            next_x = x - step * gradient
            error = self.calculate_norm(next_x)
            if callback is not None and i % 2 == 0:
                callback.append(error)
            objective = self.calculate_norm(x)
            terminal_bound = error <= objective - alpha * step * norm
            if terminal_bound:
                break
            else:
                step = self.decay(step, beta)
        # descent with the optimal step
        for i in range(num_iterations):
            x -= step * gradient
        return x, callback


def main():
    backtrack = Backtracking()
    alphas = np.linspace(0.0075, 0.011, num=5)
    callback = []
    for i, alpha in enumerate(alphas):
        x, callback = backtrack.optimize(alpha=alpha, callback=callback)
        os.makedirs("images", exist_ok=True)
        tifffile.imsave(f"images/backtrack_{i + 1}.tif", x.astype(np.uint8))
        plt.plot(np.arange(len(callback)), callback)
        plt.ylabel(f"Reconstruction error, alpha = {alpha}")
        plt.xlabel(f"# of iterations")
        plt.savefig(f"images/backtrack_callback_{i + 1}")


if __name__ == "__main__":
    main()
