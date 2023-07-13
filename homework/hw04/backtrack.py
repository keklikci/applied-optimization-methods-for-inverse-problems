# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 14.06.2023                   #
# ********************************** #

import numpy as np
import os
import matplotlib.pyplot as plt
import tifffile
import aomip


class Backtracking(aomip.Optimization):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def optimize(self, alpha=1e-3, beta=0.9, n=100) -> np.ndarray:
        x = self.x0
        step = 1.0
        gradient = self.calculate_gradient(x)
        norm = np.linalg.norm(gradient)
        for _ in range(n):
            next_x = x - step * gradient
            error = self.calculate_norm(next_x)
            objective = self.calculate_norm(x)
            terminal_bound = error <= objective - alpha * step * norm
            if terminal_bound:
                break
            else:
                step = self.decay(step, beta)
        # descent with optimal step
        for _ in range(n):
            x -= step * gradient
        return x


def main():
    backtrack = Backtracking()
    os.makedirs("images/notebook/backtrack", exist_ok=True)
    os.makedirs("images/backtrack", exist_ok=True)
    alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    for alpha in alphas:
        x = backtrack.optimize(alpha=alpha)
        # save notebook output
        plt.axis("off")
        export = plt.imshow(x, cmap="gray")
        plt.colorbar(export)
        plt.tight_layout()
        plt.title(f"α={alpha}")
        plt.savefig(f"images/notebook/backtrack/alpha_{alpha}_beta_{0.9}.png")
        plt.clf()
        # save tif output
        tifffile.imwrite(f"images/backtrack/alpha_{alpha}_beta_{0.9}.tif", x)


if __name__ == "__main__":
    main()
