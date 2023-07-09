# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 29.06.2023                   #
# ********************************** #

import numpy as np
import tifffile
import matplotlib.pyplot as plt
import os
import aomip


class POGM(aomip.Optimization):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lmbd = 1.0
        self.f = aomip.L1()

    def optimize(self, n=100) -> np.ndarray:
        w, x, z = self.x0, self.x0, self.x0
        theta, gamma = 1.0, 1.0
        for k in range(n):
            wprev, xprev, zprev = w, x, z
            thetaprev, gammaprev = theta, gamma
            gradient = self.calculate_gradient(z)
            L = np.linalg.norm(gradient, ord=2) ** 2
            self.lmbd = 1.0 / L
            if 2 <= k < num_iterations - 1:
                theta = 0.5 * (1 + np.sqrt(4 * thetaprev ** 2 + 1))
            if k == num_iterations - 1:
                theta = 0.5 * (1 + np.sqrt(8 * theta ** 2 + 1))
            gamma = self.lmbd * (2 * thetaprev + theta - 1) / theta
            w = xprev - self.lmbd * gradient
            nesterov = (thetaprev - 1) / theta * (w * wprev)
            ogm = thetaprev / theta * (w - xprev)
            pogm = (thetaprev - 1) / (L * gammaprev * theta) * (zprev - xprev)
            z = w + nesterov + ogm + pogm
            # update
            z -= self.lmbd * gradient
            x = self.f.proximal(x, self.lmbd)
        return x

def main():
    pogm = POGM()
    x = pogm.optimize()
    os.makedirs("images", exist_ok=True)
    lambdas = np.logspace(-3, 6, 10)
    for lmbd in lambdas:
        pogm.lmbd = lmbd
        x = admm.optimize()
        plt.imshow(x, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"images/lambda_{tau}.tif", transparent=True)


if __name__ == "__main__":
    main()
