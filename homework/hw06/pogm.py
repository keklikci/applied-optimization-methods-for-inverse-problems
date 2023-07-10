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
            if 2 <= k < n - 1:
                theta = 0.5 * (1 + np.sqrt(4 * thetaprev ** 2 + 1))
            if k == n - 1:
                theta = 0.5 * (1 + np.sqrt(8 * theta ** 2 + 1))
            gamma = 1 / L * (2 * thetaprev + theta - 1) / theta
            w = xprev - 1 / L * gradient
            nesterov = (thetaprev - 1) / theta * (w * wprev)
            ogm = thetaprev / theta * (w - xprev)
            pogm = (thetaprev - 1) / (L * gammaprev * theta) * (zprev - xprev)
            z = w + nesterov + ogm + pogm
            x = self.f.proximal(z, lmbd=gamma)
        return x

def main():
    pogm = POGM()
    x = pogm.optimize()
    os.makedirs("images/notebook/pogm", exist_ok=True)
    os.makedirs("images/pogm", exist_ok=True)
    lambdas = np.logspace(-3, 6, 10)
    for lmbd in lambdas:
        pogm.lmbd = lmbd
        x = pogm.optimize()
        # scale output
        x *= 1e3
        # save notebook output
        plt.axis("off")
        export = plt.imshow(x, cmap="gray")
        plt.colorbar(export)
        plt.tight_layout()
        plt.title(f"λ ={lmbd}")
        plt.savefig(f"images/notebook/pogm/lambda_{lmbd}.png")
        plt.clf()
        # save tif output
        tifffile.imwrite(f"images/pogm/lambda_{lmbd}.tif", x)


if __name__ == "__main__":
    main()
