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


class FPGM(aomip.Optimization):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lmbd = 1.0
        self.f = aomip.L1()

    def optimize(self, N=100) -> np.ndarray:
        x, z = self.x0, self.x0
        t = 1.0
        for i in range(n):
            xprev, zprev = x, z
            gradient = self.calculate_gradient(z)
            tprev = t
            t = (1 + np.sqrt(1 + 4 * t**2)) / 2
            self.lmbd = (tprev - 1) / t
            z = self.f.proximal(xprev - self.lmbd * gradient, lmbd=self.lmbd)
            x = z + self.lmbd * (z - zprev)
        return x


def main():
    fpgm = FPGM()
    os.makedirs("images/notebook/fpgm", exist_ok=True)
    os.makedirs("images/pogm", exist_ok=True)
    lambdas = np.logspace(-3, 6, 10)
    for lmbd in lambdas:
        fpgm.lmbd = lmbd
        x = fpgm.optimize()
        # save notebook output
        plt.axis("off")
        export = plt.imshow(x, cmap="gray")
        plt.colorbar(export)
        plt.tight_layout()
        plt.title(f"λ ={lmbd}")
        plt.savefig(f"images/notebook/fpgm/lambda_{lmbd}.png")
        plt.clf()
        # save tif output
        tifffile.imwrite(f"images/fpgm/lambda_{lmbd}.tif", x)


if __name__ == "__main__":
    main()
