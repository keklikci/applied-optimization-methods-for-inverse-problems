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


class Elastic(Optimization):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lmbd = 1.0
        self.f = aomip.ElasticNet()

    def optimize(self, n=100) -> np.ndarray:
        x, z = self.x0, self.x0
        t = 1
        for _ in range(n):
            xprev, zprev = x, z
            gradient = self.calculate_gradient(z)
            tprev = t
            t = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
            # calculate momentum parameter
            self.lmbd = (tprev - 1) / t
            # enforce a combination of l1 and l2 regularization
            z = self.f.proximal(xprev - 1 / L * gradient, lmbd=(1 / L))
            x = z + self.lmbd * (z - zprev)
        return x


def main():
    elastic = Elastic()
    os.makedirs("images/notebook/elastic", exist_ok=True)
    os.makedirs("images/elastic", exist_ok=True)
    lambdas = np.logspace(-3, 6, 10)
    for lmbd in lambdas:
        elastic.lmbd = lmbd
        x = elastic.optimize()
        # scale output
        x *= 1e3
        # save notebook output
        plt.axis("off")
        export = plt.imshow(x, cmap="gray")
        plt.colorbar(export)
        plt.tight_layout()
        plt.title(f"λ ={lmbd}")
        plt.savefig(f"images/notebook/elastic/lambda_{lmbd}.png")
        plt.clf()
        # save tif output
        tifffile.imwrite(f"images/elastic/lambda_{lmbd}.tif", x)


if __name__ == "__main__":
    main()
