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


def main():
    elastic = aomip.Elastic()
    os.makedirs("images/notebook/elastic", exist_ok=True)
    os.makedirs("images/elastic", exist_ok=True)
    lambdas = np.logspace(-3, 6, 10)
    for lmbd in lambdas:
        elastic.lmbd = lmbd
        x = elastic.optimize()
        # save notebook output
        plt.axis("off")
        export = plt.imshow(x, cmap="gray")
        plt.colorbar(export)
        plt.tight_layout()
        plt.title(f"λ ={lmbd}")
        plt.savefig(f"images/notebook/elastic/lambda_{lmbd}.png")
        plt.clf()
        # save tif output
        tifffile.imwrite(f"images/elastic/lambda_{lmbd}.tif", x)


if __name__ == "__main__":
    main()
