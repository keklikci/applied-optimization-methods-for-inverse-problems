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


def main():
    pogm = aomip.POGM()
    os.makedirs("images/notebook", exist_ok=True)
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
        plt.savefig(f"images/notebook/lambda_{lmbd}.png")
        plt.clf()
        # save tif output
        tifffile.imwrite(f"images/pogm/lambda_{lmbd}.tif", x)


if __name__ == "__main__":
    main()
