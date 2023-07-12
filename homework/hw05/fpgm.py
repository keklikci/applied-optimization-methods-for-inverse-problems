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
    fpgm = aomip.FPGM()
    os.makedirs("images/notebook/fpgm", exist_ok=True)
    os.makedirs("images/fpgm", exist_ok=True)
    lambdas = np.logspace(-3, 6, 10)
    for lmbd in lambdas:
        fpgm.lmbd = lmbd
        x = fpgm.optimize()
        # scale output
        x *= 1e3
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
