# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 14.06.2023                   #
# ********************************** #

import numpy as np
import tifffile
import matplotlib.pyplot as plt
import os
import aomip

def main():
    bb1 = aomip.BB1()
    os.makedirs("images/notebook/bb1", exist_ok=True)
    os.makedirs("images/bb1", exist_ok=True)
    lambdas = [1e-6, 1e-5, 1e-4, 1e-3]
    for lmbd in lambdas:
        x = bb1.optimize(lmbd=lmbd)
        # save notebook output
        plt.axis("off")
        export = plt.imshow(x, cmap="gray")
        plt.colorbar(export)
        plt.tight_layout()
        plt.title(f"λ ={lmbd}")
        plt.savefig(f"images/notebook/bb1/lambda_{lmbd}.png")
        plt.clf()
        # save tif output
        tifffile.imwrite(f"images/bb1/lambda_{lmbd}.tif", x)


if __name__ == "__main__":
    main()
