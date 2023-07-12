# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 12.07.2023                   #
# ********************************** #

import numpy as np
import tifffile
import matplotlib.pyplot as plt
import os
import aomip

def main():
    os.makedirs("images/notebook/gd", exist_ok=True)
    os.makedirs("images/gd", exist_ok=True)
    gd = aomip.GradientDescent()
    lrs = [1e-7, 1e-8, 1e-6, 1e-5]
    for lr in lrs:
        x = gd.optimize(lr=lr)
        # save notebook output
        plt.axis("off")
        export = plt.imshow(x, cmap="gray")
        plt.colorbar(export)
        plt.tight_layout()
        plt.title(f"α ={lr}")
        plt.savefig(f"images/notebook/gd/lr_{lr}.png")
        plt.clf()
        # save tif output
        tifffile.imwrite(f"images/gd/lr_{lr}.tif", x)

if __name__ == "__main__":
    main()
