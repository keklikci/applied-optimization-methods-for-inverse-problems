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
    os.makedirs("images/notebook/subgradient/constant", exist_ok=True)
    os.makedirs("images/notebook/subgradient/convergence/constant", exist_ok=True)
    os.makedirs("images/subgradient/constant", exist_ok=True)
    lrs = [1e-3, 1e-4, 1e-5]
    subgradient = aomip.Subgradient()
    for lr in lrs:
        x, history = subgradient.optimize(lr=lr)
        # save notebook output
        plt.axis("off")
        export = plt.imshow(x, cmap="gray")
        plt.colorbar(export)
        plt.title(f"α ={lr}")
        plt.savefig(f"images/notebook/subgradient/constant/lr_{lr}.png")
        plt.clf()
        plt.plot(history)
        plt.title(f"Convergence Analysis, Constant Step Size = {lr}")
        plt.xlabel("# iterations")
        plt.ylabel("Loss")
        plt.savefig(f"images/notebook/subgradient/convergence/constant/convergence_{lr}.png")
        plt.clf()
        # save tif output
        tifffile.imwrite(f"images/subgradient/constant/lr_{lr}.tif", x)

if __name__ == "__main__":
    main()
