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

# square summable step size export paths
ss_notebook_path = "images/notebook/subgradient/square_summable"
ss_convergence_path = "images/notebook/subgradient/convergence/square_summamble"
ss_tifffile_path = "images/subgradient/square_summable"

# nonsummable diminishing step size export paths
diminishing_notebook_path = "images/notebook/subgradient/diminishing"
diminishing_convergence_path = "images/notebook/subgradient/convergence/diminishing"
diminishing_tifffile_path = "images/subgradient/diminishing"

def square_summable():
    os.makedirs(ss_notebook_path, exist_ok=True)
    os.makedirs(ss_convergence_path, exist_ok=True)
    os.makedirs(ss_tifffile_path, exist_ok=True)
    subgradient = aomip.Subgradient()
    x, history = subgradient.optimize(step="square_summable")
    # save notebook output
    plt.axis("off")
    export = plt.imshow(x, cmap="gray")
    plt.colorbar(export)
    plt.title(f"Square Summable α Values")
    plt.savefig(f"{ss_notebook_path}/square_summable.png")
    plt.clf()
    plt.plot(history)
    plt.title(f"Convergence Analysis, Square Summable Step Sizes")
    plt.xlabel("# iterations")
    plt.ylabel("Loss")
    plt.savefig(f"{ss_convergence_path}/convergence.png")
    plt.clf()
    # save tif output
    tifffile.imwrite(f"{ss_tifffile_path}/square_summable.tif", x)

def nonsummable_diminishing():
    os.makedirs(diminishing_notebook_path, exist_ok=True)
    os.makedirs(diminishing_convergence_path, exist_ok=True)
    os.makedirs(diminishing_tifffile_path, exist_ok=True)
    subgradient = aomip.Subgradient()
    x, history = subgradient.optimize(step="diminishing")
    # save notebook output
    plt.axis("off")
    export = plt.imshow(x, cmap="gray")
    plt.colorbar(export)
    plt.title(f"Nonsummable Diminishing α Values")
    plt.savefig(f"{diminishing_notebook_path}/nonsummable_diminishing.png")
    plt.clf()
    plt.plot(history)
    plt.title(f"Convergence Analysis, Nonsummable Diminishing Step Sizes")
    plt.xlabel("# iterations")
    plt.ylabel("Loss")
    plt.savefig(f"{diminishing_convergence_path}/convergence.png")
    plt.clf()
    # save tif output
    tifffile.imwrite(f"{diminishing_tifffile_path}/nonsummable_diminishing.tif", x)

def main():
    print("Running square summable step sizes...")
    square_summable()
    print("Running nonsummable but diminishing step sizes...")
    nonsummable_diminishing()

if __name__ == "__main__":
    main()