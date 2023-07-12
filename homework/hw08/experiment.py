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
import time

# square summable step size export paths
ss_notebook_path = "images/notebook/subgradient/square_summable"
ss_convergence_path = "images/notebook/subgradient/convergence/square_summamble"
ss_tifffile_path = "images/subgradient/square_summable"

# nonsummable diminishing step size export paths
diminishing_notebook_path = "images/notebook/subgradient/diminishing"
diminishing_convergence_path = "images/notebook/subgradient/convergence/diminishing"
diminishing_tifffile_path = "images/subgradient/diminishing"

# admm comparison export path
comparison_notebook_path = "images/notebook/subgradient/comparison"
comparison_tifffile_path = "images/subgradient/comparison"

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

def admm_subgradient():
    os.makedirs(comparison_tifffile_path, exist_ok=True)
    os.makedirs(comparison_notebook_path, exist_ok=True)
    start_time = time.time()
    admm = aomip.ADMM()
    admm.f = aomip.L11()
    admm.g = aomip.L2()
    ax, ahistory = admm.optimize()
    print(f"ADMM complete, execution time = {time.time() - start_time:.2f}")
    start_time = time.time()
    subgradient = aomip.Subgradient()
    sx, shistory = subgradient.optimize(lr=1e-3)
    print(f"Subgradient complete, execution time = {time.time() - start_time:.2f}")
    fig, axes = plt.subplots(2, 2)
    # subgradient notebook output
    export = axes[0, 0].imshow(ax, cmap="gray")
    plt.colorbar(export, ax=axes[0, 0])
    axes[0, 0].set_title("Variable Splitting (ADMM)")
    axes[1, 0].plot(ahistory)
    export = axes[0, 1].imshow(sx, cmap="gray")
    plt.colorbar(export, ax=axes[0, 1])
    axes[0, 1].set_title("Subgradient")
    axes[1, 1].plot(shistory)
    plt.savefig(f"{comparison_notebook_path}/comparison.png")
    # save tif output
    tifffile.imwrite(f"{comparison_tifffile_path}/admm.tif", ax)
    tifffile.imwrite(f"{comparison_tifffile_path}/subgradient.tif", ax)

def main():
    print("Running square summable step sizes...")
    square_summable()
    print("Running nonsummable but diminishing step sizes...")
    nonsummable_diminishing()
    print("Running analysis on ADMM and subgradient optimization...")
    admm_subgradient()

if __name__ == "__main__":
    main()
