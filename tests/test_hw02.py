# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 07.05.2023                   #
# ********************************** #

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import os
try:
    import aomip
except:
    import sys
    sys.path.append(os.getcwd())

def test_slicing(filenames: list, output_path: str) -> None:
    projections = []
    for filename in filenames:
        projection = tifffile.imread(filename)
        projections.append(projection)
    sinogram = aomip.slicing(projections, projection.shape[0] // 2 - 1)
    plt.figure(figsize = (8, 10))
    plt.title(f"Slice of sinogram, slice index = {projection.shape[0] // 2 - 1}")
    plt.imshow(sinogram, cmap = "gray")
    os.makedirs(output_path, exist_ok = True)
    plt.savefig(os.path.join(output_path, "sinogram.png"), transparent = True)

def main():
    dataset_path = "/srv/ceph/share-all/aomip/6983008_seashell/"
    output_path = "homework/hw02/output/"
    slicing_path = os.path.join(output_path, "slicing")
    filenames = os.listdir(dataset_path)
    test_slicing(filenames = [os.path.join(dataset_path, filename) for filename in filenames], output_path = slicing_path)

if __name__ == "__main__":
    main()
