# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 07.05.2023                   #
# ********************************** #

import matplotlib.pyplot as plt
import numpy as np
import os
try:
    import aomip
except:
    import sys
    sys.path.append(os.getcwd())
from PIL import Image

def test_slicing(projections: list, output_path: str) -> None:
    volume = np.ndarray([], dtype = np.uint16)
    for projection in projections:
        image = Image.open(projection)
        image = np.array(image)
        volume = aomip.slicing(volume, image, image.shape[0] // 2)
    sinogram = aomip.radon(volume, [180], np.linspace(0, 360, 420), 1000, 150)
    plt.imshow(volume, cmap = "gray")
    plt.savefig(os.path.join(output_path, "sinogram.png"), transparent = True)

def main():
    dataset_path = "/srv/ceph/share-all/aomip/6983008_seashell/"
    output_path = "../homework/hw02/output/"
    slicing_path = os.path.join(output_path, "slicing")
    projections = os.listdir(dataset_path)
    test_slicing(projections = [os.path.join(dataset_path, projection) for projection in projections], output_path = slicing_path)

if __name__ == "__main__":
    main()