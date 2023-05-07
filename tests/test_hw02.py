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
    volume = []
    for projection in projections:
        image = Image.open(projection)
        image = np.array(image)
        volume = aomip.slicing.stack_slice(volume, image, image.shape[0] // 2)
    volume = np.array(volume, dtype = np.uint16)
    sinogram = aomip.radon(volume, volume.shape, np.linspace(0, 180, max(volume.shape), endpoint = False), 1000, 150)
    plt.imshow(sinogram, cmap = "gray")
    os.makedirs(output_path, exist_ok = True)
    plt.savefig(os.path.join(output_path, "sinogram.png"), transparent = True)

def main():
    dataset_path = "/srv/ceph/share-all/aomip/6983008_seashell/"
    output_path = "homework/hw02/output/"
    slicing_path = os.path.join(output_path, "slicing")
    projections = os.listdir(dataset_path)
    test_slicing(projections = [os.path.join(dataset_path, projection) for projection in projections], output_path = slicing_path)

if __name__ == "__main__":
    main()