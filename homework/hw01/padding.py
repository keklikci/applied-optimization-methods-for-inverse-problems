# !/usr/bin/env python
# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 01.05.2023                   #
# ********************************** #

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from dotenv import load_dotenv
# source configuration file
load_dotenv()

def apply_padding(image: np.ndarray, amount: int) -> np.ndarray:
    """
    Zero-pads given matrix by the amount, i.e., offset.
    :param:
        image: np.ndarray of 2D image
        amount: int, padding offset
    :return:
        image: np.ndarray of 2D padded image 
    """
    # padd on all sides of the projection
    image = np.pad(image, pad_width=amount)
    return image

if __name__ == "__main__":
    # source raw files
    input_dir = os.path.join(os.getcwd(), "homework", "hw01", "output", "scan", "raw")
    # create output directory
    output_dir = input_dir.replace("raw", "padded")
    os.makedirs(output_dir, exist_ok=True)
    for tag, f in enumerate(os.listdir(input_dir)):
        filename = os.path.join(input_dir, f)
        image = plt.imread(filename)
        offset = int(os.environ["PADDING_AMOUNT"])
        padded_image = apply_padding(image, offset)
        plt.axis("off")
        plt.imshow(padded_image, cmap="gray")
        save_path = os.path.join(output_dir, f"000{tag + 1}.png")
        plt.savefig(save_path, transparent=True)
        plt.close()