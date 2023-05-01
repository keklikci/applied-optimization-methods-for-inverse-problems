# !/usr/bin/env python
# -*- coding: utf-8 -*-
# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 01.05.2023                   #
# ********************************** #

import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from dotenv import load_dotenv
# source configuration file
load_dotenv()

def rotate(image: np.ndarray, offset: int, tag: int, save_path: str) -> None:
    """
    Left-shifts object to match the correct center of rotation.
    :param:
        image: np.ndarray of flat-field corrected image
        offset: shift value, sourced by configuration
        tag: int, file indicator
        save_path: str, path to save object with incorrect / correct center of rotation
    :return:
        None
    """
    xticks = np.arange(0, 972, 108)
    yticks = np.arange(0, 700, 100)
    plt.yticks(yticks)
    plt.xticks(xticks)
    correct_center_x = image.shape[-1] // 2
    correct_center_y = image.shape[0] // 2
    plt.axvline(x = correct_center_x, ymax = 0.5, color = "r")
    plt.axhline(y = correct_center_y, xmax = 0.5, color = "g")
    plt.imshow(image, cmap="gray")
    plt.title("Center of Rotation (Prior Correction)")
    raw_rotated_dir = os.path.join(save_path, "raw")
    os.makedirs(raw_rotated_dir, exist_ok=True)
    plt.savefig(os.path.join(raw_rotated_dir, f"000{tag}.png"))
    plt.close()
    # apply rotational shift alongside the last axis
    last_axis = len(image.shape) - 1
    image = np.roll(image, shift=offset, axis=last_axis)
    plt.yticks(yticks)
    plt.xticks(xticks)
    plt.imshow(image, cmap="gray")
    plt.axvline(x = correct_center_x, ymax = 0.5, color = "g")
    plt.axhline(y = correct_center_y, xmax = 0.5, color = "g")
    plt.title("Center of Rotation (After Correction)")
    corrected_rotated_dir = os.path.join(save_path, "corrected")
    os.makedirs(corrected_rotated_dir, exist_ok=True)
    plt.savefig(os.path.join(corrected_rotated_dir, f"000{tag}.png"))
    plt.close()

if __name__ == "__main__":
    # source lat-field corrected files
    input_dir = os.path.join(os.getcwd(), "homework", "hw01", "output", "scan", "flat_field_corrected")
    # create output directory
    output_dir = input_dir.replace("flat_field_corrected", "rotated")
    os.makedirs(output_dir, exist_ok=True)
    for tag, f in enumerate(os.listdir(input_dir)):
        filename = os.path.join(input_dir, f)
        image = plt.imread(filename)
        offset = int(os.environ["ROTATION_OFFSET"])
        rotate(image, offset, tag + 1, output_dir)