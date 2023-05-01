# !/usr/bin/env python
# -*- coding: utf-8 -*-
# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 02.05.2023                   #
# ********************************** #

from phantom import shepp_logan
from sinogram import transform, get_projection_angles
from scipy.fft import fft, ifft
from filters import ramp
import numpy as np
import os
import matplotlib.pyplot as plt

def get_coordinate_system(x: int) -> tuple:
    """
    Creates a coordinate system centered at the origin.
    :param:
        x: int, length of phantom
    :return:
        grid_x: np.ndarray, x-axis
        grid_y: np.ndarray, y-axis
    """
    x = np.arange(x) - x / 2
    grid_x, grid_y = np.meshgrid(x, x)
    return grid_x, grid_y

def backproject(sinogram: np.ndarray, projection_angles: np.ndarray) -> np.ndarray:
    """
    Backprojects the parametrized sinogram.
    :param:
        sinogram: np.ndarray of radon transformed phantom
        projection_angles: np.ndarray of angles used during forward projection
    :return:
        result: backprojected
    """
    sinogram_x = sinogram.shape[0]
    projection = fft(sinogram, axis=0) * ramp(size=sinogram_x)
    sinogram = np.real(ifft(projection, axis=0)[:sinogram_x, :])
    # assumes a square phantom
    result = np.zeros((sinogram_x, sinogram_x))
    # initialize the grid
    grid_x, grid_y = get_coordinate_system(sinogram_x)
    # convert degrees to radians
    projection_angles = np.deg2rad(projection_angles)
    # loop over every projection angle
    for i in range(len(projection_angles)):
        # find point of rotation
        por = grid_x * np.cos(projection_angles[i]) - grid_y * np.sin(projection_angles[i])
        # shift rotated matrix to the phantom coordinates
        shifted = np.floor(por + sinogram_x / 2).astype("int")
        angular_projection = np.zeros((sinogram_x, sinogram_x))
        m0, m1 = np.where((shifted >= 0) & (shifted <= (sinogram_x - 1)))
        projection = sinogram[:, i]
        angular_projection[m0, m1] = projection[shifted[m0, m1]]
        # add up the projections
        result += angular_projection
    return result

if __name__ == "__main__":
    phantom = shepp_logan()
    angles = get_projection_angles(phantom)
    sinogram = transform(phantom, angles)
    result = backproject(sinogram, angles)
    out_dir = os.path.join(os.getcwd(), "homework", "hw01", "output", "phantom", "backprojection")
    os.makedirs(out_dir, exist_ok=True)
    plt.axis("off")
    plt.imshow(result, cmap="gray")
    plt.savefig(out_dir + "/backprojection.png", transparent=True)
    plt.close()
