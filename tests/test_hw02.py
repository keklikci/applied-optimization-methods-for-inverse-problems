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
    import aomip

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

def test_gradient_descent_opt(output_path: str, learning_rate: float = 0.0005, n_iterations: int = 300_000) -> None:
    volume_shape = [128, 128]
    sinogram_shape = [128]
    d2c = volume_shape[0] * 100.0
    c2d = volume_shape[0] * 5.0
    thetas = np.arange(360)
    phantom = aomip.shepp_logan(volume_shape)
    sinogram = aomip.radon(phantom, sinogram_shape, thetas, d2c, c2d)
    x = np.zeros(volume_shape)
    operator = aomip.XrayOperator(volume_shape, sinogram_shape, thetas, volume_shape[0] * 100.0, volume_shape[0] * 5.0)
    for i in range(n_iterations):
        if i % 100_000 == 0:
            if i == 0:
                print("Optimizing via gradient descent..")
            else:
                print(f"Processed {i} iterations.") 
        gradient = operator.applyAdjoint(operator.apply(x) - sinogram)
        x -= learning_rate * gradient
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
    ax1.set_title("Original")
    ax1.imshow(phantom, cmap = "gray")
    ax2.set_title(f"Reconstructed")
    ax2.imshow(x, cmap = "gray")
    os.makedirs(output_path, exist_ok = True)
    plt.savefig(os.path.join(output_path, "least_squares.png"), transparent = True)

def test_gradient_descent_opt(output_path: str, learning_rate: float = 0.0005, n_iterations: int = 300_000) -> None:
    volume_shape = [128, 128]
    sinogram_shape = [128]
    d2c = volume_shape[0] * 100.0
    c2d = volume_shape[0] * 5.0
    thetas = np.arange(360)
    phantom = aomip.shepp_logan(volume_shape)
    sinogram = aomip.radon(phantom, sinogram_shape, thetas, d2c, c2d)
    x = np.zeros(volume_shape)
    operator = aomip.XrayOperator(volume_shape, sinogram_shape, thetas, volume_shape[0] * 100.0, volume_shape[0] * 5.0)
    for i in range(n_iterations):
        if i % 100_000 == 0:
            if i == 0:
                print("Optimizing via gradient descent..")
            else:
                print(f"Processed {i} iterations.") 
        gradient = operator.applyAdjoint(operator.apply(x) - sinogram)
        x -= learning_rate * gradient
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
    ax1.set_title("Original")
    ax1.imshow(phantom, cmap = "gray")
    ax2.set_title(f"Reconstructed")
    ax2.imshow(x, cmap = "gray")
    os.makedirs(output_path, exist_ok = True)
    plt.savefig(os.path.join(output_path, "least_squares.png"), transparent = True)

def test_tikhonov_opt(output_path: str, learning_rate: float = 0.0005, n_iterations: int = 300_000) -> None:
    volume_shape = [128, 128]
    sinogram_shape = [128]
    d2c = volume_shape[0] * 100.0
    c2d = volume_shape[0] * 5.0
    thetas = np.arange(360)
    phantom = aomip.shepp_logan(volume_shape)
    sinogram = aomip.radon(phantom, sinogram_shape, thetas, d2c, c2d)
    x = np.zeros(volume_shape)
    operator = aomip.XrayOperator(volume_shape, sinogram_shape, thetas, volume_shape[0] * 100.0, volume_shape[0] * 5.0)
    for i in range(n_iterations):
        if i % 100_000 == 0:
            if i == 0:
                print("Optimizing via Tikhonov regularization..")
            else:
                print(f"Processed {i} iterations.") 
        # add x-dependent term to the gradient
        gradient = operator.applyAdjoint(operator.apply(x) - sinogram) + sinogram * x
        x -= learning_rate * gradient
    plt.imshow(x, cmap = "gray")
    os.makedirs(output_path, exist_ok = True)
    # reserved for submission, hence .tif extension
    plt.savefig(os.path.join(output_path, "tikhonov_regularization.tif"), transparent = True)

def test_huber_functional_opt(output_path: str, learning_rate: float = 0.0005, n_iterations: int = 100_000, delta: float = 0.0001) -> None:
    volume_shape = [128, 128]
    sinogram_shape = [128]
    d2c = volume_shape[0] * 100.0
    c2d = volume_shape[0] * 5.0
    thetas = np.arange(360)
    phantom = aomip.shepp_logan(volume_shape)
    sinogram = aomip.radon(phantom, sinogram_shape, thetas, d2c, c2d)
    x = np.zeros(volume_shape)
    operator = aomip.XrayOperator(volume_shape, sinogram_shape, thetas, volume_shape[0] * 100.0, volume_shape[0] * 5.0)
    for i in range(n_iterations):
        if i % 100_000 == 0:
            if i == 0:
                print("Optimizing via Huber functional...")
            else:
                print(f"Processed {i} iterations.") 
        gradient = operator.applyAdjoint(operator.apply(x) - sinogram)
        # add term to the gradient
        if np.linalg.norm(x, 1) < delta:
            gradient += x
        else:
            gradient += delta * np.sign(x)
        x -= learning_rate * gradient
    plt.imshow(x, cmap = "gray")
    os.makedirs(output_path, exist_ok = True)
    plt.savefig(os.path.join(output_path, "huber_functional.png"), transparent = True)

def test_fair_potential_opt(output_path: str, learning_rate: float = 0.0005, n_iterations: int = 100_000, delta: float = 0.0001) -> None:
    volume_shape = [128, 128]
    sinogram_shape = [128]
    d2c = volume_shape[0] * 100.0
    c2d = volume_shape[0] * 5.0
    thetas = np.arange(360)
    phantom = aomip.shepp_logan(volume_shape)
    sinogram = aomip.radon(phantom, sinogram_shape, thetas, d2c, c2d)
    x = np.zeros(volume_shape)
    operator = aomip.XrayOperator(volume_shape, sinogram_shape, thetas, volume_shape[0] * 100.0, volume_shape[0] * 5.0)
    for i in range(n_iterations):
        if i % 100_000 == 0:
            if i == 0:
                print("Optimizing via Huber functional...")
            else:
                print(f"Processed {i} iterations.") 
        # add term to the gradient
        gradient = operator.applyAdjoint(operator.apply(x) - sinogram) + (x / (1 + x) / delta)
        x -= learning_rate * gradient
    plt.imshow(x, cmap = "gray")
    os.makedirs(output_path, exist_ok = True)
    plt.savefig(os.path.join(output_path, "fair_potential.png"), transparent = True)

def test_gradient_descent_opt(output_path: str, learning_rate: float = 0.0005, n_iterations: int = 300_000) -> None:
    volume_shape = [128, 128]
    sinogram_shape = [128]
    d2c = volume_shape[0] * 100.0
    c2d = volume_shape[0] * 5.0
    thetas = np.arange(360)
    phantom = aomip.shepp_logan(volume_shape)
    sinogram = aomip.radon(phantom, sinogram_shape, thetas, d2c, c2d)
    x = np.zeros(volume_shape)
    operator = aomip.XrayOperator(volume_shape, sinogram_shape, thetas, volume_shape[0] * 100.0, volume_shape[0] * 5.0)
    for i in range(n_iterations):
        if i % 100_000 == 0:
            if i == 0:
                print("Optimizing via gradient descent..")
            else:
                print(f"Processed {i} iterations.") 
        gradient = operator.applyAdjoint(operator.apply(x) - sinogram)
        x -= learning_rate * gradient
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
    ax1.set_title("Original")
    ax1.imshow(phantom, cmap = "gray")
    ax2.set_title(f"Reconstructed")
    ax2.imshow(x, cmap = "gray")
    os.makedirs(output_path, exist_ok = True)
    plt.savefig(os.path.join(output_path, "least_squares.png"), transparent = True)

def main():
    dataset_path = "/srv/ceph/share-all/aomip/6983008_seashell/"
    output_path = "homework/hw02/output/"
    slicing_path = os.path.join(output_path, "slicing")
    filenames = [os.path.join(dataset_path, filename) for filename in os.listdir(dataset_path)]
    test_slicing(filenames = filenames, output_path = slicing_path)
    least_squares_path = os.path.join(output_path, "least_squares")
    test_gradient_descent_opt(output_path = least_squares_path)
    tikhonov_regularization_path = os.path.join(output_path, "tikhonov_regularization")
    test_tikhonov_opt(output_path = tikhonov_regularization_path)
    huber_functional_path = os.path.join(output_path, "huber_functional")
    test_huber_functional_opt(output_path = huber_functional_path)
    fair_potential_path = os.path.join(output_path, "fair_potential")
    test_fair_potential_opt(output_path = fair_potential_path)

if __name__ == "__main__":
    main()
