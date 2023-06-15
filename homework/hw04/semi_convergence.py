# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 14.06.2023                   #
# ********************************** #

import aomip
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os

# usual setup
vol_shape = [512, 512]
sino_shape = [512]
d2c = vol_shape[0] * 100.0
c2d = vol_shape[0] * 5.0
thetas = np.arange(360)
phantom = aomip.shepp_logan(vol_shape)
sino = aomip.radon(phantom, sino_shape, thetas, d2c, c2d)
operator = aomip.XrayOperator(vol_shape, sino_shape, thetas, d2c, c2d)


def apply_filter(sino, sino_shape):
    H = np.linspace(-1, 1, sino_shape[0])
    ram_lak = np.abs(H)
    h = np.tile(ram_lak, (360, 1)).T
    fftsino = np.fft.fft(sino, axis=0)
    projection = np.fft.fftshift(fftsino, axes=1) * np.fft.fftshift(h, axes=0)
    sino = np.real(np.fft.ifft(np.fft.ifftshift(projection, axes=1), axis=0))
    return sino


def experiment(
    x,
    operator,
    sino,
    sino_shape,
    alpha=1e-4,
    num_iterations=100,
    callback=None,
    loc=0.0,
    scale=1.0,
) -> None:
    noise = np.random.normal(loc=loc, scale=scale, size=[512, 360])
    sino += noise
    sino = apply_filter(sino, sino_shape)
    for i in range(num_iterations):
        error = operator.apply(x) - sino
        norm = np.linalg.norm(error)
        gradient = operator.applyAdjoint(error)
        x -= alpha * gradient
        if callback is not None and i % 2 == 0:
            callback.append(norm)
    return x, callback


def main():
    x = np.zeros(vol_shape)
    callback = []
    locs = np.linspace(0, 2, num=5)
    scales = np.linspace(0, 6, num=5)
    for i, data in enumerate(zip(locs, scales)):
        x, callback = experiment(
            x, operator, sino, sino_shape, callback=callback, loc=data[0], scale=data[1]
        )
        os.makedirs("images", exist_ok=True)
        tifffile.imsave(f"images/experiment_recon_{i + 1}.tif", x.astype(np.uint8))
        plt.plot(np.arange(len(callback)), callback)
        plt.ylabel(f"Reconstruction error, alpha = {1e-4}")
        plt.xlabel(f"# of iterations")
        plt.savefig(f"images/experiment_error_{i + 1}.png")


if __name__ == "__main__":
    main()
