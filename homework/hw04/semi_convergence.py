# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 14.06.2023                   #
# ********************************** #

import aomip
import numpy as np
import tifffile
import os

# usual parameter setup
vol_shape = [512, 512]
sino_shape = [512]
d2c = vol_shape[0] * 100.0
c2d = vol_shape[0] * 5.0
thetas = np.arange(360)
phantom = aomip.shepp_logan(vol_shape)
sino = aomip.radon(phantom, sino_shape, thetas, d2c, c2d)

# define the operator
operator = aomip.XrayOperator(vol_shape, sino_shape, thetas, d2c, c2d)

# apply filter to the sinogram
def apply_filter(sino, sino_shape):
    H = np.linspace(-1, 1, sino_shape[0])
    ram_lak = np.abs(H)
    h = np.tile(ram_lak, (360, 1)).T
    fftsino = np.fft.fft(sino, axis=0)
    projection = np.fft.fftshift(fftsino, axes=1) * np.fft.fftshift(h, axes=0)
    sino = np.real(np.fft.ifft(np.fft.ifftshift(projection, axes=1), axis=0))
    return sino

def experiment(x, operator, sino, sino_shape, alpha=1e-4, num_iterations=100, log_error=True) -> None:
    # define a callback array
    callback = []
    # --- noise snippet ---
    noise = aomip.noise.Noise()
    noise.update(method="Poisson")
    sino = noise.transform(sino)
    # --- noise snippet ---
    sino = apply_filter(sino, sino_shape)
    for i in range(num_iterations):
        error = np.linalg.norm(operator.apply(x) - sino)
        if log_error:
            callback.append(error)
        gradient = operator.applyAdjoint(error)
        x -= alpha * gradient
    return x, callback

def main():
    x = np.zeros(vol_shape) 
    output, callback = experiment(x, operator, sino, sino_shape)
    os.makedirs("images", exist_ok=True)
    tifffile.imsave("images/experiment_recon.tif", output.astype(np.uint8))
    plt.plot(np.arange(len(callback)), callback)
    plt.figsave("images/experiment_error.png")

if __name__ == "__main__":
    main()


