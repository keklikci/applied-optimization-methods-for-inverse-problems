# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 22.05.2023                   #
# ********************************** #

import numpy as np
from scipy.fft import fft2, ifft2

ALLOWED_MODES = {
    "blurred",
    "deblurred",
}


class Blur:
    """
    Convolution / Deconvolution with Gaussian filter

    Parameters
    ----------
    method : :obj:`np.ndarray`
        Input image shape
    kernel_stride : :obj: `int`
        Stride, i.e., size of the gaussian kernel
    scale: : :obj: `str`
        Standard deviation of the gaussian kernel
    _mode: :obj: `str`, optional
    """

    def __init__(
        self,
        shape: np.ndarray,
        kernel_stride: int,
        scale: float,
        _mode: str = "blurred",
    ) -> None:
        self.image_shape = shape
        self.blur = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * scale**2))
            * np.exp(-(x**2 + y**2) / (2 * scale**2)),
            (kernel_stride, kernel_stride),
        )
        if self.__check_args(_mode.lower()):
            self.mode = _mode

    def __str__(self) -> str:
        return self.mode

    def __check_args(self, arg: str) -> bool:
        if not isinstance(arg, str):
            raise ValueError("Mode must be of type str!")
        if not arg in ALLOWED_MODES:
            raise ValueError(f"Mode must be an element of {ALLOWED_MODES}")
        return True

    def __convolution(self, image: np.ndarray) -> np.ndarray:
        fft_image = fft2(image)
        blur_fft = fft2(self.blur, s=image.shape)
        fft_blurred = fft_image * blur_fft
        fft_blurred = np.real(ifft2(fft_blurred))
        return fft_blurred

    def __deconvolution(self, image: np.ndarray) -> np.ndarray:
        fft_image = fft2(image)
        deblur_fft = fft2(self.blur, s=fft_image.shape)
        fft_deblurred = fft_image / deblur_fft
        fft_deblurred = np.real(ifft2(fft_deblurred))
        return fft_deblurred

    def update(self, _mode: str) -> None:
        self.mode = _mode

    def transform(self, image: np.ndarray) -> np.ndarray:
        if self.mode == "blurred":
            return self.__convolution(image)
        return self.__deconvolution(image)
