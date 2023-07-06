# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 06.07.2023                   #
# ********************************** #

import numpy as np


class FirstDerivative:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply(self, x):
        dim = x.ndim
        grad = np.zeros((dim,) + x.shape)
        for i in range(dim):
            grad[i, ...] = forward_diff(x, axis=i).copy()
        return grad

    def applyAdjoint(self, x):
        grad = np.zeros(x.shape[1:])
        for i in range(x.shape[0]):
            grad[...] += adjoint_forward_diff(x[i], axis=i).copy()
        return grad


def forward_diff(x, axis=-1, sampling=1):
    x = x.swapaxes(axis, -1)
    y = np.zeros_like(x)
    y[..., :-1] = (x[..., 1:] - x[..., :-1]) / sampling
    return y.swapaxes(axis, -1)


def adjoint_forward_diff(x, axis=-1):
    x = x.swapaxes(axis, -1)
    y = np.zeros_like(x)
    y[..., :-1] -= x[..., :-1]
    y[..., 1:] += x[..., :-1]
    return y.swapaxes(axis, -1)
