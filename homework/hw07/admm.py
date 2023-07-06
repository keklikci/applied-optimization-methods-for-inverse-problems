# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 05.07.2023                   #
# ********************************** #

import aomip
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import os
from optimize import Optimization

class ADMM(Optimization):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mu = 1.0
        self.tau = 1.0
        self.f = None
        self.g = None

    def optimize(self, n=100, callback=None) -> None:
        x, z = self.x0, self.operator.apply(self.x0)
        u = np.zeros(self.sino_shape)[:, np.newaxis]
        for k in range(n):
            prevx, prevz, prevu = x, z, u
            x = self.fproximal(prevx - self.mu / self.tau * self.operator.applyAdjoint(self.operator.apply(prevx) - prevz + prevu))
            z = self.gproximal(self.operator.apply(x) + prevu)
            u = prevu + self.operator.apply(x) - z
        return x

    def fproximal(self, x) -> np.ndarray:
        return self.f.proximal(x, lmbd=1.0)

    def gproximal(self, x) -> np.ndarray:
        return self.g.proximal(x, lmbd=1.0)
