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
from proximal import *

class ADMM(Optimization):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mu = 1.0
        self.tau = 0.25
        self.sigma = 0.75
        self.f = L11()
        self.g = L2()

    def optimize(self, n=100, callback=None) -> None:
        x, z, u = self.x0, self.operator.apply(self.x0), self.x0
        for k in range(n):
            prevx, prevz, prevu = x, z, u
            x = self.fproximal(prevx - self.mu / self.tau * self.operator.applyAdjoint(self.operator.apply(prevx) - prevz + u))
            z = self.gproximal(self.operator.apply(x) + prevu)
            u = prevu + self.operator.apply(x) - z
        return x

    def fproximal(self, x) -> np.ndarray:
        return self.f.proximal(x, lmbd=1.0)

    def gproximal(self, x) -> np.ndarray:
        x -= self.sigma * self.calculate_gradient(x)
        return self.g.proximal(x, sigma=self.sigma, lmbd=1.0)


def main():
    admmm = ADMM()
    os.makedirs("images", exist_ok=True)
    x = admmm.optimize()
    tifffile.imwrite(f"images/admmm.tif", x)