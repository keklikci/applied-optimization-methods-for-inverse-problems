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
from admm import ADMM
from proximal.l11 import L11
from proximal.l21 import L21

class LADMM(ADMM):
    def __init__(self, *args, **kwargs) -> None:
        super.__init__(*args, **kwargs)
        self.f = L11()
        self.g = L21()

    def optimize(self, n=100, mu=1.0, tau=1.0, callback=None) -> None:
        x, z = self.x0, self.sino
        u = np.zeros(self.sino_shape)[:, np.newaxis]
        norm = aomip.power()
        mu = self.compute(tau, norm)
        for k in range(n):
            prevx, prevz, prevu = x, z, u
            x = self.fproximal(prevx - mu / tau * self.operator.applyAdjoint(self.operator.apply(prevx) - prevz + prevu))
            z = self.gproximal(self.operator.apply(x) + prevu)
            u = prevu + self.operator.apply(x) - z
        return x

    def fproximal(self, x) -> np.ndarray:
        return self.f.proximal(x, lmbd=1.0)

    def gproximal(self, x) -> np.ndarray:
        return self.g.proximal(x, lmbd=1.0)

    def compute(self, tau, norm) -> float:
        lmbd = 0.95 * tau / norm
        return lmbd

def main():
    admm = LADMM()
    x = admm.optimize()
    os.makedirs("images", exist_ok=True)
    tifffile.imwrite(f"images/admm_lasso.tif", x)

if __name__ == "__main__":
    main()