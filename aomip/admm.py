# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 05.07.2023                   #
# ********************************** #

import numpy as np
import aomip


class ADMM(aomip.Optimization):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.f = None
        self.g = None

    def optimize(self, n=100, mu=1.0, tau=1.0, callback=None) -> None:
        x, z = self.x0, self.operator.apply(self.x0)
        u = np.zeros(self.sino_shape)[:, np.newaxis]
        for k in range(n):
            prevx, prevz, prevu = x, z, u
            x = self.fproximal(
                prevx
                - mu
                / tau
                * self.operator.applyAdjoint(self.operator.apply(prevx) - prevz + prevu)
            )
            z = self.gproximal(self.operator.apply(x) + prevu)
            u = prevu + self.operator.apply(x) - z
        return x

    def fproximal(self, x, lmbd) -> np.ndarray:
        return self.f.proximal(x, lmbd=lmbd)

    def gproximal(self, x, lmbd) -> np.ndarray:
        return self.g.proximal(x, lmbd=lmbd)
