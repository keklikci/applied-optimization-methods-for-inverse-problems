# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 05.07.2023                   #
# ********************************** #

import numpy as np
import aomip


class LADMM(aomip.ADMM):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.f = aomip.L1()
        self.g = aomip.L2()

    def optimize(self, n=100, mu=1.0, tau=1.0, callback=None) -> np.ndarray:
        x, z = self.x0, self.sino
        u = np.zeros(self.sino_shape)[:, np.newaxis]
        norm = aomip.PowerIteration().power()
        mu = self.compute(tau, norm)
        for k in range(n):
            prevx, prevz, prevu = x, z, u
            x = self.fproximal(
                prevx
                - mu
                / tau
                * self.operator.applyAdjoint(
                    self.operator.apply(prevx) - prevz + prevu
                ),
                lmbd=mu,
            )
            z = self.gproximal(self.operator.apply(x) + prevu, lmbd=tau)
            x = prevx - mu / tau * self.operator.applyAdjoint(
                self.operator.apply(prevx) - prevz + prevu
            )
            z = self.operator.apply(x) + prevu
            u = prevu + self.operator.apply(x) - z
        return x

    def fproximal(self, x, lmbd) -> np.ndarray:
        return self.f.proximal(x, lmbd=lmbd)

    def gproximal(self, x, lmbd) -> np.ndarray:
        return self.g.proximal(x, lmbd=lmbd)

    def compute(self, tau, norm) -> float:
        lmbd = 0.95 * tau / norm
        return lmbd