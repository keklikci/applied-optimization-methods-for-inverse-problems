# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 06.07.2023                   #
# ********************************** #

import numpy as np
import tifffile
import matplotlib.pyplot as plt
import os
import aomip

# setup
vols, bs = [512, 512], [512]
d2c, c2d = vols[0] * 100.0, vols[0] * 5.0
thetas = np.arange(360)
x = np.zeros(vols)
A = aomip.XrayOperator(vols, bs, thetas, d2c, c2d)

class IsotropicTV(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ops = [A, aomip.FirstDerivative()]
        self.operator = aomip.StackedOperator(self.ops)

    def optimize(self, n=100, mu=1.0, tau=1.0, callback=None) -> None:
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


    