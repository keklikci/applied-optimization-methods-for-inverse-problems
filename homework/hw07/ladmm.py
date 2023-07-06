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
from proximal.l1 import L1
from proximal.l2 import L2

class LADMM(ADMM):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.f = L1()
        self.g = L2()

    def optimize(self, n=100, mu=1.0, tau=1.0, callback=None) -> None:
        x, z = self.x0, self.sino
        u = np.zeros(self.sino_shape)[:, np.newaxis]
        norm = aomip.PowerIteration().power()
        mu = self.compute(tau, norm)
        for k in range(n):
            prevx, prevz, prevu = x, z, u
            x = self.fproximal(prevx - mu / tau * self.operator.applyAdjoint(self.operator.apply(prevx) - prevz + prevu), lmbd=mu)
            z = self.gproximal(self.operator.apply(x) + prevu, lmbd=tau)
            x = prevx - mu / tau * self.operator.applyAdjoint(self.operator.apply(prevx) - prevz + prevu)
            z = self.operator.apply(x) + prevu
            u = prevu + self.operator.apply(x) - z
        return x

    def fproximal(self, x, lmbd) -> np.ndarray:
        return self.f.proximal(x, lmbd=lmbd)

    def gproximal(self, x, lmbd) -> np.ndarray:
        return self.g.proximal(x, lmbd=lmbd, sigma=1.0)

    def compute(self, tau, norm) -> float:
        lmbd = 0.95 * tau / norm
        return lmbd

def main():
    admm = LADMM()
    taus = np.logspace(-3, 6, 10)
    for tau in taus:
        x = admm.optimize(tau=tau)
        os.makedirs("images", exist_ok=True)
        # normalize
        x = (x - x.min()) / (x.max() - x.min())
        plt.imshow(x, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"images/tau_{tau}.tif", transparent=True)


if __name__ == "__main__":
    main()
