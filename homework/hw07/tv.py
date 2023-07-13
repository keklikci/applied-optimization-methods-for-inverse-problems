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


class TV(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ops = [A, aomip.FirstDerivative()]
        self.operator = aomip.StackedOperator(self.ops)
        self.vols = [512, 512]
        self.bs = [512]
        self.d2c = self.vols[0] * 100.0
        self.c2d = self.vols[0] * 5.0
        self.thetas = np.arange(360)
        self.x0 = np.zeros(self.vols)
        self.target = tifffile.imread(
            "/srv/ceph/share-all/aomip/htc2022_ground_truth/htc2022_05c_recon.tif"
        )
        self.sino = aomip.radon(self.target, self.bs, self.thetas, self.d2c, self.c2d)
        self.f = aomip.L11()
        self.g = aomip.L21()
        self.mode = None

    def optimize(self, n=100, mu=1.0, tau=1.0, callback=None) -> None:
        x, z = self.x0, self.sino
        u = np.zeros(self.bs)[:, np.newaxis]
        norm = aomip.PowerIteration().power()
        mu = self.compute(tau, norm)
        for k in range(n):
            prevx, prevz, prevu = x, z, u
            forward = self.operator.apply(prevx)
            f1, f2 = forward[0], forward[1]
            x = prevx - mu / tau * self.operator.applyAdjoint(
                self.operator.apply(prevx) - prevz + prevu
            )
            z = self.operator.apply(x) + prevu
            u = prevu + self.operator.applyAdjoint(x) - z
        return x

    def fproximal(self, x, lmbd) -> np.ndarray:
        return self.f.proximal(x, lmbd=lmbd)

    def gproximal(self, x, lmbd) -> np.ndarray:
        return self.g.proximal(x, lmbd=lmbd)

    def compute(self, tau, norm) -> float:
        lmbd = 0.95 * tau / norm
        return lmbd


def main():
    tv = TV()
    x = tv.optimize()
    os.makedirs("images", exist_ok=True)
    plt.imshow(x, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"images/tv.tif", transparent=True)


if __name__ == "__main__":
    # main()
    pass
