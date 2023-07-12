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
        self.objective = aomip.leastSquares

    def optimize(self, n=100, mu=1.0, tau=1.0, beta=0.1) -> None:
        A = self.operator
        x, z = self.x0, A.apply(self.x0)
        u = np.zeros(self.sino_shape)[:, np.newaxis]
        history, loss = [], 0.0
        for k in range(n):
            if k % 10 == 0:
                print(f"Loss @ {k}-th iteration = {loss:.2f}")
            prevx, prevz, prevu = x, z, u
            x = self.fproximal(
                prevx
                - mu
                / tau
                * A.applyAdjoint(A.apply(prevx) - prevz + prevu)
            )
            loss = self.objective(A.apply(prevx), self.sino) + beta * np.linalg.norm(prevx, ord=1)
            z = self.gproximal(A.apply(x) + prevu)
            u = prevu + A.apply(x) - z
            history.append(loss)
        print(f"Completed, loss = {loss:.2f}")
        return x

    def fproximal(self, x, lmbd=1.0) -> np.ndarray:
        return self.f.proximal(x, lmbd=lmbd)

    def gproximal(self, x, lmbd=1.0) -> np.ndarray:
        return self.g.proximal(x, lmbd=lmbd)
