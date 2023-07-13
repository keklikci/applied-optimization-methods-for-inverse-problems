# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 12.07.2023                   #
# ********************************** #

import aomip
import numpy as np


class FPGM(aomip.Optimization):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lmbd = 1.0
        self.f = aomip.L1()
        self.objective = aomip.leastSquares

    def optimize(self, n=10, beta=0.1) -> tuple:
        A = self.operator
        x, z = self.x0, self.x0
        t = 1.0
        history, loss = [], 0.0
        for i in range(n):
            if i % 10 == 0:
                print(f"Loss @ {i}-th iteration = {loss:.2f}")
            xprev, zprev = x, z
            gradient = self.calculate_gradient(z)
            L = np.linalg.norm(gradient, ord=2) ** 2
            tprev = t
            t = (1 + np.sqrt(1 + 4 * t**2)) / 2
            self.lmbd = (tprev - 1) / t
            z = self.f.proximal(xprev - 1 / L * gradient, lmbd=(1 / L))
            x = z + self.lmbd * (z - zprev)
            loss = self.objective(A.apply(x), self.sino) + beta * np.linalg.norm(
                x, ord=1
            )
            history.append(loss)
        print(f"Completed, loss = {loss:.2f}")
        return x, history
