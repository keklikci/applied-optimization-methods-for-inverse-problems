# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 12.07.2023                   #
# ********************************** #

import numpy as np
import aomip
import tifffile
from abc import ABC

class Gradient(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = 1e-2

    def setlr(self, lr) -> None:
        self.lr = lr

    def update(self, x, gradient, **kwargs) -> np.ndarray:
        self.lr =  kwargs.get("lr", self.lr)
        return x - self.lr * gradient

class GradientDescent(aomip.Optimization):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scheduler = Gradient()

    def optimize(self, n=10, **kwargs) -> np.ndarray:
        x = self.x0
        self.scheduler.lr = kwargs.get("lr", self.scheduler.lr)
        for _ in range(n):
            gradient = self.calculate_gradient(x)
            x = self.scheduler.update(x, gradient, lr=self.scheduler.lr)
        return x

class Subgradient(aomip.Optimization):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scheduler = Gradient()

    def optimize(self, n=100, beta=0.1, **kwargs) -> tuple:
        # backprojection as the initial guess
        A = self.operator
        backprojection = A.applyAdjoint(A.apply(self.target))
        x = backprojection.reshape(self.vol_shape, order="F")
        self.scheduler.lr = kwargs.get("lr", self.scheduler.lr)
        grad = aomip.FirstDerivative()
        # assert self.precondition(self.scheduler.lr), "α-value did not meet all preconditions!"
        history, running_loss = [], 0.0
        for _ in range(n):
            prevx = x
            dx = grad.applyAdjoint(grad.apply(prevx))
            norm = np.linalg.norm(dx, ord=1)
            subgradient = np.sign(norm)
            x = self.scheduler.update(prevx, (dx + self.calculate_gradient(prevx)), lr=self.scheduler.lr)
            loss = aomip.leastSquares(A.apply(x), self.sino) + norm
            history.append(loss)
        return x, history

    def precondition(self, loss_fn) -> bool:
        pass

