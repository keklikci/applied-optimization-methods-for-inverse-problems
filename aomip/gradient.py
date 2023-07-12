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

    def optimize(self, n=10, beta=0.001, **kwargs) -> np.ndarray:
        # backprojection as the initial guess
        backprojection = self.operator.applyAdjoint(self.operator.apply(self.target))
        x = backprojection.reshape(self.vol_shape, order="F")
        self.scheduler.lr = kwargs.get("lr", self.scheduler.lr)
        # assert self.precondition(self.scheduler.lr), "α-value did not meet all preconditions!"
        for _ in range(n):
            prevx = x
            subgradient = np.sign(prevx)
            x = self.scheduler.update(prevx, subgradient, lr=self.scheduler.lr)
            # print(x.sum(), prevx.sum())
        return x

    def precondition(self, loss_fn) -> bool:
        pass
    
