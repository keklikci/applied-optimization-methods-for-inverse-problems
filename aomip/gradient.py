# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 12.07.2023                   #
# ********************************** #

import numpy as np
import aomip
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
    
    def optimize(self, n=100, **kwargs) -> np.ndarray:
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

    def optimize(self, n=100, beta=1.0, **kwargs) -> np.ndarray:
        x = self.x0
        self.scheduler.lr = kwargs.get("lr", self.scheduler.lr)
        subgradient = aomip.Subgradient()
        derivative = aomip.FirstDerivative()
        prev_loss, curr_loss = 0.0, 0.0
        # assert self.precondition(self.scheduler.lr), "α-value did not meet all preconditions!"
        for _ in range(n):
            prev_loss = curr_loss
            subgradient = np.sign(x)
            x = self.scheduler.update(x, subgradient, lr=self.scheduler.lr)
            # compute derivative in all directions
            grad = derivative.apply(x)
            # compute running loss
            least_squares_loss = aomip.leastSquares(x, aomip.Optimization.target)
            regularization = beta * np.linalg.norm(grad, ord=1)
            loss = least_squares_loss + regularization
            curr_loss += loss
            # TODO: convergence analysis
        return x

    def precondition(self, loss_fn) -> bool:
        pass
    
