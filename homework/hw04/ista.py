# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 14.06.2023                   #
# ********************************** #

import aomip
import numpy as np
import tifffile
from optimize import Optimization

class ISTA(Optimization):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.step = 0.1
        self.beta = 1
    
    @property
    def step(self) -> float:
        return self._step

    @step.setter
    def step(self, step) -> float:
        self._step = step

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, beta) -> float:
        self._beta = beta

    def project(self, x, gradient) -> np.ndarray:
        xproj = np.maximum(x - self.step * gradient, 0)
        return xproj

    def soft_threshold(self, x) -> np.ndarray:
        return np.sign(x) * np.maximum(np.abs(x) - self.beta, 0)

    def optimize(self, num_iterations=100) -> None:
        x = self.x0
        for i in range(num_iterations):
            gradient = self.calculate_gradient(x)
            x -= self.step * gradient
            x = self.soft_threshold(x)
        return x


def main():
    ista = ISTA()
    output = ista.optimize()
    os.makedirs("images", exist_ok=True)
    tifffile.imsave("images/ista.tif", output.astype(np.uint8))


if __name__ == "__main__":
    main()