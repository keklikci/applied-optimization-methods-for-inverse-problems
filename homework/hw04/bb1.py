# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 14.06.2023                   #
# ********************************** #

import numpy as np
import tifffile
import os
import sys
import aomip
from optimize import Optimization


class BB1(Optimization):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def optimize(self, num_iterations=100) -> None:
        x = self.x0
        gradient = self.calculate_gradient(x)
        prev_gradient = gradient
        prev_x = x
        for i in range(num_iterations):
            # avoid division by zero for the first iteration
            if not i:
                step = 1e-3
            else:
                gradient_diff = grad - prev_gradient
                x_diff = x - prev_x
                step = np.dot(x_diff.T, gradient_diff) / gradient_diff.T * gradient_diff
            next_x = x - step * gradient
            error = self.calculate_norm(next_x)
            objective = self.calculate_norm(x)
            terminal_bound = error < objective
            if terminal_bound:
                break
            prev_x = x
            x = next_x
            prev_gradient = gradient
            gradient = self.calculate_gradient(x)
        # descent with the optimal step
        for i in range(num_iterations):
            x -= step * gradient
        return x


def main():
    bb1 = BB1()
    x = bb1.optimize()
    os.makedirs("images", exist_ok=True)
    tifffile.imsave("images/bb1.tif", x.astype(np.uint8))


if __name__ == "__main__":
    main()
