# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 14.06.2023                   #
# ********************************** #

import numpy as np
import os
import matplotlib.pyplot as plt
import tifffile
from line_search import LineSearch


class Backtracking(LineSearch):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def optimize(self, alpha=1e-5, beta=0.75, num_iterations=100) -> None:
        x = self.x0
        step = 1.0
        gradient = self.calculate_gradient(x)
        norm = np.linalg.norm(gradient)
        for i in range(num_iterations):
            next_x = x - step * gradient
            error = self.calculate_norm(next_x)
            objective = self.calculate_norm(x)
            terminal_bound = error <= objective - alpha * step * norm
            if terminal_bound:
                break
            else:
                step = self.decay(step, beta)
        # descent with the optimal step
        for i in range(num_iterations):
            x -= step * gradient
        return x


def main():
    backtrack = Backtracking()
    output = backtrack.optimize()
    os.makedirs("images", exist_ok=True)
    tifffile.imsave("images/backtrack.tif", output.astype(np.uint8))


if __name__ == "__main__":
    main()
