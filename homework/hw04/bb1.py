# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 14.06.2023                   #
# ********************************** #

import numpy as np
import tifffile
import os
import aomip

class BB1(Optimization):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def optimize(self, n=100, lmbd=1e-3) -> np.ndarray:
        x = self.x0
        gradient = self.calculate_gradient(x)
        prev_gradient = gradient
        prev_x = x
        for i in range(n):
            # avoid division by zero for the first iteration
            if not i:
                step = lmbd
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
        # descent with optimal step
        for _ in range(n):
            x -= step * gradient
        return x


def main():
    bb1 = BB1()
    os.makedirs("images/notebook/bb1", exist_ok=True)
    os.makedirs("images/bb1", exist_ok=True)
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1]
    for lmbd in lambdas:
        x = bb1.optimize(lmbd=lmbd)
        # save notebook output
        plt.axis("off")
        export = plt.imshow(x, cmap="gray")
        plt.colorbar(export)
        plt.tight_layout()
        plt.title(f"α={alpha}")
        plt.savefig(f"images/notebook/bb1/lambda_{lmbd}.png")
        plt.clf()
        # save tif output
        tifffile.imwrite(f"images/bb1/bb1/lambda_{lmbd}.tif", x)


if __name__ == "__main__":
    main()
