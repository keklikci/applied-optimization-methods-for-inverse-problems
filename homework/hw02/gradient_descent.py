# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 08.05.2023                   #
# ********************************** #

import numpy as np


def gradient_descent(
    x: np.ndarray,
    gradient: float,
    learning_rate: float = 0.001,
    n_iterations: int = 1000,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """
    x: np.ndarray of 1D/2D signal
    gradient: float, derivative of the cost function
    learning_rate: float, learning parameter
    n_iterations: int, number of iterations
    epsilon: float, fixed margin to 0.00001
    """
    for i in range(n_iterations):
        margin = -learning_rate * gradient
        if abs(margin) < epsilon:
            break
        # update
        x -= learning_rate * gradient
    return x
