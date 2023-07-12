# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 12.07.2023                   #
# ********************************** #

import numpy as np

def leastSquares(forward, b) -> float:
    return 0.5 * np.linalg.norm(forward - b, ord=2) ** 2
