# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 16.06.2023                   #
# ********************************** #

import numpy as np
from .proximal_operator import ProximalOperator

class Nonnegativity(ProximalOperator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def proximal(self, x, step, gradient):
        return np.maximum(x - step * gradient, 0)
