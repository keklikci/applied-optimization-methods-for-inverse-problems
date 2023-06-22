# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 22.06.2023                   #
# ********************************** #

import numpy as np
from .proximal_operator import ProximalOperator


class Huber(ProximalOperator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def proximal(self, x, alpha, sigma):
        x = (1.0 - sigma / max(np.linalg.norm(x), sigma + alpha)) * x
        return x
