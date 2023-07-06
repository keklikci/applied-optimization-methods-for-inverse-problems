# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 22.06.2023                   #
# ********************************** #

import numpy as np
from .proximal import ProximalOperator


class L1(ProximalOperator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def proximal(self, x, lmbd):
        return np.sign(x) * np.maximum(np.abs(x) - lmbd, 0)
