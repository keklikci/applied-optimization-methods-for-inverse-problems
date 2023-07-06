# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 05.07.2023                   #
# ********************************** #

import aomip
import numpy as np
from abc import ABC


class PowerIteration(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vol_shape = [512, 512]
        self.x0 = np.zeros(self.vol_shape)
        self.sino_shape = [512]
        self.d2c = self.vol_shape[0] * 100.0
        self.c2d = self.vol_shape[0] * 5.0
        self.thetas = np.arange(360)
        self.operator = aomip.XrayOperator(
            self.vol_shape, self.sino_shape, self.thetas, self.d2c, self.c2d
        )

    def power(self, n=100) -> float:
        # flatten
        prevb = np.random.rand(self.x0.flatten().size)
        for i in range(n):
            b_k = self.operator.applyAdjoint(self.operator.apply(prevb)).flatten()
            # normalize
            b = b_k / np.linalg.norm(b_k)
            lmbd = b.T.dot(prevb) / prevb.T.dot(prevb)
            prevb = b
        return lmbd
