# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 05.07.2023                   #
# ********************************** #

import numpy as np
from l1 import *


class L11(ProximalOperator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1 = L1()

    def proximal(self, x, lmbd):
        v = np.zeros_like(x)
        nrows, _ = v.shape
        for k in range(nrows):
                v[k] = self.l1.proximal(v[k], lmbd)
        return v

def main():
    prox = L11()
    x = np.random.sample((5, 10))
    # apply proximal
    x = prox.proximal(x=x, lmbd=1.0)
    assert x.sum() == 0

if __name__ == "__main__":
    main()
