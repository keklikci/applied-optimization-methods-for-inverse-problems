# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 05.07.2023                   #
# ********************************** #

import numpy as np
from proximal import ProximalOperator

class L21(ProximalOperator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def proximal(self, x, lmbd):
        prox = np.zeros_like(x)
        norms = np.linalg.norm(x, axis=0, ord=2)
        scale = 1 - lmbd / np.maximum(norms, lmbd)
        prox[:, :] = scale * x[:, :]
        return prox

def main():
    prox = L21()
    x = np.random.sample((5, 10))
    # apply proximal
    x = prox.proximal(x=x, lmbd=1.0)
    # print(x)

if __name__ == "__main__":
    main()
