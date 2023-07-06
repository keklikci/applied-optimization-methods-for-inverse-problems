# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 06.07.2023                   #
# ********************************** #

import numpy as np
from abc import ABC, abstractmethod


class ProximalOperator(ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def proximal(self) -> NotImplementedError:
        raise NotImplementedError


class Nonnegativity(ProximalOperator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def proximal(self, x, step, gradient):
        return np.maximum(x - step * gradient, 0)


class L21(ProximalOperator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def proximal(self, x, lmbd):
        prox = np.zeros_like(x)
        norms = np.linalg.norm(x, axis=0, ord=2)
        scale = 1 - lmbd / np.maximum(norms, lmbd)
        prox[:, :] = scale * x[:, :]
        return prox


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


class L2(ProximalOperator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def proximal(self, x, lmbd, sigma=1):
        return x / (1 + sigma * lmbd)


class L1(ProximalOperator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def proximal(self, x, lmbd):
        return np.sign(x) * np.maximum(np.abs(x) - lmbd, 0)


class Huber(ProximalOperator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def proximal(self, x, lmbd, sigma):
        x = (1.0 - sigma / max(np.linalg.norm(x), sigma + lmbd)) * x
        return x


class Constant(ProximalOperator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def proximal(self, x):
        return x

class StackedOperator:
    def __init__(self, operators):
        self.ops = operators

    def apply(self, x):
        l = np.array([], dtype=object)
        l.resize(len(self.ops))
        for i, op in enumerate(self.ops):
            l[i] = op.apply(x)
        return l

    def applyAdjoint(self, y):
        x = self.ops[0].applyAdjoint(y[0])
        for yi, op in zip(y[1:], self.ops[1:]):
            x += op.applyAdjoint(yi)
        return x