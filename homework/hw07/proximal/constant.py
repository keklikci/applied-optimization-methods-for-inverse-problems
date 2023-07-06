# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 16.06.2023                   #
# ********************************** #

from .proximal import ProximalOperator


class Constant(ProximalOperator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def proximal(self, x):
        return x
