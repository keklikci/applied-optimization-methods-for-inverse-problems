# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 16.06.2023                   #
# ********************************** #

from abc import ABC, abstractmethod

class ProximalOperator(ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def proximal(self) -> NotImplementedError:
        raise NotImplementedError