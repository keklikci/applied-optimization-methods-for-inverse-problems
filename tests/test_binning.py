# !/usr/bin/env python
# -*- coding: utf-8 -*-
# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 01.05.2023                   #
# ********************************** #

import numpy as np
from aomip.binning import bin

def test_signal_01():
    signal = np.array([[1, 2, 3, 4, 5, 6], [2, 4, 6, 8, 10, 12]])
    return bin(image=signal, factor=2)

def test_signal_02():
    signal = np.array([[1, 2, 3, 4, 5, 6], [2, 4, 6, 8, 10, 12]])
    return bin(image=signal, factor=3)

# sample execution
print(test_signal_01())

# an odd factor should throw an error
try:
    test_signal_02()
except:
    print("Correct error handling.")