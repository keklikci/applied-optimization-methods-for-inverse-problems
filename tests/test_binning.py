# !/usr/bin/env python
# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 01.05.2023                   #
# ********************************** #

import numpy as np
import pytest

from aomip.binning import bin


def test_signal_01():
    signal = np.array([[1, 2, 3, 4, 5, 6], [2, 4, 6, 8, 10, 12]])
    assert bin(image=signal, factor=2).shape == (2, 3)


def test_odd_factor_rejected():
    signal = np.array([[1, 2, 3, 4, 5, 6], [2, 4, 6, 8, 10, 12]])
    with pytest.raises(ValueError, match="power of 2"):
        bin(image=signal, factor=3)
