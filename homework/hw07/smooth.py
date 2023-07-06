# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 06.07.2023                   #
# ********************************** #

import numpy as np
import tifffile
import matplotlib.pyplot as plt
import os
import aomip

def main():
    N = 512
    os.makedirs("images", exist_ok=True)
    plt.axis("off")
    s = plt.imshow(aomip.smooth(N))
    plt.colorbar(s)
    plt.tight_layout()
    plt.savefig("smooth.png")


if __name__ == "__main__":
    main()
