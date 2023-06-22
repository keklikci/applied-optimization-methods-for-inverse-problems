# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 22.06.2023                   #
# ********************************** #

import aomip
from pgm import PGM
from fpgm import FPGM
from elastic import Elastic
from ista import ISTA

def main():
    alphas = 1e-3
    betas = 1e-6
    variants = ["phantom", "challenge"]
    algos = [PGM(), FPGM(), Elastic(), ISTA()]
    for variant in variants:
        if variant == "phantom":
            algo = algos[0]
            algo.target = aomip.shepp_logan([128, 128])
            algo.sino = aomip.radon(algo.target, [128], np.arange(360), 128 * 100, 128 * 5)
        callback = []
        algo.step = alpha
        x, callback = algo.optimize(callback=callback)
        os.makedirs("images", exist_ok=True)
        # TODO: modify and package the algorithms
        # TODO: add control over number of iterations for restart exercise


if __name__ == "__main__":
    main()
