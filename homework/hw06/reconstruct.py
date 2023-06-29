# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 29.06.2023                   #
# ********************************** #

import aomip
import tifffile
import matplotlib.pyplot as plt
from loader import *
from config import *
from pogm import *

fpath = "/srv/ceph/share-all/aomip/mayo_clinical/out/L310_flat_fan_projections_fd.tif"
qpath = "/srv/ceph/share-all/aomip/mayo_clinical/out/L310_flat_fan_projections_qd.tif"


def main():
    fdata, fmeta = load(fpath)
    qdata, qmeta = load(qpath)
    fconfig, qconfig = configure(fdata, fmeta), configure(qdata, qmeta)
    operator = aomip.XrayOperator(
        fconfig["volshape"],
        fconfig["sinoshape"],
        fconfig["angles"],
        fconfig["ds2c"],
        fconfig["dc2d"],
    )
    fsino = aomip.radon(
        fdata, fconfig["sinoshape"], fconfig["angles"], fconfig["ds2c"], fconfig["dc2d"]
    )
    qsino = aomip.radon(
        qdata, qconfig["sinoshape"], qconfig["angles"], qconfig["ds2c"], qconfig["dc2d"]
    )
    lr, n = 1e-3, 100
    for i in range(n):
        fgrad = operator.applyAdjoint(operator.apply(fdata) - fsino)
        qgrad = operator.applyAdjoint(operator.apply(qdata) - qsino)
        fdata -= lr * fgrad
        qdata -= lr * qgrad
    # fig, (ax1, ax2) = plt.subplots(1, 2, figshape = (12, 5))
    os.makedirs("images", exist_ok=True)
    tifffile.imwrite("images/qrecon.tif", qdata)
    tifffile.imwrite("images/frecon.tif", fdata)


if __name__ == "__main__":
    main()
