# -*- coding: utf-8 -*-

# ************************************** #
# Author: frank@in.tum.de                #
# Modified by: kaanguney.keklikci@tum.de #
# Date: 29.06.2023                       #
# ************************************** #

import numpy as np

config = {}


def configure(data, metadata) -> dict:
    # config["angles"] = np.degrees(np.array(metadata["angles"])[: metadata["rotview"]])
    config["angles"] = np.arange(0, 360, 5)
    config["voxels"] = 0.7
    config["voxscale"] = 1 / config["voxels"]
    config["sinoshape"] = [128, 128]
    config["ndetector"] = data.shape[:-1]
    config["volspace"] = config["voxscale"]
    config["volshape"] = data.shape
    config["detectorspace"] = config["voxscale"] * metadata["du"]
    config["ds2c"] = config["voxscale"] * metadata["dso"]
    config["dc2d"] = config["voxscale"] * metadata["ddo"]
    return config
