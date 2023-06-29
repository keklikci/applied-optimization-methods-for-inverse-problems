# -*- coding: utf-8 -*-

# ************************************** #
# Author: frank@in.tum.de                #
# Modified by: kaanguney.keklikci@tum.de #
# Date: 29.06.2023                       #
# ************************************** #

import tifffile
import json
import numpy as np


def load(file):
    with tifffile.TiffFile(file) as tif:
        data = tif.asarray()
        metadata = tif.pages[0].tags["ImageDescription"].value
    metadata = metadata.replace("'", '"')
    try:
        metadata = json.loads(metadata)
    except Exception as e:
        print(e)
        metadata = None
    return data, metadata


if __name__ == "__main__":
    main()
