# !/usr/bin/env python
# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 30.04.2023                   #
# ********************************** #

import time
import os
from data_loader import *
from flat_field import apply_correction
from util.normalizer import normalize
from util.plotter import plot_figure
from util.scaler import uint_mapper


def main():
    start_time = time.time()
    print("Data loader executing...")
    scans, dark_frames, flat_fields = load_files()
    print("Data loader execution, done.")
    # fetch dark frame
    dark_frame = dark_frames[0]
    # create directory
    corrected_scan_dir = "./homework/hw01/output/scan/flat_field_corrected"
    os.makedirs(corrected_scan_dir, exist_ok=True)
    for tag, scan in enumerate(scans):
        out = apply_correction(scan, dark_frame, flat_fields)
        # ensure normalization
        out = normalize(out)
        # map to the correct range
        out = uint_mapper(out)
        plot_figure(out, save=True, save_dir=corrected_scan_dir, tag=tag)
        # sequential application of normalization and mapping yields higher definitions
    print("Flat-field correction, done.")
    print("Total execution time: {:.2f} seconds.".format(time.time() - start_time))


if __name__ == "__main__":
    main()
