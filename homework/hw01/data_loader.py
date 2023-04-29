# !/usr/bin/env python
# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 29.04.2023                   #
# ********************************** #

import time
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

walnut_id = 42
orbit_id = 3
data_dir = f"./datasets/Walnut{walnut_id}"
orbit_ids = np.linspace(1, 3, 3, dtype=np.int64)
projections_dir = os.path.join(data_dir, "Projections")

def read_filenames() -> tuple:
    """
    Read filenames on path for CT scans, dark frame, and flat-field images on a given orbit.
    :param:
        None
    :return:
        scan_filenames: list of strings for scan filenames
        dark_frame_filenames: list of strings for dark frame filenames
        flat_field_filenames: list of strings for flat field image filenames
    """
    scan_filenames = []
    dark_frame_filenames = []
    flat_field_filenames = []
    # filter projection filenames
    orbits_dir = os.path.join(projections_dir, f"tubeV{orbit_id}")
    projection_filenames = [projection for projection in os.listdir(orbits_dir) if not ("geom" in projection or "txt" in projection)]
    dark_frame_filenames = [projection for projection in projection_filenames if projection.startswith("d")]
    flat_field_filenames = [projection for projection in projection_filenames if projection.startswith("io")]
    scan_filenames = list(set(projection_filenames) - set(dark_frame_filenames) - set(flat_field_filenames))
    # join filtered filenames
    scan_filenames = [os.path.join(orbits_dir, scan) for scan in scan_filenames]
    dark_frame_filenames = [os.path.join(orbits_dir, dark_frame) for dark_frame in dark_frame_filenames]
    flat_field_filenames = [os.path.join(orbits_dir, flat_field) for flat_field in flat_field_filenames]
    print("Reading filenames, done.")
    return scan_filenames, dark_frame_filenames, flat_field_filenames

def load_files(debug: bool = False) -> tuple:
    """
    Load filenames on path for CT scans, dark frame, and flat-field images on a given orbit.
    :param:
        debug: boolean to display projection or not
    :return:
        scans: np.ndarray of np.ndarrays for CT scans
        dark_frames: np.ndarray of np.ndarrays dark frames
        flat_fields: np.ndarray of np.ndarrays for flat field images
    """
    scan_filenames, dark_frame_filenames, flat_field_filenames = read_filenames()
    for filename in scan_filenames:
        scan = Image.open(filename)
        # TODO: store numpy matrix
        # TODO: add debug functionality
        # TODO: close image
        # TODO: flat-field correction
        # TODO: save progress
    print("Loading filenames, done.")

def main():
    start_time = time.time()
    print("Data loader executing...")
    load_files()
    print("Data loader execution, done.")

if __name__ == "__main__":
    main()




