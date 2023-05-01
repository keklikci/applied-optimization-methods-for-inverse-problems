# !/usr/bin/env python
# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 30.04.2023                   #
# ********************************** #

import os
import numpy as np
from PIL import Image
from dotenv import load_dotenv

# source configuration file
load_dotenv()
walnut_id = os.environ["WALNUT_ID"]
orbit_id = os.environ["ORBIT_ID"]
save_dir = os.environ["SAVE_DIR"]
filecount = int(os.environ["FILE_COUNT"])

# dynamically assigned configuration
data_dir = f"./datasets/Walnut{walnut_id}"
orbit_ids = np.linspace(1, 3, 3, dtype=np.int64)
projections_dir = os.path.join(data_dir, "Projections")

def read_filenames() -> tuple:
    """
    Reads filenames on path for CT scans, dark frame, and flat-field images on a given orbit.
    :param:
        None
    :return:
        scan_filenames: list of strings for scan filenames, limit by filecount
        dark_frame_filenames: list of strings for dark frame filenames, limit by filecount
        flat_field_filenames: list of strings for flat field image filenames, limit by filecount
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
    # limit number of files
    scan_filenames = scan_filenames[:filecount]
    dark_frame_filenames = dark_frame_filenames[:filecount]
    flat_field_filenames = flat_field_filenames[:filecount]
    return scan_filenames, dark_frame_filenames, flat_field_filenames


def store_matrix(storage: list = [], filenames: list = [], save_dir: str = "") -> list:
    """
    Stores and updates data in a list for CT scans, dark frame, and flat-field images on a given orbit.
    :param:
        storage: list of np.ndarrays for CT data
        filenames: list of filenames, either walnut scans, dark frame or flat-field images
        save_dir: str for saving raw, i.e., unprocessed file
    :return:
        storage: list of aggregated np.ndarrays for CT data
    """
    for tag, filename in enumerate(filenames):
        data = Image.open(filename)
        data = np.array(data)
        storage.append(data)
        image = Image.fromarray(data)
        save_file(image, save_dir, tag + 1)
        image.close()
    return storage

def save_file(data, save_dir: str = "", tag: int = 0) -> None:
    """
    Saves Pillow image to the parametrized directory.
    :param:
        data: an open PIL image
        save_dir: directory to save PIL image
        tag: output png suffix
    :return:
        None
    """
    data.save(f"{save_dir}/000{tag}.png")

def create_directory():
    """
    Creates directory structure for saving raw, i.e., unprocessed files
    :param:
        None
    :return:
        None
    """
    scan_save_dir = os.path.join(save_dir, "scan", "raw")
    dark_frame_save_dir = os.path.join(save_dir, "dark_frame")
    flat_field_save_dir = os.path.join(save_dir, "flat_field")
    os.makedirs(scan_save_dir, exist_ok=True)
    os.makedirs(dark_frame_save_dir, exist_ok=True) 
    os.makedirs(flat_field_save_dir, exist_ok=True)
    return scan_save_dir, dark_frame_save_dir, flat_field_save_dir

def load_files() -> tuple:
    """
    Loads filenames on path for CT scans, dark frame, and flat-field images on a given orbit.
    :param:
        None
    :return:
        scans: list of np.ndarrays for CT scans
        dark_frames: list of np.ndarrays dark frames
        flat_fields: list of np.ndarrays for flat field images
    """
    scans, dark_frames, flat_fields = [], [], []
    scan_filenames, dark_frame_filenames, flat_field_filenames = read_filenames()
    scan_save_dir, dark_frame_save_dir, flat_field_save_dir = create_directory()
    scans = store_matrix(scans, scan_filenames, scan_save_dir)
    dark_frames = store_matrix(dark_frames, dark_frame_filenames, dark_frame_save_dir)
    flat_fields = store_matrix(flat_fields, flat_field_filenames, flat_field_save_dir)
    print("Loading filenames, done.")
    return scans, dark_frames, flat_fields