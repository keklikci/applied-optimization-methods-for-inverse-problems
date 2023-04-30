# !/usr/bin/env python
# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 30.04.2023                   #
# ********************************** #

import time
from data_loader import load_files

def main():
    start_time = time.time()
    print("Data loader executing...")
    scans, dark_frames, flat_fields = load_files()
    print("Data loader execution, done.")
    # TODO: flat-field correction
    # TODO: save progress
    print("Total execution time: {:.2f} seconds.".format(time.time() - start_time))

if __name__ == "__main__":
    main()




