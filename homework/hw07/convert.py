# -*- coding: utf-8 -*-

# ********************************** #
# Author: kaanguney.keklikci@tum.de  #
# Date: 05.07.2023                   #
# ********************************** #

import os
import matplotlib.pyplot as plt

def main():
    os.makedirs("images/notebook", exist_ok=True)
    images = os.listdir("images")
    images = [image for image in images if image.endswith(".tif")]
    for i, image in enumerate(images):
        filename = os.path.join(os.getcwd(), "images", image)
        plt.axis("off")
        export = plt.imread(filename)
        export = plt.imshow(export, cmap="gray")
        plt.colorbar(export)
        plt.tight_layout()
        idx = filename.find("_")
        plt.title(f"τ ={filename[(idx + 1):len(filename) - 4]}")
        plt.savefig("images" + "/" + "notebook" + "/" + images[i].replace(".tif", ".png"))
        plt.clf()

if __name__ == "__main__":
    main()