## Overview

* The algorithms related to this homework are implemented with pure object oriented programming (OOP) and are blackformatted.

* Every task is implemented and outputs reasonably well reconstructions (especially Fast Proximal Gradient Method, i.e. FPGM). 

* In some of the reconstructions, some artifacts and contrast irregularities seem to be persistent, specifically for Iterative Soft Thresholding Algorithm, i.e. ISTA. There are several possible reasons behind this, such as pixel values being so close to each other, making up for low contranast exports that blend in with black background. 

* Another potential reason for artifacts is unclipped exports and exporting interchangeably with both `tifffile` and `matplotlib`. Sometimes, although rare, I find it mandatory to export the output to visualize the underlying reconstructions. `tifffile` occasionally exports black images, although reconstructions in these cases are revelead through `matplotlib` grayscale colormaps. Then again, I believe this is a suboptimal approach and is the leading cause of artifacts on some of my grayscale outputs. Therefore, I am actively looking for a method to clip and scale in between iterations and prior to the final export.

* All implementation is available under [hw05](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw05). Again, as I introduced subclassing for my optimization methods in Homework 4, the source code for Homework 5 is implemented with a similar fashion.

* I also spotted some errors in my code regarding error accumulation in callbacks, hyperparameter setting and insufficient number of iterations for ISTA. Callbacks and hyperparameter setting is now fixed and ISTA (although still artifactual) produces a better reconstruction for a relatively high difficulty challenge image.

* Please find all images in [images](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw05/images/). Every method that involves some proximal operator has an accordingly set tag on the export image. Please also note that some of the images are black and not all callbacks are informative or meaningful. Nevertheless, they are included because they are a part of my work and they also convey some valuable information about the parameter setup in my optimization methods. More specifically, you can track down `alpha` or `beta` values depending on the algorithm to see and examine how they correlate to the reconstructed image.

* Please execute the scripts from the [hw05](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw05) directory. Note that a reconstructed ground truth image (credits to @david.frank), called `htc2022_05c_recon.tif` must be placed inside `images` as the input image to the algorithms. Compared to the previous homework, i.e. Homework 4, this challenge image has a difficulty level `c`, meaning that it lies in the most difficult and rewarding category.