## Overview

* Algorithms related to this homework are implemented with pure object oriented programming (OOP) and are blackformatted.

* `POGM` inherits from `Optimization` from the top-level `aomip` development directory where concrete implementations are stored.

* `.tif` files are clear reconstructions of the target image. On the other hand, `.png` files seem to have some streaky lines among black pixels. Regardless, reconstruction is clearly observable and colormaps are attached to these plots. Additionally, this effect is also reproducible with unfiltered forward projection, i.e., the sinogram.

* Every task is implemented, although I did not manage to get nice reconstructions out of Homework 3: Low Dose X-ray CT.

* All implementation is available under [hw06](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw06).

* Output images are stored in two different directories, i.e. [notebook](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw06/images/notebook) and [pogm](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw06/images/pogm). Reasoning behind this is explained in the following section.

* Please execute the scripts from [hw06](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw06) directory.

* Please utilize the [notebook](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw06/images/notebook) for viewing output images and quality of reconstructions. For clear reconstructions, load and show `.tif` files stored in [pogm](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw06/images/pogm).

## Homework 1: Proximal Optimized Gradient Method

* Running `pogm.py` standalone will produce the reconstruction and save outputs to two different directories, one for displaying reconstructions in the [Juypter notebook](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/homework/hw06/notebook.ipynb) and the other for the [challenge](https://submission.ciip.in.tum.de/) submission.

* Reconstruction quality-wise, the algorithm can compete with FPGM, although the convergence rate is by far slower, as opposed to being 2 times faster. Most likely, this is due to the difference in problem setting. 

* I referenced [Fessler's](https://web.eecs.umich.edu/~fessler/course/598/l/n-05-prox.pdf) notes to implement the algorithm and naturally the Lipschitz constant. The step size is crucial for all optimization algorithms but for proximal methods, I can confidently say that they are more sensitive to this parameter compared to traditional methods. So, I had to tweak number of iterations multiple times. 

* As an overall result, the rate of decrease is faster for POGM in the first few iterations, however, the overall convergence rate is much slower compared to FPGM, with the difference being a couple iterations compared to around 50 iterations.
    
    
## Homework 2: Expectation Maximization

* Please see a replica of in-class solution for expectation maximiation in this [Jupyter notebook](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/homework/hw06/mlem.ipynb).
    
## Homework 3: Low Dose X-ray CT
    
* Runnning `reconstruct.py` standalone will produce the reconstruction for both high-dose and low-dose targets.

* I modified and modularized the provided scripts to extract the data from the server under the shared directory.

* Unfortunately, my reconstructions turned out to be horizontally spread and vertically squeezed enough to not make sense. Therefore, I deliberately decided not to provide the exports for this task.

* `reconstruct.py` performs the reconstruction and saves images to the [mayo](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw06/images/mayo) directory.

* `config.py` fetches the optimization parameters for the operator.

* `loader.py` loads the data and the metadata.