## Overview

* As always, algorithms related to this homework are implemented with pure object oriented programming (OOP) and are blackformatted.

* An improvement that needs to be done is to move the `Optimization` class in `optimize.py` to the top level module `aomip`. This improvement is necessary for avoiding some repetitive code, which I will provide support for in the next homework, along with additional implementation leftover from previous tasks.

* Not a direct outcome of this homework, but I have come to the realization that my image viewer was incapable of opening floating numbered arrays. This was the reason I published some artifactory images exported with `matplotlib` while casting to `np.uint8`, which is a hacky workaround that produces the artifacts. I will export some previous artifactory images again with `tifffile` because they seem to be visible on other image viewers.

* Every task is implemented, although I did not manage to get nice reconstructions out of `Homework 3: Low Dose X-ray CT`. Scripts for generating the reconstructions are provided under `homework/hw06`.

* All implementation is available under [hw06](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw06).

* Please find all images in [images](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw06/images/).

* Please execute the scripts from the [hw06](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw06) directory. Note that a reconstructed ground truth image, `htc2022_05c_recon.tif`, must be placed inside `images` as the input image, specifically for `pogm.py`. Again, as in previous homework, this is of type `c` phantom, which is the most difficult phantom included in challenge dataset.

## Homework 1: Proximal Optimized Gradient Method

* I copied over `proximal_operators` module and `Optimization` class from previous homework to the relevant directory. As I outlined before, they will be moved to the top level module `aomip` to avoid code repetition as a major refinement to the overall project structure and my work. 

* Running `pogm.py` standalone will produce the reconstruction and save the output image and the callback plot to `homework/hw06/images`.

* Reconstruction quality-wise, the algorithm can compete with `Fast Proximal Gradient Method`, although the convergence rate is by far slower, as opposed to being 2 times faster. Most likely, this is due to the difference in problem setting. 

* I referenced [Fessler's](https://web.eecs.umich.edu/~fessler/course/598/l/n-05-prox.pdf) notes to implement the algorithm and naturally the Lipschitz constant. The step size is crucial for all optimization algorithms but for proximal methods, I can confidently say that they are more sensitive to this parameter compared to traditional methods. So, I had to tweak number of iterations multiple times. 

* As an overall result, the rate of decrease is faster for `POGM` in the first few iterations, however, the overall convergence rate is much slower compared to `FPGM`, with the difference being a couple iterations compared to around 50 iterations.
    
    
## Homework 2: Expectation Maximization

* Please see the in-class solution, i.e., `homework/hw6/mlem.ipynb`.
    
## Homework 3: Low Dose X-ray CT
    
* Please run the script called `reconstruct.py` to save both reconsructions to the [images](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw06/images/).

* I modified and modularized the provided scripts to extract the data from the server under `/srv/ceph/share-all/aomip/mayo_clinical/out/` directory.

* Unfortunately, my reconstructions turned out to be horizontally spread and vertically squeezed enough to not make sense. Therefore, I deliberately decided not to provide the exports for this task.

* Implementation is complete for reconstruction of both doses and can be executed by running `reconstruct.py` from the homework folder.

* As an additional note, `config.py` fetches the optimization parameters for the operator and `loader.py` loads the data and the metadata.