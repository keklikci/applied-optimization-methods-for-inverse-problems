## Overview

* The algorithms related to this homework are implemented with pure object oriented programming (OOP) and are blackformatted.

* Every task is implemented and outputs reasonably well reconstructions (especially Fast Proximal Gradient Method, i.e. FPGM). 

* In some of the reconstructions, some artifacts and contrast irregularities seem to be persistent, specifically for Iterative Soft Thresholding Algorithm, i.e. ISTA. There are several possible reasons behind this, such as pixel values being so close to each other, making up for low contranast exports that blend in with black background. 

* Another potential reason for artifacts is unclipped exports and exporting interchangeably with both `tifffile` and `matplotlib`. Sometimes, although rare, I find it mandatory to export the output to visualize the underlying reconstructions. `tifffile` occasionally exports black images, although reconstructions in these cases are revelead through `matplotlib` grayscale colormaps. Then again, I believe this is a suboptimal approach and is the leading cause of artifacts on some of my grayscale outputs. Therefore, I am actively looking for a method to clip and scale in between iterations and prior to the final export.

* All implementation is available under [hw05](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw05). Again, as I introduced subclassing for my optimization methods in Homework 4, the source code for Homework 5 is implemented with a similar fashion.

* I also spotted some errors in my code regarding error accumulation in callbacks, hyperparameter setting and insufficient number of iterations for ISTA. Callbacks and hyperparameter setting is now fixed and ISTA (although still artifactual) produces a better reconstruction for a relatively high difficulty challenge image.

* Please find all images in [images](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw05/images/). Every method that involves some proximal operator has an accordingly set tag on the export image. Please also note that some of the images are black and not all callbacks are informative or meaningful. Nevertheless, they are included because they are a part of my work and they also convey some valuable information about the parameter setup in my optimization methods. More specifically, you can track down `alpha` or `beta` values depending on the algorithm to see and examine how they correlate to the reconstructed image.

* Please execute the scripts from the [hw05](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw05) directory. Note that a reconstructed ground truth image (credits to @david.frank), called `htc2022_05c_recon.tif` must be placed inside `images` as the input image to the algorithms. Compared to the previous homework, i.e. Homework 4, this challenge image has a difficulty level `c`, meaning that it lies in the most difficult and rewarding category.

## Homework 1: Proximal Operators

  - ### i) Proximal Operators Module

    - I packaged all proximal operators in a module called `proximal_operators`. 
    
    - All proximal operators specified in the assignment, i.e., `constant`, `l2`, `huber` are implemented under this module.
    
    - Since proximal operators have to be ported to some of the optimization algorithms as requested in Homework 2, I also included non-negativity projection and soft-thresholding, i.e. proximal operator for L1 norm, in this module.
    
## Homework 2: Proximal Gradient Method

  - ### i) Proximal Gradient Method

    - As I mentioned in part i), the proximal operator of L1 norm is soft-thresholding, which is implemented as the accepted proximal operator by class `ISTA`.

    - Although I successfully ported the proximal operator to this algorithm, some artifacts do exist due to suboptimal parametrization of the optimization loop. 

    - Still, I was able to get the reconstruction out of the algorithm, but some of pixels are too pushed back to the lower bound 0 and some of the pixels are too bright and scattered. Algorithm itself is correct, but to reiterate, it's very difficult to find a nice parametrization or fast convergence for this method. Arguably, the option that makes the most sense is to increase the number of iterations and include some `epsilon` termination condition.

    - Projected gradient descent shows demonstrates some similar behavior but with a lot less artifacts. The difference compared to `FPGM` results from the difference in export API,  i.e., visualization via `matplotlib` grayscale colormap.

    - Overall, optimization algorithms are implemented in a way that corresponding proximal operators from the module are ported to the optimization methods.
  
  - ### ii) Fast proximal gradient method
  
    - FPGM reconstructions are excellent and far superior to any algorithm I have managed to implement so far.

    - Due to custom momentum calculation (I implemented the second bulleted approach in the assignment document), algorithm converges blazingly fast and callbacks always demonstrate a monotonically increasing reconstruction error from this point forward.

    - Convergence rate can simply be verified by inspecting one of the callback outputs and see monotonically increasing error plot (although small increases in magnitude) for reconstruction, meaning that I am able to reconstruct the image in only a handful of iterations. 
    
  - ### iii) Uniqueness of formulation
  
    - Convergence for nonnegativity proximal operator (first variant) is a lot faster than L2 proximal operator (second variant).

    - One possible explanation is that L2 proximal operator is a softer constraint than nonnegativity projection, meaning that it allows for a broader range of pixel mapping to larger field `F` over a domain of numbers with any sign. This makes the algorithm more difficult to converge and requires more iterations.
    
  - ### iv) Elastic Net Formulation
  
    - Elastic net formulation requires to add L1 and L2 penalties to the gradient, which I implemented in `elastic.py`. Note that optimization method is FPGM-like and the only difference is the added penalties.

    - It's interesting to see that despite more regularization, more regularization does not always necessarily mean slower convergence. Looking at the callbacks, in fact, elastic net formulation converges by a difference of 5 more iterations on average.
    
## Homework 3: Restart conditions
    
* After some experimentation with restart (hardcoded, duplicated test scripts), I could not find a set of values for any algorithm that converged after periodical restarts at every `k` iterations. 

* For further analysis and more detailed experimentation, I am in the process of writing a script to access and modify my optimization methods from different modules.