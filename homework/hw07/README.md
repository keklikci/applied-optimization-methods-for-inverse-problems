## Overview

* Algorithms related to this homework are implemented with pure object oriented programming (OOP) and are blackformatted.

* Lots of refinements are made available as of this release. The main development, top-level package [aomip](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/aomip) hosts concrete implementations of the following.

    * [Power iterations](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/aomip/power.py)
    * [Finite differences](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/aomip/derivative.py)
    * [Optimization](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/aomip/optimize.py)
    * [Smoothing](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/aomip/smooth.py)
    * [Operators](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/aomip/operator.py)
    
* Previously, an image had to be placed to point scripts toward the target. This now points to the shared server directory where the image is stored.

* Every script was executed from the [hw07](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw07), utilizing the concrete implementations in the top-level directory.

* Every task is implemented, although I did not manage get reconstructions from TV regularization due to difficulties in dimensional mismatches.

* All implementation is available under [hw07](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw07).

* As an improvement to previous releases, I have prepared a [Jupyter notebook](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/homework/hw07/notebook.ipynb) for you to review my work more conveniently. I will add support for this to my previous releases for backward-compatibility purposes.

## Homework 1: More Proximal Operators

* Please find the link in Overview to all operators (including transforms) of the project.

* I implemented the `L11` AND `L21` proximal operators. They are available as concrete implementations in the linked script above.
    
    
## Homework 2: ADMM

* I implemented this algorithm and had difficulties tuning the parameters passed to the proximal operators. 

* I computed the norm for further computation of `τ` and had success with some straightforward parameter setting rather than sophisticated approaches like line or logspacing over a possible set of values.

* I followed the template equation to set for `µ` and had nice results for LASSO problem, although some of the results are artifactual and artifacts can be viewed from the color bars attached to the figures.

* The class that implements `ADMM` for LASSO problem is called `LADDM` in my implementation.
    
## Homework 3: TV Regularization
    
* I implemented the list of derivatives as the usual XrayTransform and output of finite differences.

* Please see the [implementation](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/homework/hw07/tv.py). Proximal operators are set as `L11` and `L22` norms.

* Forward output must be split and stacked to make the script work, currently I could not accomplish this.

* I plotted the smoothing function and placed it in the Jupyter notebook for images with a colormap for my reference to see how color shifts are smoother around the edges.