## Overview

* No external packages were used in the implementation (except for [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft2.html) for Fast Fourier Transformation) other than provided.

* Concrete implementations for this homework, i.e., homework 3, can be found under two directories, namely [aomip](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/aomip) and [hw03](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw03).

* As of the submission data, concrete implementations of conjugate gradient, convolution, and deconvolution are provided under this release. Rest of the methods will be implemented in the near future alongside some improvements for previous homeworks (if applicable).

* Implementation follows an OOP (Object Oriented Programming) approach, unittests are available for every class under [tests](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/tests). Unittests are developed using Python's own [unit testing framework](https://docs.python.org/3/library/unittest.html#). Most commonly employed methods during implementation are `assertRaises(exception, callable, *args, **kwds)` and `assertEqual(first, second, msg=None)`.

* Some of the unittests serve for exporting outputs. They don't necessarily test something related to implementation.

* Please execute all tests from the root directory. This will be further clarified in the following sections.

* All input and output images can be found under [images](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw03/images) directory.

## Homework 1: More gradient based methods

  - ### iv) Conjugate Gradient
    -  From the root directory, run the following command to write the reconstructed image to the [images](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw03/images) directory. Note that all output images are already provided under the directory, as a result of this command they will be overwritten. Concrete implementation of the conjugate gradient is available [here](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/homework/hw03/conjugate_gradient.py).
        
    `python -B tests/test_conjugate_gradient.py`
    
    - After executing the test, the optimization algorithm will run and export the output. 
    
    - [Wikipedia](https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_resulting_algorithm) was referenced to implement the method.
    
    - In previous reconstruction algorithms such as (steepest) gradient descent, learning rate, i.e., α, was naively selected, typically as 0.01 or 0.001. In addition, convergence was achieved around 100s of iterations. With conjugate gradient descent, convergence is achieved much faster in 10s of iterations. Specifically for [hw03](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw03), reconstructed image was exported to the output folder at the end of 10 steps.
    
    - Instead of moving in the largest descent direction, we iteratively calculate an optimal step size and solve a large system of linear equations by using a sparse matrix.

## Homework 2: Solving problems other than X-ray CT

- ### Denoising

    - From the root directory, run the following command to sample some noise, combine it with the [input image](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/homework/hw03/images/input/htc2022_04b_recon.tif). Note that the image is an example of groundtruth images which were provided after last week's session.
    
    `python -B tests/test_noise.py`
    
    - After executing the test, images wtih filenames prefixed by the noise type will be written to the output directory.
    
    - One important remark here is that Poisson noise is not additive unlike Gaussian noise, where we can add the generated noise to shift the standard deviation of the image. To generate noise from a Poisson distribution for an image, we pass the image to the `np.random.poisson`.
    
    - Also note that, to make the results reproducible, a seed is provided within the concerete implementation of Noise class, which is also available [here](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/homework/hw03/noise.py).
    
- ### Deblurring

    - From the root directory, run the following command to blur and deblur the [input image](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/homework/hw03/images/input/htc2022_04b_recon.tif) with Gaussian kernel. 
    
    `python -B tests/test_blur.py`
    
    - After executing the test, both blurred and deblurred images will be written to the [images](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw03/images) directory.
    
    - Implementation of Gaussian kernel was referenced from [Wikipedia](https://en.wikipedia.org/wiki/Gaussian_blur).
    
    - As described in the problem statement, the coefficients of the Fast Fourier Transform for both the input image (signal) and the Gaussian filter is multiplied with each other to blur the image.
    
    - Conversely, deconvolution takes in the blurred image as input, projects the input image to the Fourier space and finally scales it by the Gaussian filter, again in the Fourier space. For a closer look into the implementation, please take a look at `__convolution(self, image)` and `__deconvolution(self, image)` functions [here](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/blob/main/homework/hw03/blur.py).
    