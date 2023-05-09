## Important
* Please execute `python -B tests/test_hw02.py` from repository root directory for all the output except the submission for the challenge.
* Concrete implementations are in `aomip` directory as described in the description. For the context of this assignment, this corresponds to the implementation of `gradient_descent.py`. All iterative methods are looped over using the gradient descent algorithm in `tests/test_hw02.py`. Hence, executing this file yields the results.
* No external packages were used in the implementation other than provided.

## Remarks
* Script points to the `/srv/ceph/share-all/aomip/6983008_seashell/` directory.
* [Dataset](https://zenodo.org/record/6983008#.ZFnciexByu4) was used for all methods.
* Script writes all results to the `homework/hw2/output` directory of the corresponding task.

## Homework 1: Preprocessing again
* For this task, every image, i.e., `.tif` file was iteratively sliced by the middle row index and stacked side-by-side to obtain a shallow sinogram. Implementation is available via `homework/hw02/slicing.py`. For output, please execute `python -B tests/test_hw02.py`.

## Homework 2: Iterative Reconstruction
* Gradient descent is implemented and parametrized by number of iterations, learning_rate, the data to be to be optimized, and the gradient.
* The calculation of the gradient is cost function depended, it was parametrized for readability and re-usability.

## Homework 3: Solving CT Problems
* The `XrayOperator.py` class was used to generate the system matrix (sparse) `A` and the formulation in the assignment for obtaining the derivative was implemented.
* The original and reconstructed image are plotted side-by-side and [results](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw02/output/least_squares) are available via the subfolder corresponding to the task in the output directory.

### ℓ-2 Norm square
* For every reconstruction method from least squares onwards, a term to the gradient was added. The added term is a byproduct of calculation of the gradient.
* [Results](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw02/output/tikhonov_regularization) are written to the hyperlinked directory. Tikhonov term added to the gradient is `x`.

### Huber Functional
* [Results](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw02/output/huber_functional) are written to the hyperlinked directory. Gradient term is calculated w.r.t the formualtion in the homework statement.
* Reconstruction is comparable to the Tikhonov regularization but worse than least squares. Selection of delta is a purely experimental process, this is the primary reason for performance discrepancy.

### Fair potential
* [Results](https://gitlab.lrz.de/IP/teaching/applied-optimization-methods-for-inverse-problems/aomip-kaan-guney-keklikci/-/tree/main/homework/hw02/output/fair_potential) are written to the hyperlinked directory. Gradient term is calculated w.r.t the formualtion in the homework statement.
* Unfortunately, I could not achieve a nice reconstruction with this method. Finding the right delta is extremely important for fair potential. If selected very small, regularization decreases heavily due to the inversion of the term at the denominator.

## Homework 5: Iterative algorithm vs FBP
* Iterative methods run longer, as they need very good calibration of parameters such as the learning rate and delta selection.
* FBP is more stable than iterative methods because gradient could make the optimization process shaky. FBP also takes advantage of transforming the signals to their frequency domain. Therefore, it is more efficient and fast. Also, filters, to my experience in my implementations so far, affect FBP far more than it affects iterative methods.
