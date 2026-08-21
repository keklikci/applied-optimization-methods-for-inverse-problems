"""Script export of notebook.ipynb."""

# ### Notebook
#
# * Module: Applied Optimization Methods for Inverse Problems
# * Author: kaanguney.keklikci@tum.de
# * Please find all images produced by all scripts in this notebook.

# ##### Homework 1: Proximal Optimized Gradient Method
#
# * Test for `λ` over `np.logspace(-3, 6, 10)`.
# * LASSO proximal for POGM is defined as `np.sign(x) * np.maximum(np.abs(x) - lmbd, 0)`.
# * The proximal shrinks the coefficients towards zero based on the regularization parameter `λ`.

# | ![λ = 0.1](images/notebook/lambda_0.001.png) | ![λ = 1.0](images/notebook/lambda_0.01.png) |
# |:---:|:---:|
# | λ = 0.001 | λ = 0.01 |
#
# | ![λ = 10.0](images/notebook/lambda_0.1.png) | ![λ = 100.0](images/notebook/lambda_1.0.png) |
# |:---:|:---:|
# | λ = 0.1 | λ = 1 |
#
# | ![λ = 1000.0](images/notebook/lambda_10.0.png) | ![λ = 10000.0](images/notebook/lambda_100.0.png) |
# |:---:|:---:|
# | λ = 10 | λ = 100 |
#
# | ![λ = 100000.0](images/notebook/lambda_1000.0.png) | ![λ = 1000000.0](images/notebook/lambda_10000.0.png) |
# |:---:|:---:|
# | λ = 1000 | λ = 10000 |
#
# | ![λ = 100000.0](images/notebook/lambda_100000.0.png) | ![λ = 1000000.0](images/notebook/lambda_1000000.0.png) |
# |:---:|:---:|
# | λ = 100000 | λ = 1000000 |
