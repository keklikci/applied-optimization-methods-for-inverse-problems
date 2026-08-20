"""Script export of notebook.ipynb."""

# ### Notebook
#
# * Module: Applied Optimization Methods for Inverse Problems
# * Author: kaanguney.keklikci@tum.de
# * Please find all images produced by all scripts in this notebook.

# ##### Homework 1 & 2: Homework 1: Proximal Operators & Proximal Gradient Method

# * Proximal Operators
#
#     * Updated the proximal operator for Elastic net formulation.
#
#     * Please find all proximal operators in `operator.py` in the top-level development module `aomip`.

# * FPGM
#
#     * Visualizations are generated for a set of `λ` over `np.logspace(-3, 6, 10)`.

# | ![λ = 0.001](images/notebook/fpgm/lambda_0.001.png) | ![λ = 1.0](images/notebook/fpgm/lambda_0.01.png) |
# |:---:|:---:|
# | λ = 0.001 | λ = 0.01 |
#
# | ![λ = 0.1](images/notebook/fpgm/lambda_0.1.png) | ![λ = 1.0](images/notebook/fpgm/lambda_1.0.png) |
# |:---:|:---:|
# | λ = 0.1 | λ = 1.0 |
#
# | ![λ = 10.0](images/notebook/fpgm/lambda_10.0.png) | ![λ = 100.0](images/notebook/fpgm/lambda_100.0.png) |
# |:---:|:---:|
# | λ = 10 | λ = 100 |
#
# | ![λ = 1000.0](images/notebook/fpgm/lambda_1000.0.png) | ![λ = 10000.0](images/notebook/fpgm/lambda_10000.0.png) |
# |:---:|:---:|
# | λ = 1000 | λ = 10000 |
#
# | ![λ = 100000.0](images/notebook/fpgm/lambda_100000.0.png) | ![λ = 1000000.0](images/notebook/fpgm/lambda_1000000.0.png) |
# |:---:|:---:|
# | λ = 100000 | λ = 1000000 |

# * Elastic Net Formulation
#
#     * For higher sparsity, preference of l1 regularization over l2 regularization.
#
#     * For smoothness and noise reduction, preference over l2 regularization over l1 regularization.
#
#     * Visualizations are generated for a set of `λ` over `np.logspace(-3, 6, 10)`.

# | ![λ = 0.001](images/notebook/elastic/lambda_0.001.png) | ![λ = 1.0](images/notebook/elastic/lambda_0.01.png) |
# |:---:|:---:|
# | λ = 0.001 | λ = 0.01 |
#
# | ![λ = 0.1](images/notebook/elastic/lambda_0.1.png) | ![λ = 1.0](images/notebook/elastic/lambda_1.0.png) |
# |:---:|:---:|
# | λ = 0.1 | λ = 1.0 |
#
# | ![λ = 10.0](images/notebook/elastic/lambda_10.0.png) | ![λ = 100.0](images/notebook/elastic/lambda_100.0.png) |
# |:---:|:---:|
# | λ = 10 | λ = 100 |
#
# | ![λ = 1000.0](images/notebook/elastic/lambda_1000.0.png) | ![λ = 10000.0](images/notebook/elastic/lambda_10000.0.png) |
# |:---:|:---:|
# | λ = 1000 | λ = 10000 |
#
# | ![λ = 100000.0](images/notebook/elastic/lambda_100000.0.png) | ![λ = 1000000.0](images/notebook/elastic/lambda_1000000.0.png) |
# |:---:|:---:|
# | λ = 100000 | λ = 1000000 |
