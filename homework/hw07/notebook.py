"""Script export of notebook.ipynb."""

# ### Notebook
#
# * Module: Applied Optimization Methods for Inverse Problems
# * Author: kaanguney.keklikci@tum.de
# * Please find all images produced by all scripts in this notebook.

# ##### Homework 1 & 2: Proximal Operators & ADMM
#
# * LASSO
#
#     * `λ` = 0.95τ / norm.
#     * Norm is computed via power iterations. 
#     * Visualizations are generated for a set of `τ` over `np.logspace(-3, 6, 10)`.

# | ![τ = 0.1](images/notebook/tau_0.1.png) | ![τ = 1.0](images/notebook/tau_1.0.png) |
# |:---:|:---:|
# | τ = 0.1 | τ = 1 |
#
# | ![τ = 10.0](images/notebook/tau_10.0.png) | ![τ = 100.0](images/notebook/tau_100.0.png) |
# |:---:|:---:|
# | τ = 10 | τ = 100 |
#
# | ![τ = 1000.0](images/notebook/tau_1000.0.png) | ![τ = 10000.0](images/notebook/tau_10000.0.png) |
# |:---:|:---:|
# | τ = 1000 | τ = 10000 |
#
# | ![τ = 100000.0](images/notebook/tau_100000.0.png) | ![τ = 1000000.0](images/notebook/tau_1000000.0.png) |
# |:---:|:---:|
# | τ = 100000 | τ = 1000000 |

# ##### Homework 2 & 3 TV Regularization
#
# * Stacked operators.
# * Implemented relevant proximals.
# * Below is the default smoothing function provided in the assignment template.

# | ![Smoothing Function](images/notebook/smooth.png) |
# |:---:|
# | Smoothing Function |
