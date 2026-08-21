"""Script export of 01_exercise02.ipynb."""


# ## Backprojection
#
# Here, we will explore the backprojection. The goal of this exercise is to explore the adjoint operations of the forward projection.
#
# First, instead of using the `radon` function now directly, create the `XrayTransform` yourself (the interface,
# is pretty similar to the `radon` function one ;-))
#
# Next, create a phantom and forward project it to get the sinogram. Now apply the adjoint of the X-ray Transform and look at the result. What do you see?
#
# Try it for different number of positions in the trajectory and different phantoms.


# ## Reconstruction
#
# Can you think of an algorithm to solve the problem, i.e. get as close to the original phantom as possible? Discuss with your neighbours and play around wiht possible solutions a bit!
