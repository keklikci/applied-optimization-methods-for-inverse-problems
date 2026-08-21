"""Script export of 01_exercise01.ipynb."""

import matplotlib.pyplot as plt
import numpy as np

import aomip

# ## Create phantom with small rectangle in the middle

# Create a phantom with a small rectangle in the middle and forward project it with the given code. What output do you expect?

size = (128, 128)
phantom = np.zeros(size)

arc = 360
nangles = 420
sino = aomip.radon(phantom, [180], np.linspace(0, arc, nangles), 1000, 150)
plt.imshow(sino, cmap="gray")

# Now play around with the above `radon` command. Try to find out what the different parameters do, and how they affect the sinogram.

# ## More phantoms
#
# Create different phantoms and forward projection them. To goal of the exercise is you to start finding patters. Start by guessing the sinogram to a phantom. Try to create specific sinograms. Play around a little bit.
#
# Ideas for phantoms:
# - Change the rectangle with a circle
# - Move it to a different location
# - Try adding different shapes in one phantom


# ## 3D phantoms
#
# 2D phantoms are nice and well, but many interesting things are done in 3D. Hence, create the 3D Shepp-Logan phantom,
# and forward project it (how long will it take :D)

phantom = aomip.shepp_logan([32, 32, 32])

# C
