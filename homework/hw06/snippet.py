
from skimage.io import imread, imsave
import json
import tifffile
import numpy as np

def load_tiff_stack_with_metadata(file):
    with tifffile.TiffFile(file) as tif:
        data = tif.asarray()
        metadata = tif.pages[0].tags["ImageDescription"].value
    metadata = metadata.replace("'", "\"")
    try:
        metadata = json.loads(metadata)
    except:
        print('The tiff file you try to open does not seem to have metadata attached.')
        metadata = None
    return data, metadata

# load data, metadata
source = "/srv/ceph/share-all/aomip/mayo_clinical/out/L310_flat_fan_projections_fd.tif"
data, metadata = load_tiff_stack_with_metadata(source)

breakpoint()

# extract angles in degree
angles = np.degrees(np.array(metadata["angles"])[: metadata["rotview"]])

# setup some spacing and sizes
image_size = [512, 512]
voxel_size = 0.7 # can be adjusted
vox_scaling = 1 / voxel_size
vol_spacing = vox_scaling
vol_size = image_size

# size of detector
det_count = sino_data.shape[:-1]
det_spacing = vox_scaling * metadata["du"]

# Distances from source to center, and center to detector
ds2c = vox_scaling * metadata["dso"]
dc2d = vox_scaling * metadata["ddo"]

breakpoint()
