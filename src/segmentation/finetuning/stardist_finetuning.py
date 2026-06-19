from __future__ import absolute_import, division, print_function, unicode_literals

import sys
from pathlib import Path as StdPath

import matplotlib
import numpy as np

matplotlib.rcParams["image.interpolation"] = "none"

from glob import glob

from csbdeep.utils import Path, normalize
from csbdeep.utils.tf import limit_gpu_memory
from stardist import (
    calculate_extents,
    fill_label_holes,
    gputools_available,
    random_label_cmap,
)
from stardist.models import Config2D, StarDist2D
from tifffile import imread
from tqdm import tqdm

np.random.seed(42)
lbl_cmap = random_label_cmap()

X = sorted(
    glob(
        "data/input/segmentation_finetuning/finetuning_data/stardist/train/images/*.tif"
    )
)
Y = sorted(
    glob(
        "data/input/segmentation_finetuning/finetuning_data/stardist/train/masks/*.tif"
    )
)
assert all(Path(x).name == Path(y).name for x, y in zip(X, Y))


X = list(map(imread, X))
Y = list(map(imread, Y))
n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]


axis_norm = (0, 1)  # normalize channels independently
# axis_norm = (0,1,2) # normalize channels jointly
if n_channel > 1:
    print(
        "Normalizing image channels %s."
        % ("jointly" if axis_norm is None or 2 in axis_norm else "independently")
    )
    sys.stdout.flush()

X = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X)]
Y = [fill_label_holes(y) for y in tqdm(Y)]

assert len(X) > 1, "not enough training data"
rng = np.random.RandomState(42)
ind = rng.permutation(len(X))
n_val = max(1, int(round(0.15 * len(ind))))
ind_train, ind_val = ind[:-n_val], ind[-n_val:]
X_val, Y_val = [X[i] for i in ind_val], [Y[i] for i in ind_val]
X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]
print("number of images: %3d" % len(X))
print("- training:       %3d" % len(X_trn))
print("- validation:     %3d" % len(X_val))


# 32 is a good default choice (see 1_data.ipynb)
n_rays = 32

# Use OpenCL-based computations for data generator during training (requires 'gputools')
use_gpu = True and gputools_available()

# Predict on subsampled grid for increased efficiency and larger field of view
grid = (2, 2)

conf = Config2D(
    n_rays=n_rays,
    grid=grid,
    use_gpu=use_gpu,
    n_channel_in=n_channel,
)
print(conf)
vars(conf)


limit_gpu_memory(0.8, allow_growth=False, total_memory=16000)

model = StarDist2D(None, str(StdPath("data/models/segmentation/stardist_finetuned")))

median_size = calculate_extents(list(Y), np.median)
fov = np.array(model._axes_tile_overlap("YX"))
print(f"median object size:      {median_size}")
print(f"network field of view :  {fov}")
if any(median_size > fov):
    print(
        "WARNING: median object size larger than field of view of the neural network."
    )


def random_fliprot(img, mask):
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim)))
    mask = mask.transpose(perm)
    for ax in axes:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask


def random_intensity_change(img):
    img = img * np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)
    return img


def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    # add some gaussian noise
    sig = 0.02 * np.random.uniform(0, 1)
    x = x + sig * np.random.normal(0, 1, x.shape)
    return x, y


model.train(X_trn, Y_trn, validation_data=(X_val, Y_val), augmenter=augmenter)

print("training finished, saving model to disk...")
