# HR/LR Coordinate Conventions

## Purpose

This note summarizes the coordinate conventions used when working with the current
high-resolution (HR) TIFF slices and low-resolution (LR) MINC slices, together
with the affine-based mapping rules between them.

The main goal is to avoid confusion between:

- image array coordinates
- full-image pixel coordinates
- world coordinates from the affines


## Coordinate Spaces

There are three useful spaces.

### 1. Image array coordinates

These are the coordinates used to index a loaded 2D image array.

- Convention: `(z, x)`
- Meaning: `(row, col)`
- Indexing form: `img[z, x]`

This is the natural convention when working directly with numpy arrays or image
display code.

### 2. Full-image pixel coordinates

These are pixel coordinates in the full HR or LR image reference frame, not in a
local crop.

- Convention: `(x, z)`
- Meaning: geometric pixel axes in the full slice

If a point is defined inside a crop, it must be converted to full-image pixel
coordinates before performing HR/LR mapping.

### 3. World coordinates

These are physical coordinates obtained with the affine matrices.

- Convention: `(x, y, z)`
- Units: physical space defined by the input affines

World coordinates are useful mainly at the affine boundary. Most processing can
stay in image or full-image pixel coordinates.


## HR Data Convention

For the current HR data:

- the raw 2D image array is indexed as `(z, x)`
- the HR affine expects voxel coordinates in `(x, y, z)` order

This means that a point read from the HR array must be reordered before being
passed to the affine.

If a point is observed on the raw HR array as:

```python
(z_hr, x_hr)
```

then the corresponding HR voxel to feed into the affine is:

```python
(x_hr, 0, z_hr)
```


## LR Data Convention

For the current LR data:

- the raw 2D slice view is indexed as `(z, x)`
- the LR affine uses voxel order `(z, y, x)`

This means that after applying the inverse LR affine, the returned voxel already
follows LR voxel order:

```python
(z_lr, y_lr, x_lr)
```

For 2D work on a single LR slice, the useful coordinates are therefore:

```python
(z_lr, x_lr) = (lr_voxel[0], lr_voxel[2])
```

This is not an extra swap applied after the inverse affine. It is simply the
correct way to read the LR voxel result, because LR voxel order is already
`(z, y, x)`.


## Mask-Generation Coordinate Flow

The preprocessing mask-generation helpers use a small internal convention chain
that is worth writing down explicitly.

For both HR and LR mask generation:

- sliced surface contours in world space are represented as `(x, z)`
- after mapping through the inverse affine, image-space contour arrays are kept as `(z, x)`
- before exporting GeoJSON, those contour arrays are converted to polygon coordinates in `(x, z)`
- bbox JSON files are saved in full-image coordinates as `(x_min, x_max, z_min, z_max)`

This matches the practical distinction between:

- image indexing work, which is easier in `(z, x)`
- geometric polygon export, which is easier in `(x, z)`


## Vertical Orientation of the Raw Images

The HR and LR raw rasters are vertically opposite in world `z`.

In practice:

- HR raw row `0` corresponds to high world `z`
- HR raw row `H - 1` corresponds to low world `z`
- LR raw row `0` corresponds to low world `z`
- LR raw row `H - 1` corresponds to high world `z`

So the two raw images can appear vertically inverted with respect to one another
even when both are geometrically correct.


## LR Crop Export Convention

For the current LR mask-generation workflow, there is an intentional difference
between the saved bbox frame and the saved crop/GeoJSON frame.

The workflow is:

1. compute the bbox in the original full-image LR coordinate system
2. use that bbox directly to crop the raw LR image array
3. keep the saved bbox JSON unchanged in that original LR full-image frame
4. flip the cropped LR image vertically for export and viewing
5. shift the contour coordinates into crop-local coordinates and flip them in the same way before GeoJSON export

So, for LR outputs:

- bbox JSON is unflipped and remains in full-image LR coordinates
- exported crop PNG is flipped vertically
- exported GeoJSON is crop-local and flipped to match the exported crop PNG

This difference is intentional. The bbox is meant for re-opening the raw LR
image, while the exported crop and contours are meant for direct visual use.


## Important Note About Plotting

`matplotlib.pyplot.imshow()` does not use the affine.

It only displays array rows and columns. Therefore, a plot may look correct or
flipped simply because of raw raster orientation, not because the affine has been
applied.

Any visual comparison between HR and LR should keep this in mind.


## Canonical HR -> LR Point Mapping

Assume the input point is expressed in full-image HR pixel coordinates as:

```python
(x_hr, z_hr)
```

Then the canonical mapping is:

```python
hr_voxel = np.array([x_hr, 0, z_hr], dtype=float)
world = apply_affine(hr_affine, hr_voxel)
lr_voxel = apply_affine(np.linalg.inv(lr_affine), world)
```

The LR 2D point in raw array coordinates is then:

```python
z_lr = lr_voxel[0]
x_lr = lr_voxel[2]
```

or equivalently:

```python
lr_point_zx = np.array([lr_voxel[0], lr_voxel[2]])
```


## Canonical HR Array -> LR Array Point Mapping

If the starting point comes from the raw HR image array, it is first known as:

```python
(z_hr, x_hr)
```

To map it correctly:

```python
hr_voxel = np.array([x_hr, 0, z_hr], dtype=float)
world = apply_affine(hr_affine, hr_voxel)
lr_voxel = apply_affine(np.linalg.inv(lr_affine), world)
lr_point_zx = np.array([lr_voxel[0], lr_voxel[2]])
```

This is the simplest correct rule when both source and destination are raw image
arrays.


## If the HR Image Was Flipped for Display

If the HR image was vertically flipped before selecting a point, only the `z`
coordinate must be undone before the affine step.

Let `H_hr` be the HR image height.

If a point is selected on the flipped display as:

```python
(z_display, x_display)
```

then the raw HR array coordinate is:

```python
z_raw = H_hr - 1 - z_display
x_raw = x_display
```

After that, use the standard rule:

```python
hr_voxel = np.array([x_raw, 0, z_raw], dtype=float)
```

Only `z` is unflipped. `x` is unchanged.


## What Not To Do

The following patterns are not good general rules:

- flipping both HR axes before applying the affine
- feeding `(z, y, x)` directly into the HR affine
- assuming the LR inverse affine returns `(x, y, z)`
- mixing array coordinates `(z, x)` with geometric coordinates `(x, z)` without an explicit reorder

If a complex transform seems necessary, it is usually a sign that array order and
affine voxel order have been mixed together.


## About the LR `y` Coordinate

When mapping a point from HR world space into LR voxel space, the returned LR
voxel may contain a `y` value that does not look intuitive for single-slice work.

For matched 2D slice mapping, this is usually not the quantity of interest.
The useful outputs are the LR `z` and `x` indices on the known slice.

Therefore, for 2D HR/LR correspondence:

- use the affine to transform through world space
- keep the returned LR `z` and `x`
- do not over-interpret the LR `y` value unless true 3D reasoning is needed

There is also a separate workflow-specific point for the current LR coronal
stack used by `generate_masks_20um_from_volumes.py`.

In that dataset, the LR affine does not carry the usable coronal slice position
in its translation term, so the slice `y_world` is reconstructed from the image
identifier instead:

```python
y_world = -70.02 + int(image_id) * 0.02
```

This is a dataset convention used for those LR mask-generation inputs, not a
general property of LR affine usage.


## Recommended Mental Model

Use this rule consistently:

- raw image arrays: `(z, x)`
- full-image geometric pixel coordinates: `(x, z)`
- affine/world interface: reorder explicitly when entering or leaving the affine

In practice:

1. If a point comes from image indexing or clicking on a displayed image, think in `(z, x)`.
2. If a point is stored as a bbox corner, contour vertex, or full-image location, think in `(x, z)`.
3. Before applying the HR affine, always build `(x, y, z)` explicitly.
4. After applying the inverse LR affine, read the 2D LR point as `(z, x)`.
5. If you are looking at exported LR crops, remember that the bbox stays in the unflipped full-image frame while the exported crop and GeoJSON are flipped.


## Recommended Helper Contract

A stable helper for 2D point mapping should declare its conventions explicitly.

Example contract:

```python
Input:  full-image HR pixel coordinates (x, z)
Output: raw LR array coordinates (z, x)
```

or, for direct image-array use:

```python
Input:  raw HR array coordinates (z, x)
Output: raw LR array coordinates (z, x)
```

The most important thing is not the exact contract, but that it is written down
clearly and used consistently.


## BBoxes and Rounding

For point mapping, keep coordinates as floats as long as possible.

For bounding boxes:

- use `floor` for minima
- use `ceil` for maxima

Avoid forcing integer conversion too early inside the core mapping step, since
that can introduce off-by-one crop errors.


## Short Debugging Checklist

If a mapped point or crop looks wrong, check these questions in order:

1. Is the source point in local crop coordinates or full-image coordinates?
2. Is the source point currently `(z, x)` or `(x, z)`?
3. Was the HR image flipped only for display?
4. Was the HR affine fed with `(x, y, z)`?
5. Was the LR result interpreted as `(z, y, x)` before plotting or cropping?
6. Am I comparing an unflipped LR bbox with a flipped LR exported crop or GeoJSON?
7. Is the apparent error a real mapping error, or only a vertical raster-orientation difference in display?


## Final Rule Of Thumb

If the input comes from an HR image array, reorder once:

```python
(z, x) -> (x, 0, z)
```

Apply the HR affine, then the inverse LR affine, and read the LR result as:

```python
(z, x) = (lr_voxel[0], lr_voxel[2])
```

That is the simplest correct baseline for 2D HR/LR point mapping in the current
data setup.
