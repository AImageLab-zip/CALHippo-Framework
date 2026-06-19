import numpy as np

### WORLD TO IMAGE COORDINATE CONVERSION
# Both helpers start from world coordinates stored as (x, z) and append the shared
# slice coordinate y_world to build full world points (x, y, z). The only
# difference is how the inverse affine output must be interpreted:
# HR inverse affine returns voxel coordinates in (x, y, z), while LR inverse
# affine returns voxel coordinates in (z, y, x).


def check_parse_single_point(coords: np.ndarray) -> tuple[np.ndarray, bool]:
    # Check if functions input is a single point and reshape to (1, 2)
    coords = np.asarray(coords, dtype=np.float64)
    single_point = coords.shape == (2,)
    if single_point:
        coords = coords.reshape(1, 2)
    return coords, single_point


def map_world_xz_to_LR_zx(
    world_coords_xz: np.ndarray,
    image_affine: np.ndarray,
    y_world: float = 0,
) -> np.ndarray:
    """
    Convert world coordinates to LR image pixel coordinates using the inverse affine.

    For LR images, the inverse affine returns voxel coordinates in `(z, y, x)`
    order, so the 2D image coordinates are obtained by keeping indices 0 and 2.

    Args:
        world_coords_xz (np.ndarray): World coordinates. Shape `(2,)` or
            `(n_points, 2)` in `(x, z)` order.
        image_affine (np.ndarray): 4x4 image affine matrix.
        y_world (float): The world Y coordinate corresponding to the slice.

    Returns:
        lr_coords_zx (np.ndarray): LR image pixel coordinates. Shape `(2,)`
            or `(n_points, 2)` in `(z, x)` order.
    """

    world_coords_xz, single_point = check_parse_single_point(world_coords_xz)
    image_affine_inv = np.linalg.inv(image_affine)

    # Add the y_world coordinate to get (x, y, z, 1) homogeneous coordinates in world space
    world_coords_xyz = np.column_stack(
        [
            world_coords_xz[:, 0],
            np.full(world_coords_xz.shape[0], y_world, dtype=np.float64),
            world_coords_xz[:, 1],
            np.ones(world_coords_xz.shape[0], dtype=np.float64),
        ]
    )

    # Apply the inverse affine to get LR voxel coordinates in (z, y, x) order.
    lr_coords_zyx = (world_coords_xyz @ image_affine_inv.T)[:, :3]

    # Keep the image indexing axes only: (z, x).
    lr_coords_zx = lr_coords_zyx[:, [0, 2]]  # (z, x)

    if single_point:
        lr_coords_zx = lr_coords_zx[0]

    return lr_coords_zx


def map_LR_zx_to_world_xz(
    lr_coords_zx: np.ndarray,
    image_affine: np.ndarray,
) -> np.ndarray:
    """
    Convert LR image pixel coordinates to world coordinates using the affine.

    For LR images, the affine requires input coordinates in `(z, y, x)` order
    and returns world coordinates in `(x, y, z)` order. 

    Args:
        lr_coords_zx (np.ndarray): LR image pixel coordinates. Shape `(2,)`
            or `(n_points, 2)` in `(z, x)` order.
        image_affine (np.ndarray): 4x4 image affine matrix.

    Returns:
        world_coords_xz (np.ndarray): World coordinates. Shape `(2,)` or
            `(n_points, 2)` in `(x, z)` order.
    """

    lr_coords_zx, single_point = check_parse_single_point(lr_coords_zx)

    # Add a zero y-coordinate to get (z, y, x, 1) homogeneous coordinates in LR space
    lr_coords_zyx = np.column_stack(
        [
            lr_coords_zx[:, 0],
            np.zeros(lr_coords_zx.shape[0], dtype=np.float64),
            lr_coords_zx[:, 1],
            np.ones(lr_coords_zx.shape[0], dtype=np.float64),
        ]
    )

    # Apply the affine to get world coordinates in (x, y, z) order.
    world_coords_xyz = (lr_coords_zyx @ image_affine.T)[:, :3]

    # Keep the world indexing axes only: (x, z).
    world_coords_xz = world_coords_xyz[:, [0, 2]]  # (x, z)

    if single_point:
        world_coords_xz = world_coords_xz[0]

    return world_coords_xz


def map_world_xz_to_HR_zx(
    world_coords_xz: np.ndarray,
    image_affine: np.ndarray,
    y_world: float = 0,
) -> np.ndarray:
    """
    Convert world coordinates to HR image pixel coordinates using the inverse affine.

    For HR images, the inverse affine returns voxel coordinates in `(x, y, z)`
    order, so the 2D image coordinates are obtained by reordering them to
    `(z, x)` for image indexing.

    Args:
        world_coords_xz (np.ndarray): World coordinates. Shape `(2,)` or
            `(n_points, 2)` in `(x, z)` order.
        image_affine (np.ndarray): 4x4 image affine matrix.
        y_world (float): The world Y coordinate corresponding to the slice.

    Returns:
        hr_coords_zx (np.ndarray): HR image pixel coordinates. Shape `(2,)`
            or `(n_points, 2)` in `(z, x)` order.
    """

    world_coords_xz, single_point = check_parse_single_point(world_coords_xz)
    image_affine_inv = np.linalg.inv(image_affine)

    # Add the y_world coordinate to get (x, y, z, 1) homogeneous coordinates in world space
    world_coords_xyz = np.column_stack(
        [
            world_coords_xz[:, 0],
            np.full(world_coords_xz.shape[0], y_world, dtype=np.float64),
            world_coords_xz[:, 1],
            np.ones(world_coords_xz.shape[0], dtype=np.float64),
        ]
    )

    # Apply the inverse affine to get HR voxel coordinates in (x, y, z) order,
    # then reorder them for image indexing.
    hr_coords_xyz = (world_coords_xyz @ image_affine_inv.T)[:, :3]
    hr_coords_zx = hr_coords_xyz[:, [2, 0]]  # Swap to (z, x)

    if single_point:
        hr_coords_zx = hr_coords_zx[0]

    return hr_coords_zx


def image_id_to_world_y(image_id: str, y_start: float, y_step: float) -> float:
    """
    Map the LR image id to the world-space y coordinate of that coronal slice.

    The current LR MINC affines keep a zero translation along y, so this value
    cannot be recovered from the affine itself. The historical formula was the
    correct one for these slices, so we keep it explicit here.
    """

    return y_start + int(image_id) * y_step


## HR -> LR mappings


def map_HR_xz_to_LR_xz(
    hr_coords_xz: np.ndarray,
    hr_affine: np.ndarray,
    lr_affine_inv: np.ndarray,
) -> np.ndarray:

    hr_coords_xz, single_point = check_parse_single_point(hr_coords_xz)

    hr_coords_xyz = np.column_stack(
        [
            hr_coords_xz[:, 0],
            np.zeros(hr_coords_xz.shape[0], dtype=np.float64),
            hr_coords_xz[:, 1],
            np.ones(hr_coords_xz.shape[0], dtype=np.float64),
        ]
    )

    world_coords_xyz = hr_coords_xyz @ hr_affine.T
    lr_coords_zyx = world_coords_xyz @ lr_affine_inv.T  # LR affine swaps axes

    lr_coords_xz = lr_coords_zyx[:, [2, 0]]  # swap back to (x, z) for output

    if single_point:
        lr_coords_xz = lr_coords_xz[0]

    return lr_coords_xz
