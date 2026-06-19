from pathlib import Path
from typing import Dict, Tuple

import nibabel as nib
import numpy as np
import pyvista as pv


def load_gifti_surface_world(gii_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load verts and faces from a from a GIfTI surface.

    Args:
        gii_path (str): path to the GIfTI file containing the surface.

    Returns:
        verts (np.ndarray): shape (n_verts, 3) in world coordinates (mm).
        faces (np.ndarray): shape (n_faces, 3) of vertex indices (0-based) defining triangular faces.
    """

    gii = nib.load(gii_path)

    # Vertices
    pointset_da = next(
        da
        for da in gii.darrays
        if da.intent == nib.nifti1.intent_codes["NIFTI_INTENT_POINTSET"]
    )

    # Faces
    tri_da = next(
        da
        for da in gii.darrays
        if da.intent == nib.nifti1.intent_codes["NIFTI_INTENT_TRIANGLE"]
    )

    verts = pointset_da.data.astype(np.float64)
    faces = tri_da.data.astype(np.int64)

    return verts, faces


def polydata_from_surface(verts_world: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    """
    Given loaded verts and faces as numpy arrays,
    create a PyVista PolyData object representing the surface.

    Args:
        verts_world (np.ndarray): shape (n_verts, 3) in world coordinates (mm).
        faces (np.ndarray): shape (n_faces, 3) of vertex indices (0-based) defining triangular faces.

    Returns:
        pv.PolyData: polydata object representing the surface
    """

    # Process faces for PyVista:
    # Prepend '3' (vertex count per face) to each triangle, then flatten.
    # PyVista requires format: [3, v0, v1, v2,  3, v3, v4, v5, ...]

    faces_pv = np.hstack(
        [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]
    ).ravel()

    # Create PyVista PolyData and clean to ensure triangle consistency
    surf = pv.PolyData(verts_world, faces_pv)
    surf = surf.triangulate().clean()

    return surf


def load_multiple_surfaces(
    surface_paths: Dict[str, str | Path],
) -> dict[str, pv.PolyData]:
    """Load multiple GIfTI surfaces and keep them in world coordinates."""

    surfaces = {}

    for key, surface_path in surface_paths.items():
        verts, faces = load_gifti_surface_world(str(surface_path))
        surfaces[key] = polydata_from_surface(verts, faces)

    return surfaces


def cut_multiple_surfaces_at_y(
    surfaces: Dict[str, pv.PolyData], y_world: float
) -> Dict[str, list[np.ndarray]]:
    """
    Intersect each surface with the plane y = y_world in world space.
    The surface vertices are already in world coordinates.
    This does not handles holes and just returns the list of raw contours (holes included).

    Args:
        surfaces (Dict[str, pv.PolyData]): dictionary of surfaces to slice, with CA regions as identifiers.
        y_world (float): the y coordinate in world space where to slice the surfaces.

    Returns:
        Dict[str, list[np.ndarray]]: output contours where for each region, we have a list of contours as (n_points_in_contour, 2) arrays in world space.
    """

    output_contours: Dict[str, list[np.ndarray]] = {}

    for key, surf in surfaces.items():
        # Slice the section with a plane (normal) with the origin in the given y_world
        section = surf.slice(normal=(0, 1, 0), origin=(0.0, y_world, 0.0)).clean()
        if section.n_points == 0 or section.lines.size == 0:
            output_contours[key] = []
            continue

        # Connect the different sliced segment into contours
        section = section.strip(join=True).clean()

        # Like the surface, the section is represented by a list of points and a list of lines that connect them.
        # section.points is (n_points, 3) array of the coordinates of the points in world space.
        # section.lines is structured as: [n_points_in_contour1, id_pt1, id_pt2, ..., n_points_in_contour2, id_pt1, id_pt2, ...]

        contours_world = []
        lines = section.lines
        points = section.points
        idx = 0

        while idx < len(lines):
            # Get the points in the current contour
            n_points = int(lines[idx])  # first value is the number of points
            point_ids = lines[
                idx + 1 : idx + 1 + n_points
            ]  # the following are the indexes
            idx += n_points + 1  # move to the next contour

            if n_points < 3:
                continue

            # Get the points coordinates, keeping only x and z
            contour_xz = points[point_ids][:, [0, 2]].astype(np.float64)
            # Ensure that the contour is closed, if not, append the first point at the end
            if not np.allclose(contour_xz[0], contour_xz[-1]):
                contour_xz = np.vstack([contour_xz, contour_xz[0]])

            contours_world.append(contour_xz)

        output_contours[key] = contours_world

    return output_contours
