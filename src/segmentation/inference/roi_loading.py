import json
from pathlib import Path
from typing import List

from shapely.geometry import Polygon, shape


def load_rois_from_geojson(mask_json_path: Path) -> List[Polygon]:
    """Load ROIs from a GeoJSON file and return a list of shapely Polygons."""

    with mask_json_path.open("r") as f:
        roi_data = json.load(f)

    roi_polygons = []
    for feat in roi_data.get("features", []):
        geom = shape(feat["geometry"])
        roi_polygons.append(geom)

    return roi_polygons
