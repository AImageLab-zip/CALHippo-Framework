from functools import partial

import cv2
import numpy as np
from shapely.geometry import Polygon
from skimage.filters import threshold_sauvola
from skimage.measure import regionprops
from skimage.util import img_as_float

from src.segmentation.inference.merging_functions import binary_outlines_list
from src.segmentation.utils.detection import Detection
from src.utils.helpers import polygon_to_mask, round_polygon_coords, validate_polygon

"""
Adaptive Threshold Model for detecting regions in images using adaptive thresholding.
Used for astrocyte detection.

Unlike the other models, this one does not return a mask with identified regions,
but rather a list of Detection objects representing the detected regions.

The output is adapted to be compatible with the rest of the pipeline.
mask (np.ndarray): fake mask, with only one pixel: 0 if no detections, 255 if detections
metadata (list[Detection]): list of Detection objects representing detected regions.
"""


class AdaptiveThresholdModel:
    def __init__(
        self,
        method: str = "cv2",
        window_size: int = 27,
        second_param: int = 5,
        min_area: int = 10,
        max_area: int = 100,
        erosion_size: int = 1,
        max_eccentricity: float = 0.9,
        final_min_area: int = 5,
        background_threshold: float = 1.8,
        max_mean_color: int = 100,
    ):
        self.config_name = f"ATM_{method}_{window_size}_{second_param}"

        # Define the adaptive thresholding function based on the selected method
        adaptive_threshold_func = None
        if method == "cv2":
            adaptive_threshold_func = partial(
                self.adaptive_threshold_cv2, block_size=window_size, C=second_param
            )
        elif method == "sauvola":
            adaptive_threshold_func = partial(
                self.adaptive_threshold_sauvola, window_size=window_size, k=second_param
            )
        else:
            raise ValueError(f"Unknown adaptive threshold method: {method}")

        # Build the processing pipeline
        self.pipeline = [
            adaptive_threshold_func,
            partial(self.extract_polygons_from_mask),
            partial(self.filter_polygons_by_area, min_area=min_area, max_area=max_area),
            partial(
                self.filter_polygons_by_eccentricity, max_eccentricity=max_eccentricity
            ),
            partial(
                self.erode_fill, erosion_size=erosion_size, min_area=final_min_area
            ),
            partial(
                self.color_filtering,
                max_mean_cell_color=max_mean_color,
                backgroud_threshold=background_threshold,
            ),
            partial(self.create_detection_list),
        ]

    def eval(self, img: np.ndarray) -> tuple[np.ndarray, list[Detection]]:
        print(f"Running AdaptiveThresholdModel {self.config_name}...")

        # Store original image for steps that need it (like color_filtering)
        self.image = img

        # Convert to grayscale if needed
        if len(img.shape) == 3:
            data = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            data = img.astype(np.uint8)

        # Run the pipeline
        for step in self.pipeline:
            data = step(data)
        detection_list = data

        # Create fake mask for compatibility with other models
        mask = np.array([[255]]) if len(detection_list) > 0 else np.array([[0]])

        return mask, detection_list

    # Step functions for the pipeline

    def adaptive_threshold_cv2(
        self, image: np.ndarray, block_size=27, C=5
    ) -> np.ndarray:
        # Apply adaptive thresholding to create a binary mask

        mask = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=block_size,
            C=C,
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.erode(mask, kernel, iterations=1)

        return mask

    def adaptive_threshold_sauvola(self, image, window_size=25, k=0.2) -> np.ndarray:
        # Apply Sauvola adaptive thresholding to create a binary mask

        gray_float = img_as_float(image)
        thresh_sauvola = threshold_sauvola(gray_float, window_size=window_size, k=k)
        binary_sauvola = gray_float < thresh_sauvola
        mask = (binary_sauvola * 255).astype(np.uint8)

        return mask

    def extract_polygons_from_mask(self, mask: np.ndarray) -> list[Polygon]:
        # Extract polygons from a binary mask (black background, white foreground)

        contours = binary_outlines_list(mask)

        polygons = []
        for contour in contours:
            try:
                poly = Polygon(contour)
            except ValueError:
                continue

            polygons.extend(validate_polygon(poly))

        return polygons

    def filter_polygons_by_area(
        self, polygons: list[Polygon], min_area=10, max_area=100
    ) -> list[Polygon]:
        # Filter polygons based on area

        filtered_polygons = []
        for poly in polygons:
            area = poly.area
            if min_area <= area <= max_area:
                filtered_polygons.append(poly)
        return filtered_polygons

    def filter_polygons_by_eccentricity(
        self, polygons: list[Polygon], max_eccentricity=0.9
    ) -> list[Polygon]:
        # Filter polygons based on eccentricity

        filtered_polygons = []
        for poly in polygons:
            poly_mask = polygon_to_mask(poly)
            props = regionprops(poly_mask.astype(int))
            if len(props) > 0:
                eccentricity = props[0].eccentricity
                if eccentricity <= max_eccentricity:
                    filtered_polygons.append(poly)
        return filtered_polygons

    def erode_fill(
        self, polygons: list[Polygon], erosion_size: int = 1, min_area: int = 5
    ) -> list[Polygon]:
        # Erode and fill polygons to smooth boundaries

        tollerance = 0.2

        # TODO: this can be parallelized

        adapted_polygons = []
        for poly in polygons:
            shrinked_poly = poly.buffer(-erosion_size)
            final_poly = shrinked_poly.buffer(erosion_size)

            # split multipolygons
            if final_poly.geom_type == "MultiPolygon":
                for sub_poly in final_poly.geoms:
                    adapted_polygons.append(sub_poly)
            else:
                adapted_polygons.append(final_poly)

        # Round coordinates to integers
        adapted_polygons = [
            round_polygon_coords(poly, tollerance=tollerance)
            for poly in adapted_polygons
        ]

        adapted_polygons = [
            poly
            for poly in adapted_polygons
            if poly is not None and poly.area > min_area
        ]
        return adapted_polygons

    def color_filtering(
        self,
        polygons: list[Polygon],
        max_mean_cell_color: int = 100,
        backgroud_threshold: float = 1.8,
    ) -> list[Polygon]:
        # Filter polygons based on cell color and background contrast
        # polygons with median cell color above max_mean_cell_color are removed
        # polygons with background to cell color ratio below backgroud_threshold are removed

        image = self.image
        y_img, x_img = image.shape[:2]

        epsilon = 1e-6
        dilation_size = 10

        filtered_poly = []
        for poly in polygons:
            # TODO: this can be parallelized

            # Crop image around polygon
            min_x, min_y, max_x, max_y = map(int, poly.bounds)
            min_x, max_x = (
                max(0, min_x - dilation_size),
                min(x_img, max_x + dilation_size),
            )
            min_y, max_y = (
                max(0, min_y - dilation_size),
                min(y_img, max_y + dilation_size),
            )

            h_box, w_box = max_y - min_y, max_x - min_x
            if h_box <= 0 or w_box <= 0:
                continue

            crop_image = image[min_y:max_y, min_x:max_x]

            # Create mask for the polygon and extract cell pixels
            cell_mask_padded = np.zeros((h_box, w_box), dtype=np.uint8)
            local_outline = (
                np.array(poly.exterior.coords) - np.array([min_x, min_y])
            ).astype(np.int32)

            cell_mask_padded = cv2.fillPoly(cell_mask_padded, [local_outline], 1)

            cell_pixels = crop_image[cell_mask_padded == 1]

            if len(cell_pixels) == 0:
                continue

            # Check median cell color
            median_cell_color = np.median(cell_pixels, axis=0).max()
            if median_cell_color > max_mean_cell_color:
                continue

            # Create a dilated mask to get the background
            dilated_mask = cv2.dilate(
                cell_mask_padded,
                np.ones((dilation_size, dilation_size), np.uint8),
                iterations=1,
            )
            background_mask = dilated_mask.astype(bool) & (
                ~cell_mask_padded.astype(bool)
            )

            background_pixels = crop_image[background_mask == 1]

            if len(background_pixels) == 0:
                continue

            # Check background to cell color ratio
            bg_percentile_color = np.percentile(background_pixels, 70, axis=0)

            ratio = bg_percentile_color / (median_cell_color + epsilon)
            if not np.all(ratio > backgroud_threshold):
                continue

            filtered_poly.append(poly)

        return filtered_poly

    def create_detection_list(self, polygons: list[Polygon]) -> list[Detection]:
        # Create a list of Detection objects from polygons

        detection_list = []
        for poly in polygons:
            detection = Detection(
                model_name=self.config_name,
                outline=np.array(poly.exterior.coords),
                polygon=poly,
                probability=1.0,
            )
            detection_list.append(detection)
        return detection_list
