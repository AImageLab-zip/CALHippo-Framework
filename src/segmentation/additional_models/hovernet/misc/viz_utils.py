"""Minimal visualization utilities for HoverNet inference."""

import colorsys
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .utils import get_bounding_box


def colorize(ch, vmin, vmax):
    """Colorize a single channel with jet colormap. Clamps values outside range."""
    cmap = plt.get_cmap("jet")
    ch = np.squeeze(ch.astype("float32"))
    vmin = vmin if vmin is not None else ch.min()
    vmax = vmax if vmax is not None else ch.max()
    ch[ch > vmax] = vmax
    ch[ch < vmin] = vmin
    ch = (ch - vmin) / (vmax - vmin + 1.0e-16)
    ch_cmap = (cmap(ch)[..., :3] * 255).astype("uint8")
    return ch_cmap


def random_colors(N, bright=True):
    """Generate random colors in HSV space then convert to RGB."""
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def visualize_instances_map(
    input_image, inst_map, type_map=None, type_colour=None, line_thickness=2
):
    """Overlays segmentation results on image as contours."""
    overlay = np.copy((input_image).astype(np.uint8))

    inst_list = list(np.unique(inst_map))
    inst_list.remove(0)

    inst_rng_colors = random_colors(len(inst_list))
    inst_rng_colors = np.array(inst_rng_colors) * 255
    inst_rng_colors = inst_rng_colors.astype(np.uint8)

    for inst_idx, inst_id in enumerate(inst_list):
        inst_map_mask = np.array(inst_map == inst_id, np.uint8)
        y1, y2, x1, x2 = get_bounding_box(inst_map_mask)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= inst_map.shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= inst_map.shape[0] - 1 else y2
        inst_map_crop = inst_map_mask[y1:y2, x1:x2]
        contours_crop = cv2.findContours(
            inst_map_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        contours_crop = np.squeeze(contours_crop[0][0].astype("int32"))
        contours_crop += np.asarray([[x1, y1]])
        if type_map is not None:
            type_map_crop = type_map[y1:y2, x1:x2]
            type_id = np.unique(type_map_crop).max()
            inst_colour = type_colour[type_id]
        else:
            inst_colour = (inst_rng_colors[inst_idx]).tolist()
        cv2.drawContours(overlay, [contours_crop], -1, inst_colour, line_thickness)
    return overlay


def visualize_instances_dict(
    input_image, inst_dict, draw_dot=False, type_colour=None, line_thickness=2
):
    """Overlays segmentation results (dictionary) on image as contours."""
    overlay = np.copy((input_image))

    inst_rng_colors = random_colors(len(inst_dict))
    inst_rng_colors = np.array(inst_rng_colors) * 255
    inst_rng_colors = inst_rng_colors.astype(np.uint8)

    for idx, [inst_id, inst_info] in enumerate(inst_dict.items()):
        inst_contour = inst_info["contour"]
        if "type" in inst_info and type_colour is not None:
            inst_colour = type_colour[inst_info["type"]][1]
        else:
            inst_colour = (inst_rng_colors[idx]).tolist()
        cv2.drawContours(overlay, [inst_contour], -1, inst_colour, line_thickness)

        if draw_dot:
            inst_centroid = inst_info["centroid"]
            inst_centroid = tuple([int(v) for v in inst_centroid])
            overlay = cv2.circle(overlay, inst_centroid, 3, (255, 0, 0), -1)
    return overlay
