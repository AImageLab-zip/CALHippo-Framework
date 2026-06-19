import os

import matplotlib.pyplot as plt
import numpy as np


def plot_geojson_annots_on_image(
    image,
    geojson_data,
    save_path=None,
    class_map=None,
    figsize=(12, 12),
    dpi=300,
):
    """
    Visualizes GeoJSON annotations on an image with robust fallback handling.

    Args:
        image: Numpy array of the image (H, W, C) or (H, W).
        geojson_data: Dictionary containing GeoJSON features.
        save_path: (Optional) Path to save the plot. If provided, plot is not shown.
        class_map: Dictionary of allowed class names. If None, all classes are plotted.
        figsize: Tuple for figure size.
        dpi: Resolution for saving.
    """
    # Create figure explicitly to allow closing later
    fig, ax = plt.subplots(figsize=figsize)

    # Handle Grayscale vs RGB display
    if image.ndim == 2:
        ax.imshow(image, cmap="gray")
    else:
        ax.imshow(image)

    ax.axis("off")
    ax.set_title("Image with GeoJSON Annotations")

    features = geojson_data.get("features", [])

    if not features:
        print("   [!] No features found in GeoJSON.")

    for i, feature in enumerate(features):
        properties = feature.get("properties", {})
        classification = properties.get("classification", {})

        # --- Robust Property Extraction ---
        # 1. Get Class Name (fallback to "Unknown")
        cell_class = classification.get("name", "Unknown")

        # 2. Filter by Class Map (if provided)
        if class_map is not None and cell_class not in class_map:
            continue

        # 3. Get Color (Handle missing or malformed colors)
        raw_color = classification.get("color", [255, 0, 0])  # Default Red
        try:
            # Ensure it is a list/array and normalize to 0-1
            color = np.array(raw_color) / 255.0
        except Exception:
            # Fallback if color format is weird
            color = (1.0, 0.0, 0.0)

        # --- Robust Geometry Handling ---
        geometry = feature.get("geometry", {})
        geom_type = geometry.get("type")
        coords = geometry.get("coordinates")

        if not coords:
            print(f"   [!] Skipping feature {i}: No coordinates found.")
            continue

        try:
            # CASE 1: Polygon (Most common for segmentation)
            # Structure: [ [ [x,y], ... ] ] (List of rings, usually we plot the first/exterior ring)
            if geom_type == "Polygon":
                for ring in coords:
                    poly_np = np.array(ring)
                    ax.plot(poly_np[:, 0], poly_np[:, 1], color=color, linewidth=1.5)

            # CASE 2: Point (Fallback for simple annotations)
            # Structure: [x, y]
            elif geom_type == "Point":
                x, y = coords
                # Plot as a scatter dot instead of a line
                ax.scatter(x, y, color=color, s=20, label=cell_class)

            # CASE 3: MultiPolygon (Complex disjoint regions)
            # Structure: [ [ [[x,y]...] ], [ ... ] ]
            elif geom_type == "MultiPolygon":
                for polygon in coords:
                    for ring in polygon:
                        poly_np = np.array(ring)
                        ax.plot(
                            poly_np[:, 0], poly_np[:, 1], color=color, linewidth=1.5
                        )

            # CASE 4: LineString
            elif geom_type == "LineString":
                line_np = np.array(coords)
                ax.plot(line_np[:, 0], line_np[:, 1], color=color, linewidth=1.5)

        except Exception as e:
            print(f"   [!] Error plotting feature {i} ({geom_type}): {e}")
            continue

    # --- Save or Show Logic ---
    if save_path:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            # print(f"Saved to {save_path}") # Optional: Uncomment for verbosity
        except Exception as e:
            print(f"   [!] Failed to save image to {save_path}: {e}")
    else:
        plt.show()

    # Always close the figure to free memory
    plt.close(fig)
