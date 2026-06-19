import os

import pyvips
from tqdm import tqdm

# Define input and output directories
INPUT_DIR = "/work/grana_urologia/WSI-data"
OUTPUT_DIR = "/work/grana_urologia/WSI-data-tif"
GEOJSON_DIR = "/work/grana_urologia/wsi_geojson_data"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# extract the list of geojson files patient id
annotated_wsi = []
for annotation in os.listdir(GEOJSON_DIR):
    if annotation.endswith(".geojson"):
        # Extract the patient ID from the filename
        patient_id = os.path.splitext(annotation)[0]
        print(f"Found annotation for patient ID: {patient_id}")
        annotated_wsi.append(patient_id)

print(f"Found {len(annotated_wsi)} annotated WSI files.")
print("Annotated WSI patient IDs:")
print(annotated_wsi)

# iterate over the list of wsi images, check if the patient id is in the list of annotated wsi and convert them if so
already_converted = []
converted = []
non_annotated = []
for wsi in tqdm(os.listdir(INPUT_DIR)):
    if wsi.lower().endswith((".jpg", ".jpeg", ".png")):
        # extract the patient id from the filename
        patient_id = os.path.splitext(wsi)[0]
        # print(f"Found WSI for patient ID: {patient_id}")
        # Check if the patient ID is in the list of annotated WSI
        if patient_id in annotated_wsi:
            print(
                f"Found WSI for patient ID: {patient_id} in the list of annotated WSI"
            )

        else:
            print(f"WSI for patient ID: {patient_id} not in the list of annotated WSI")
            non_annotated.append(patient_id)
            continue
        output_filename = patient_id + ".tiff"
        input_filepath = os.path.join(INPUT_DIR, wsi)
        output_filepath = os.path.join(OUTPUT_DIR, output_filename)
        # check if the patient id is already converted

        if os.path.exists(output_filepath):
            print(f"WSI for patient ID: {patient_id} already converted to tiff")
            already_converted.append(patient_id)
            continue
        else:
            # else, convert the WSI to tiff
            print(f"Converting WSI for patient ID: {patient_id} to tiff")
            # Load the image

            image = pyvips.Image.new_from_file(input_filepath, access="sequential")
            # Handle transparency if the image has an alpha channel
            if image.hasalpha():
                # Replace transparency with white background
                image = image.flatten(background=255)

            # Save as pyramidal TIFF
            image.tiffsave(
                output_filepath,
                tile=True,
                tile_width=256,
                tile_height=256,
                pyramid=True,
                compression="lzw",
                bigtiff=True,
                # Add additional options if needed
            )

            converted.append(patient_id)

print("Done converting WSI to tiff for the following patient IDs:")
print(converted)
print("Already converted WSI to tiff for the following patient IDs:")
print(already_converted)
print("Non-annotated WSI for the following patient IDs:")
print(non_annotated)
