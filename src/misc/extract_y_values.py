import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract HR affine y-values.")
    parser.add_argument(
        "--hr-folder-path", type=Path, default=Path("data/raw/high_res")
    )
    parser.add_argument(
        "--output-file", type=Path, default=Path("data/output/misc/y_values.xlsx")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = []

    for filepath in args.hr_folder_path.glob("*_affine.json"):
        # Extract image_id from filename, e.g. B20_3196_affine.json -> 3196.
        image_id = filepath.name.split("_")[1]

        with filepath.open("r") as file:
            affine = json.load(file)
            y_value = float(affine[1][3])

        data.append({"image_id": image_id, "y_value": y_value})

    df = pd.DataFrame(data)
    df = df.sort_values("image_id").reset_index(drop=True)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(args.output_file, index=False)
    print(f"Saved {len(df)} entries to {args.output_file}")
    print(df.head())


if __name__ == "__main__":
    main()
