import argparse
from pathlib import Path
from urllib.error import HTTPError

import wget
from tqdm import tqdm

BASE_URL = (
    "https://ftp.bigbrainproject.org/bigbrain-ftp/"
    "BigBrainRelease.2015/2D_Final_Sections/Coronal/Minc/"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download BigBrain LR MINC slices.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/low_res"))
    parser.add_argument("--start-idx", type=int, default=2776)
    parser.add_argument("--last-idx", type=int, default=3998)
    parser.add_argument("--base-url", default=BASE_URL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for image_id in tqdm(range(args.start_idx, args.last_idx + 1)):
        filename = f"pm{image_id}o.mnc"
        url = args.base_url + filename
        outpath = args.output_dir / filename

        if outpath.exists():
            print("Already downloaded:", filename)
            continue

        try:
            print("DOWNLOADING:", filename)
            wget.download(url, str(outpath))
            print("Downloaded:", filename)
        except HTTPError as error:
            print(error)
            continue


if __name__ == "__main__":
    main()
