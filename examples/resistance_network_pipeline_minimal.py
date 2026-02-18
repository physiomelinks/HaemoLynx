#!/usr/bin/env python3
"""Minimal ImageLynx example using run_pipeline()."""
import sys
from pathlib import Path

# Make local src/ importable when running from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ImageLynx import run_pipeline

# Beginner-friendly settings: edit these values and run the script.
INPUT_PATH = "path/to/your_image.tif"
INPUT_FORMAT = "tif"  # "tif" or "h5"
H5_DATASET_NAME = None  # e.g. "data" when INPUT_FORMAT = "h5"
STARTING_NODES = [426, 184, 509]


def main() -> None:
    image, skeleton, graph_obj, stats = run_pipeline(
        INPUT_PATH,
        input_format=INPUT_FORMAT,
        dataset_name=H5_DATASET_NAME,
        starting_nodes=STARTING_NODES,
        debug=True,
    )

    print("Image shape:", image.shape)
    print("Skeleton voxels:", int(skeleton.sum()))
    print("Graph nodes/edges:", graph_obj.number_of_nodes(), graph_obj.number_of_edges())
    print("\n=== Statistics ===")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
