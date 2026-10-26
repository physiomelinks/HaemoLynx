"""Crops of a reference dataset that keep its large-vessel stumps.

``speed.10a`` and ``memory.10a`` time a dataset cut to an eighth, a quarter
and a half. A centred crop of E14.5 holds none of the places its
large-vessel masks leave the image, so ``mask_stump`` finds no inlet and
the run stops at Boundaries. These crops keep every stump the whole stack
has, on its own face (:func:`.full.stump_keeping_box`), so a crop's
boundaries are found the way the whole stack's are.

    python -m release_review.crop_boxes --config config.yaml --of e14p5

(from ``scripts/``, with ``PYTHONPATH`` holding the ``src`` to find the
stumps with) prints the datasets file's crop entries. The stumps are the
pipeline's own (``graph.large_vessel_network``), from the masks loaded as
``build_network`` loads them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .full import box_voxels, stump_keeping_box

FRACTIONS = {"eighth": 0.125, "quarter": 0.25, "half": 0.5}


def find_stumps(config: Path) -> dict[str, Any]:
    """The run's image shape, and each large-vessel mask's stumps with
    their face and footprint extent in voxels."""
    import tifffile

    from haemolynx import io
    from haemolynx.graph.large_vessel_network import _mask_stumps
    from haemolynx.parsers.config import load_config
    from haemolynx.pipeline import stages
    from haemolynx.pipeline.schema import SCHEMA
    from haemolynx.pipeline.settings import resolve_settings

    settings = resolve_settings(load_config(config, SCHEMA), schema=SCHEMA, config_path=config)
    with tifffile.TiffFile(settings["input_path"]) as image:
        shape = tuple(int(side) for side in image.series[0].shape[-3:])
    mask_settings = {
        **SCHEMA.section_values(settings, "Vessel masks"),
        **SCHEMA.section_values(settings, "Input and segmentation"),
        "use_memmap_loading": settings["use_memmap_loading"],
        "memmap_directory": settings["memmap_directory"],
    }
    voxel_xyz = tuple(settings["voxel_size_override_xyz"])
    arteriole, venule, *_ = io.load_and_validate_vessel_masks(
        **stages._vessel_mask_arguments(mask_settings, "large"),
        image_shape=shape,
        main_voxel_size_xyz=voxel_xyz,
    )
    voxel_zyx = io.voxel_size_zyx_from_xyz(voxel_xyz)
    fraction = float(settings["large_vessel_stump_min_component_fraction"])
    stumps = []
    for name, mask in (("arteriole", arteriole), ("venule", venule)):
        if mask is None:
            continue
        for stump in _mask_stumps(mask, voxel_size_zyx=voxel_zyx, min_component_fraction=fraction)[1]:
            voxels = stump.footprint_um / voxel_zyx
            stumps.append({
                "mask": name,
                "axis": "zyx"[stump.axis],
                "face_index": stump.face_index,
                "min_zyx": [int(v) for v in voxels.min(axis=0).round()],
                "max_zyx": [int(v) for v in voxels.max(axis=0).round()],
            })
    return {"shape_zyx": list(shape), "stumps": stumps}


def entries(of: str, shape_zyx: Sequence[int], stumps: Sequence[dict[str, Any]]) -> str:
    """The datasets file's crop entries, as YAML lines."""
    total = int(shape_zyx[0]) * int(shape_zyx[1]) * int(shape_zyx[2])
    lines = []
    for name, fraction in FRACTIONS.items():
        box = stump_keeping_box(shape_zyx, fraction, stumps)
        lines.append(
            f"{of}_{name}: {{of: {of}, crop_fraction: {fraction}, crop_box_zyx: {json.dumps(box)}}}"
            f"  # {box_voxels(box) / total:.3f} of the volume"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Crops that keep a dataset's large-vessel stumps.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--of", default="e14p5", help="the dataset's name in the datasets file")
    arguments = parser.parse_args(argv)
    found = find_stumps(arguments.config)
    for stump in found["stumps"]:
        print(f"# {stump['mask']} stump on the {stump['axis']} = {stump['face_index']} face, "
              f"z {stump['min_zyx'][0]}-{stump['max_zyx'][0]}, y {stump['min_zyx'][1]}-{stump['max_zyx'][1]}, "
              f"x {stump['min_zyx'][2]}-{stump['max_zyx'][2]}")
    print(entries(arguments.of, found["shape_zyx"], found["stumps"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
