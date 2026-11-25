#!/usr/bin/env python3
"""Run maskel on one mask and write what it measured as JSON.

Standalone: maskel needs Python 3.13 or newer, so this runs in an environment
of its own (``pip install maskel``), never inside HaemoLynx's, and imports
nothing from this repository. ``maskel_side.run`` calls it::

    <maskel python> -I maskel_runner.py MASK.npy OUT.json --spacing-zyx 1.0 0.98 0.98 [--cleanup]

``--cleanup`` turns on maskel's own optional clean-ups (triangle-junction
collapse and short-spur pruning, at maskel's default thresholds); without it
every extraction setting is maskel's default, bar the per-branch records and
the mask radius the comparison needs.
"""
import argparse
import json
import math

import numpy as np


def _plain(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mask")
    parser.add_argument("out")
    parser.add_argument("--spacing-zyx", nargs=3, type=float, required=True)
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()

    import maskel
    from maskel.config import ExtractionConfig, OutputConfig, PipelineConfig
    from maskel.pipeline import analyze_segmentation_mask

    extraction = ExtractionConfig(
        branches=True,
        summary=True,
        mask_radius=True,
        junction_cleanup=args.cleanup,
        prune_spurs=args.cleanup,
        spacing=tuple(args.spacing_zyx),
    )
    config = PipelineConfig(extraction=extraction, output=OutputConfig())
    mask = np.load(args.mask).astype(np.uint8)
    result = analyze_segmentation_mask(mask, config)
    columns = ("branch-distance", "euclidean-distance", "mean_diameter", "branch-type", "object_id")
    out = {
        "maskel_version": getattr(maskel, "__version__", None),
        "config": extraction.to_dict(),
        "summary": [{k: _plain(v) for k, v in s.items()} for s in result.summary_features],
        "branches": [{c: _plain(b.get(c)) for c in columns} for b in result.branch_records],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
