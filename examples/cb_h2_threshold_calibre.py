"""How far the threshold moves measured calibre, which sets the size of the correlated error.

Assessment finding S12 established that independent calibre error averages down across a network
solve while correlated error does not. The threshold is the dominant correlated term: every edge in
a specimen is measured from one mask produced at one threshold, so moving it moves every diameter
together. This measures by how much, so that the noise floor in S15 rests on the real perturbation
rather than on a round number.

The three runs are the frozen threshold's batch run and the H1 sensitivity runs at its two grid
neighbours, so this reads them rather than recomputing anything. Since the 2026-09-28
re-selection the frozen value is 0.95 and the neighbours are 0.93 and 0.97. The clean interval is
the lower one, 0.93 to 0.95: 0.97 is the fragmentation onset in four of the six specimens, which
contaminates that column. (Until then it was 0.85 / 0.90 / 0.95, on centre-cropped runs.)

The neighbours come from ``cb_h1_batch.py --stage sensitivity``, which holds the seed at the
frozen ``HYSTERESIS_HIGH`` so only the flood threshold moves (open item 41). Runs made before
that seeded the 0.93 neighbour at 0.98, so its shift mixed two parameters.

Run with::

    venv/bin/python examples/cb_h2_threshold_calibre.py
"""
import numpy as np

from ImageLynx import cb_settings
from ImageLynx.batch_outputs import open_batch_run
from ImageLynx.specimens import get_specimen

from cb_h1_batch import SENSITIVITY_DIR

SPECIMENS = ("WKY-A", "WKY-B", "WKY-C", "SHR-A", "SHR-B", "SHR-C")
VOXEL_UM = 1.866

_GRID = list(cb_settings.THRESHOLD_GRID)
_FROZEN = _GRID.index(cb_settings.FROZEN_THRESHOLD)
LOWER, FROZEN, UPPER = (f"{t:.2f}" for t in _GRID[_FROZEN - 1:_FROZEN + 2])


def _sensitivity(label):
    return lambda specimen_id: SENSITIVITY_DIR / f"t{label}" / specimen_id


# Each run is the folder to open for a specimen; None is the specimen's batch run.
RUNS = (
    (LOWER, _sensitivity(LOWER)),
    (FROZEN, lambda specimen_id: None),
    (UPPER, _sensitivity(UPPER)),
)


def median_calibre(specimen_id, run_dir):
    """Median EDT calibre of one run's edge table; opening it checks the placed ROI."""
    run = open_batch_run(get_specimen(specimen_id), run_dir)
    d = np.array([float(row["edt_diameter_um"]) for row in run.edge_table().values()
                  if row["edt_diameter_um"] != ""], dtype=float)
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d))


def main():
    """Print the per-specimen table and return the mean shift over the clean interval."""
    print(f"{'specimen':10}" + "".join(f"{label:>9}" for label, _ in RUNS)
          + f"{LOWER + '->' + FROZEN:>13}{FROZEN + '->' + UPPER:>13}")
    table = []
    for specimen_id in SPECIMENS:
        medians = [median_calibre(specimen_id, folder(specimen_id)) for _, folder in RUNS]
        table.append(medians)
        print(f"{specimen_id:10}" + "".join(f"{m:9.3f}" for m in medians)
              + f"{medians[1]-medians[0]:13.3f}{medians[2]-medians[1]:13.3f}")

    values = np.array(table)
    clean = np.abs(values[:, 1] - values[:, 0])
    contaminated = np.abs(values[:, 2] - values[:, 1])
    baseline = values[:, 1].mean()

    print(f"\nmedian calibre at the frozen threshold: {baseline:.3f} um")
    print(f"mean shift over the clean {LOWER}-{FROZEN} interval: {clean.mean():.3f} um "
          f"({clean.mean()/VOXEL_UM:.3f} voxel)")
    print(f"mean shift over the contaminated {FROZEN}-{UPPER} interval: {contaminated.mean():.3f} um "
          f"({contaminated.mean()/VOXEL_UM:.3f} voxel)")

    # Calibre falls monotonically with threshold for every specimen. That common direction is what
    # makes the error correlated rather than independent, and so what stops it averaging down.
    monotonic = np.all(np.diff(values, axis=1) < 0, axis=1)
    print(f"specimens where calibre falls monotonically with threshold: "
          f"{monotonic.sum()}/{len(SPECIMENS)}")

    print(f"\nimplied per-edge error over the clean interval: "
          f"dd/d = {100*clean.mean()/baseline:.1f}%, "
          f"dR/R = 4*dd/d = {400*clean.mean()/baseline:.1f}%")
    print("Feed the measured shift into the network solve with:")
    print(f"  venv/bin/python examples/cb_h2_error_propagation.py "
          f"--perturbation-um {clean.mean():.3f}")
    return float(clean.mean())


if __name__ == "__main__":
    main()
