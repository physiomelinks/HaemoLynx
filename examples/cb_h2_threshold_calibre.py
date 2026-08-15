"""How far the threshold moves measured calibre, which sets the size of the correlated error.

Assessment finding S12 established that independent calibre error averages down across a network
solve while correlated error does not. The threshold is the dominant correlated term: every edge in
a specimen is measured from one mask produced at one threshold, so moving it moves every diameter
together. This measures by how much, so that the noise floor in S15 rests on the real perturbation
rather than on a round number.

The three runs already exist from the H1 sensitivity sweep, so this reads them rather than
recomputing anything. The clean interval is 0.85 to 0.90: H1 section 6.4 records that at 0.95 four
of the six specimens sit at or beyond their fragmentation onset, which contaminates that column.

Run with::

    venv/bin/python examples/cb_h2_threshold_calibre.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUTS = Path(__file__).resolve().parent / "outputs"
SPECIMENS = ("WKY-A", "WKY-B", "WKY-C", "SHR-A", "SHR-B", "SHR-C")
VOXEL_UM = 1.866

RUNS = (
    ("0.85", lambda s: OUTPUTS / "cb_h1_sensitivity" / "t0.85" / s / "per_edge_morphometry.csv"),
    ("0.90", lambda s: OUTPUTS / "cb_h1_batch" / s / "per_edge_morphometry.csv"),
    ("0.95", lambda s: OUTPUTS / "cb_h1_sensitivity" / "t0.95" / s / "per_edge_morphometry.csv"),
)


def median_calibre(path):
    d = pd.read_csv(path, usecols=["edt_diameter_um"])["edt_diameter_um"].to_numpy(float)
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d))


def main():
    print(f"{'specimen':10}" + "".join(f"{label:>9}" for label, _ in RUNS)
          + f"{'0.85->0.90':>13}{'0.90->0.95':>13}")
    table = []
    for specimen_id in SPECIMENS:
        medians = [median_calibre(path(specimen_id)) for _, path in RUNS]
        table.append(medians)
        print(f"{specimen_id:10}" + "".join(f"{m:9.3f}" for m in medians)
              + f"{medians[1]-medians[0]:13.3f}{medians[2]-medians[1]:13.3f}")

    values = np.array(table)
    clean = np.abs(values[:, 1] - values[:, 0])
    contaminated = np.abs(values[:, 2] - values[:, 1])
    baseline = values[:, 1].mean()

    print(f"\nmedian calibre at the frozen threshold: {baseline:.3f} um")
    print(f"mean shift over the clean 0.85-0.90 interval: {clean.mean():.3f} um "
          f"({clean.mean()/VOXEL_UM:.3f} voxel)")
    print(f"mean shift over the contaminated 0.90-0.95 interval: {contaminated.mean():.3f} um "
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


if __name__ == "__main__":
    main()
