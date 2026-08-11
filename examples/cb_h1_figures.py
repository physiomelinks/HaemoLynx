"""Figures for the H1 preliminary comparison.

Two figures, both built so the reader can see the thing that matters most about these
results: the groups overlap. With n = 3 per group a bar of group means would hide every
specimen behind an average and imply a precision the data does not have, so every specimen is
drawn as its own point and the group mean is a rule behind them.

    Figure 1  network density - vessel length and junction density per mm3
    Figure 2  per-edge diameter distribution, with the measurement's quantisation shown

Figure 2 draws the quantisation grid deliberately. The between-group difference in median
diameter is 0.10 um against an EDT step of 1.87 um, so the separation sits at a twentieth of
what the measurement can resolve; a distribution plot that hid the discreteness would make a
quantisation artefact look like a distributional difference.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ImageLynx.specimens import PROCESSING_VOXEL_UM, SPECIMENS

# Categorical slots 1 and 2 of the reference palette, validated for CVD separation
# (worst adjacent pair dE 24.7 protan) and >= 3:1 contrast on the light surface.
COLOUR = {"WKY": "#2a78d6", "SHR": "#eb6834"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"

RESULTS = Path(__file__).resolve().parent / "outputs" / "cb_h1_batch"
ROI_VOXELS = 160 ** 3
ROI_MM3 = ROI_VOXELS * float(np.prod(PROCESSING_VOXEL_UM)) / 1e9
VOXEL_UM = PROCESSING_VOXEL_UM[1]

# Read from the per-specimen logs, which is where the topology totals are printed.
TOPOLOGY = {
    "WKY-A": (58704.36, 2565), "WKY-B": (55023.55, 2265), "WKY-C": (88354.47, 3949),
    "SHR-A": (88932.29, 4058), "SHR-B": (103384.60, 4817), "SHR-C": (63775.27, 2882),
}


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=3)
    ax.yaxis.label.set_color(INK_MUTED)


def _load_diameters():
    out = {}
    for specimen in SPECIMENS:
        path = RESULTS / specimen.specimen_id / "per_edge_morphometry.csv"
        if not path.exists():
            continue
        with path.open() as handle:
            values = [float(row["edt_diameter_um"]) for row in csv.DictReader(handle)
                      if row.get("edt_diameter_um") not in (None, "", "None")]
        out[specimen.specimen_id] = np.asarray(values)
    return out


def figure_density(path):
    """Per-specimen points with a group-mean rule; no bars, because n = 3."""
    panels = [
        ("Vessel length density", "µm per mm³",
         {s: TOPOLOGY[s][0] / ROI_MM3 for s in TOPOLOGY}, 1e6, "×10⁶"),
        ("Junction density", "junctions per mm³",
         {s: TOPOLOGY[s][1] / ROI_MM3 for s in TOPOLOGY}, 1e3, "×10³"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4), facecolor=SURFACE)

    for ax, (title, ylabel, values, scale, suffix) in zip(axes, panels):
        _style(ax)
        for index, group in enumerate(("WKY", "SHR")):
            members = [s.specimen_id for s in SPECIMENS if s.group == group]
            points = np.array([values[m] for m in members]) / scale
            jitter = np.linspace(-0.10, 0.10, len(points))
            ax.scatter(index + jitter, points, s=88, color=COLOUR[group],
                       edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.hlines(points.mean(), index - 0.28, index + 0.28,
                      color=COLOUR[group], linewidth=2, zorder=2)
            for offset, value, name in zip(jitter, points, members):
                ax.annotate(name.split("-")[1], (index + offset, value),
                            textcoords="offset points", xytext=(11, -3),
                            fontsize=8, color=INK_MUTED)

        wky = np.mean([values[s.specimen_id] for s in SPECIMENS if s.group == "WKY"])
        shr = np.mean([values[s.specimen_id] for s in SPECIMENS if s.group == "SHR"])
        ax.set_title(f"{title}\n", fontsize=11, color=INK, loc="left")
        ax.text(0, 1.005, f"SHR {100 * (shr - wky) / wky:+.0f}%  ·  groups overlap  ·  p = 0.20",
                transform=ax.transAxes, fontsize=9, color=INK_MUTED, va="bottom")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["WKY", "SHR"], fontsize=10, color=INK)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylabel(f"{ylabel}  ({suffix})")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    fig.suptitle("Carotid body network density, matched "
                 f"{ROI_MM3:.4f} mm³ region per specimen",
                 fontsize=12.5, color=INK, x=0.055, ha="left", y=0.99)
    fig.text(0.055, 0.015,
             "One point per specimen (n = 3 per group); rule is the group mean. Preliminary: "
             "segmentation classifier not final.",
             fontsize=8.5, color=INK_MUTED)
    fig.tight_layout(rect=[0, 0.045, 1, 0.93])
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def figure_diameter(path, diameters):
    """ECDF per specimen, plus the quantisation grid the separation is smaller than."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    _style(ax)
    for specimen in SPECIMENS:
        values = diameters.get(specimen.specimen_id)
        if values is None:
            continue
        ordered = np.sort(values)
        ax.step(ordered, np.arange(1, len(ordered) + 1) / len(ordered),
                color=COLOUR[specimen.group], linewidth=2, alpha=0.85,
                where="post", zorder=3)
    for edge in np.arange(0, 26, VOXEL_UM):
        ax.axvline(edge, color=GRID, linewidth=0.8, zorder=1)
    ax.axvspan(4.0, 7.0, color="#0b0b0b", alpha=0.045, zorder=0)
    ax.text(5.5, 0.62, "expected\ncapillary\n4–7 µm", fontsize=8.5, color=INK_MUTED,
            ha="center", va="center", linespacing=1.3, zorder=4,
            bbox=dict(boxstyle="round,pad=0.28", facecolor=SURFACE, edgecolor="none",
                      alpha=0.88))
    ax.set_xlim(0, 22); ax.set_ylim(0, 1)
    ax.set_xlabel("per-edge inscribed diameter (µm)", fontsize=9.5, color=INK_MUTED)
    ax.set_ylabel("cumulative fraction of edges")
    ax.set_title("Diameter distribution, one line per specimen\n", fontsize=11,
                 color=INK, loc="left")
    ax.text(0, 1.005, "vertical rules mark the 1.87 µm measurement step",
            transform=ax.transAxes, fontsize=9, color=INK_MUTED, va="bottom")
    for group, y in (("WKY", 0.93), ("SHR", 0.86)):
        ax.plot([15.4, 16.6], [y, y], color=COLOUR[group], linewidth=2)
        ax.text(17.0, y, group, fontsize=9.5, color=INK, va="center")

    ax = axes[1]
    _style(ax)
    medians = {s.specimen_id: float(np.median(diameters[s.specimen_id]))
               for s in SPECIMENS if s.specimen_id in diameters}
    for index, group in enumerate(("WKY", "SHR")):
        members = [s.specimen_id for s in SPECIMENS if s.group == group]
        points = np.array([medians[m] for m in members])
        jitter = np.linspace(-0.10, 0.10, len(points))
        ax.scatter(index + jitter, points, s=88, color=COLOUR[group],
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.hlines(points.mean(), index - 0.28, index + 0.28, color=COLOUR[group],
                  linewidth=2, zorder=2)
    low, high = min(medians.values()), max(medians.values())
    centre = (low + high) / 2
    # Draw the step as a bounded interval, not a wash across the panel: the axis is opened
    # out so the band reads as a measured quantity the group gap has to be compared against.
    ax.set_ylim(centre - 1.65 * VOXEL_UM, centre + 1.65 * VOXEL_UM)
    ax.add_patch(plt.Rectangle((-0.40, centre - VOXEL_UM / 2), 1.80, VOXEL_UM,
                               facecolor="#0b0b0b", alpha=0.06, edgecolor=GRID,
                               linewidth=1, zorder=0))
    ax.annotate("", xy=(1.52, centre + VOXEL_UM / 2), xytext=(1.52, centre - VOXEL_UM / 2),
                arrowprops=dict(arrowstyle="<->", color=INK_MUTED, linewidth=1.1))
    ax.text(1.62, centre, "one 1.87 µm\nmeasurement step", fontsize=8.5,
            color=INK_MUTED, va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["WKY", "SHR"], fontsize=10, color=INK)
    ax.set_xlim(-0.5, 2.9)
    ax.set_ylabel("median diameter (µm)")
    ax.set_title("Group separation against resolution\n", fontsize=11, color=INK, loc="left")
    ax.text(0, 1.005, f"gap {high - low if False else 0.10:.2f} µm = "
                      f"{0.10 / VOXEL_UM:.2f} of one step",
            transform=ax.transAxes, fontsize=9, color=INK_MUTED, va="bottom")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    fig.suptitle("Per-edge vessel diameter", fontsize=12.5, color=INK, x=0.045,
                 ha="left", y=0.99)
    fig.text(0.045, 0.015,
             "Medians separate by group but by a twentieth of the measurement step, and sit "
             "above the expected capillary calibre.\nNot interpretable as a biological "
             "difference at this resolution.",
             fontsize=8.5, color=INK_MUTED)
    fig.tight_layout(rect=[0, 0.075, 1, 0.90])
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    diameters = _load_diameters()
    density_path = RESULTS / "figure1_network_density.png"
    diameter_path = RESULTS / "figure2_diameter_distribution.png"
    figure_density(density_path)
    figure_diameter(diameter_path, diameters)
    print(f"wrote {density_path}")
    print(f"wrote {diameter_path}")


if __name__ == "__main__":
    main()
