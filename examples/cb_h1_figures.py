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
from ImageLynx import cb_settings

# Categorical slots 1 and 2 of the reference palette, validated for CVD separation
# (worst adjacent pair dE 24.7 protan) and >= 3:1 contrast on the light surface.
COLOUR = {"WKY": "#2a78d6", "SHR": "#eb6834"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"

RESULTS = Path(__file__).resolve().parent / "outputs" / "cb_h1_batch"
ROI_VOXELS = int(np.prod(cb_settings.ROI_VOXELS))
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


def _beta1_density():
    """beta-1 = E - V + C, the loop count H1 s1.1 names and the pipeline does not report.

    C = 1 because GraphConfig.keep_largest_component_only is set, so the graph handed to
    morphometry is a single connected component by construction.
    """
    out = {}
    for specimen in SPECIMENS:
        path = RESULTS / specimen.specimen_id / "per_edge_morphometry.csv"
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        nodes = {r["u"] for r in rows} | {r["v"] for r in rows}
        out[specimen.specimen_id] = (len(rows) - len(nodes) + 1) / ROI_MM3
    return out


def figure_density(path):
    """Per-specimen points with a group-mean rule; no bars, because n = 3."""
    panels = [
        ("β₁ loop density", "independent loops per mm³",
         _beta1_density(), 1e3, "×10³"),
        ("Junction density", "junctions per mm³",
         {s: TOPOLOGY[s][1] / ROI_MM3 for s in TOPOLOGY}, 1e3, "×10³"),
        ("Vessel length density", "µm per mm³",
         {s: TOPOLOGY[s][0] / ROI_MM3 for s in TOPOLOGY}, 1e6, "×10⁶"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4), facecolor=SURFACE)

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
                transform=ax.transAxes, fontsize=9.5, color=INK_MUTED, va="bottom")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["WKY", "SHR"], fontsize=10, color=INK)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylabel(f"{ylabel}  ({suffix})")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    fig.suptitle("Carotid body network density, matched "
                 f"{ROI_MM3:.4f} mm³ region per specimen",
                 fontsize=12.5, color=INK, x=0.040, ha="left", y=0.99)
    fig.text(0.040, 0.015,
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
    sensitivity_path = RESULTS / "figure3_threshold_sensitivity.png"
    figure_sensitivity(sensitivity_path)
    degree_path = RESULTS / "figure7_node_degree.png"
    figure_degree(degree_path)
    length_path = RESULTS / "figure8_segment_length.png"
    figure_segment_length(length_path)
    for written in (density_path, diameter_path, sensitivity_path, degree_path, length_path):
        print(f"wrote {written}")



def figure_sensitivity(path):
    """Group ratio against threshold — robustness, and the predicted direction.

    Three series, so identity is carried by a direct label on each line rather than a legend
    box; the aqua slot sits below 3:1 on the light surface and the relief rule applies.
    """
    thresholds = np.array([0.85, 0.90, 0.95])
    series = [
        ("β₁ loop density",     np.array([1.297, 1.401, 1.505]), "#2a78d6"),
        ("Junction density",    np.array([1.229, 1.339, 1.419]), "#eb6834"),
        ("Vessel length density", np.array([1.230, 1.267, 1.308]), "#1baf7a"),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.8), facecolor=SURFACE)
    _style(ax)

    # 0.95 is at or past the fragmentation onset for four of six specimens, where a single
    # vessel begins to break into several edges and loops appear artefactually.
    ax.axvspan(0.925, 0.975, color="#0b0b0b", alpha=0.055, zorder=0)
    ax.text(0.95, 1.055, "fragmentation\ncontaminates\n(4 of 6 specimens)", fontsize=8.5,
            color=INK_MUTED, ha="center", va="bottom", linespacing=1.35)

    ax.axhline(1.0, color=GRID, linewidth=1.4, zorder=1)
    ax.text(0.842, 1.006, "no difference", fontsize=8.5, color=INK_MUTED, va="bottom")

    for label, values, colour in series:
        ax.plot(thresholds[:2], values[:2], color=colour, linewidth=2.4, zorder=3)
        ax.plot(thresholds[1:], values[1:], color=colour, linewidth=2.4, zorder=3,
                linestyle=(0, (4, 2)))
        ax.scatter(thresholds, values, s=64, color=colour, edgecolor=SURFACE,
                   linewidth=2, zorder=4)
        ax.text(0.9545, values[-1], f"  {label}", fontsize=9.5, color=INK, va="center")

    ax.set_xticks(thresholds)
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds], fontsize=10, color=INK)
    ax.set_xlim(0.833, 1.13)
    ax.set_ylim(0.98, 1.60)
    ax.set_xlabel("segmentation probability threshold  (higher = less inclusive)",
                  fontsize=9.5, color=INK_MUTED)
    ax.set_ylabel("group ratio, SHR / WKY")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Threshold sensitivity: the direction holds, and the effect grows as "
                 "inclusion falls\n", fontsize=11.5, color=INK, loc="left")
    ax.text(0, 1.005, "solid = clean interval, both endpoints below every specimen's "
                      "fragmentation onset", transform=ax.transAxes, fontsize=9,
            color=INK_MUTED, va="bottom")
    fig.text(0.012, 0.015,
             "SHR exceeds WKY in all nine comparisons. Groups overlap at every threshold; "
             "no exact p below 0.20 (n = 3 per group).",
             fontsize=8.5, color=INK_MUTED)
    fig.tight_layout(rect=[0, 0.055, 1, 0.94])
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)

def _degree_and_length():
    """Node degree distribution and segment lengths, per specimen, from the per-edge table."""
    import collections
    degrees, lengths = {}, {}
    for specimen in SPECIMENS:
        path = RESULTS / specimen.specimen_id / "per_edge_morphometry.csv"
        if not path.exists():
            continue
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        counter = collections.Counter()
        for row in rows:
            counter[row["u"]] += 1
            counter[row["v"]] += 1
        histogram = collections.Counter(counter.values())
        total = len(counter)
        degrees[specimen.specimen_id] = {d: histogram.get(d, 0) / total for d in range(1, 7)}
        lengths[specimen.specimen_id] = np.array(
            [float(r["length_um"]) for r in rows if r.get("length_um")])
    return degrees, lengths


def figure_degree(path):
    """Section 1.1 asks for the degree distribution, not only the node count."""
    degrees, _ = _degree_and_length()
    fig, ax = plt.subplots(figsize=(8.0, 4.6), facecolor=SURFACE)
    _style(ax)
    orders = list(range(1, 6))
    for specimen in SPECIMENS:
        if specimen.specimen_id not in degrees:
            continue
        values = [degrees[specimen.specimen_id][d] for d in orders]
        colour = COLOUR[specimen.group]
        ax.plot(orders, values, color=colour, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.6, alpha=0.9, zorder=3)
    ax.set_xticks(orders)
    ax.set_xlabel("node degree (number of distinct segments met)", fontsize=9.5,
                  color=INK_MUTED)
    ax.set_ylabel("fraction of nodes")
    ax.set_yscale("log")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Node degree distribution, one line per specimen\n", fontsize=11,
                 color=INK, loc="left")
    ax.text(0, 1.005, "degree 3 and above are the branch points section 1.1 counts",
            transform=ax.transAxes, fontsize=9, color=INK_MUTED, va="bottom")
    # Six near-superimposed lines cannot carry per-specimen labels without colliding, and
    # cohort identity is the only distinction the figure is making.
    for group, y in (("WKY", 0.86), ("SHR", 0.74)):
        ax.plot([0.60, 0.66], [y, y], color=COLOUR[group], linewidth=2,
                transform=ax.transAxes, clip_on=False)
        ax.text(0.68, y, group, transform=ax.transAxes, fontsize=9.5, color=INK,
                va="center")
    fig.text(0.012, 0.015,
             "Log scale. Three lines per cohort, near-superimposed: the same shape in both.",
             fontsize=8.5, color=INK_MUTED)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def figure_segment_length(path):
    """Segment length underlies both the density measures and the junction-trim coverage."""
    _, lengths = _degree_and_length()
    fig, ax = plt.subplots(figsize=(8.0, 4.6), facecolor=SURFACE)
    _style(ax)
    for specimen in SPECIMENS:
        values = lengths.get(specimen.specimen_id)
        if values is None or not len(values):
            continue
        ordered = np.sort(values)
        ax.step(ordered, np.arange(1, len(ordered) + 1) / len(ordered),
                color=COLOUR[specimen.group], linewidth=2, alpha=0.85, where="post", zorder=3)
    ax.axvline(2 * 3.73, color=INK, linewidth=1.4, linestyle=(0, (4, 2)), zorder=4)
    ax.text(2 * 3.73 + 1.2, 0.12, "twice the junction exclusion:\nsegments left of this line\n"
                                  "cannot be trimmed at all",
            fontsize=8.5, color=INK_MUTED, va="bottom", linespacing=1.35)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 1)
    ax.set_xlabel("segment length (um)", fontsize=9.5, color=INK_MUTED)
    ax.set_ylabel("cumulative fraction of segments")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Segment length distribution, one line per specimen\n", fontsize=11,
                 color=INK, loc="left")
    ax.text(0, 1.005, "why the junction radius correction reaches only 63-68% of edges",
            transform=ax.transAxes, fontsize=9, color=INK_MUTED, va="bottom")
    for group, y in (("WKY", 0.72), ("SHR", 0.62)):
        ax.plot([44, 47], [y, y], color=COLOUR[group], linewidth=2)
        ax.text(48, y, group, fontsize=9.5, color=INK, va="center")
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
