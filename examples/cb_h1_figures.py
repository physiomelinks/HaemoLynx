"""Figures for the H1 preliminary comparison.

Every panel is built so the reader can see the thing that matters most about these results:
with n = 3 per group the groups can overlap, so every specimen is drawn as its own point and
the group mean is a rule behind them, never a bar.

    Figure 1  network density - loop, junction and vessel length density per mm3
    Figure 2  per-edge diameter distribution, with the measurement's quantisation shown
    Figure 3  group ratio against threshold, at the frozen value and its grid neighbours
    Figure 7  node degree distribution
    Figure 8  segment length distribution

Every number drawn or printed on a figure is read from the batch and sensitivity outputs
(re-run package D). Figure 1 once drew its junction and length panels from a table copied
out of an old run's logs, and went on showing SHR +34% / +27% after the data had moved to
+6% / +2%; nothing here is copied by hand any more. Group comparisons use the exact
permutation p of `assess_cohort_split`, whose floor at n = 3 is 0.10.

Figure 2 draws the quantisation grid deliberately, so that a group gap can be read against
the EDT step it has to be resolved by.
"""
import argparse
import collections
import dataclasses
import functools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ImageLynx.specimens import (
    PROCESSING_VOXEL_UM, SPECIMENS, get_specimen, sensitivity_run_name,
)
from ImageLynx import cb_settings
from ImageLynx.batch_outputs import open_batch_run
from ImageLynx.statistics.cohort_split import CohortSplit, assess_cohort_split
from ImageLynx.statistics.threshold_selection import CAPILLARY_DIAMETER_RANGE_UM

from cb_h1_batch import OUTPUT_DIR, SENSITIVITY_DIR, sensitivity_thresholds

# Categorical slots 1 and 2 of the reference palette, validated for CVD separation
# (worst adjacent pair dE 24.7 protan) and >= 3:1 contrast on the light surface.
COLOUR = {"WKY": "#2a78d6", "SHR": "#eb6834"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"

ROI_VOXELS = int(np.prod(cb_settings.ROI_VOXELS))
ROI_MM3 = ROI_VOXELS * float(np.prod(PROCESSING_VOXEL_UM)) / 1e9
VOXEL_UM = PROCESSING_VOXEL_UM[1]

# (key, title, unit, scale, scale label) for the three density measures.
MEASURES = (
    ("beta1", "β₁ loop density", "independent loops per mm³", 1e3, "×10³"),
    ("junctions", "Junction density", "junctions per mm³", 1e3, "×10³"),
    ("length", "Vessel length density", "µm per mm³", 1e6, "×10⁶"),
)
# Figure 3 colours: the two categorical slots above plus the aqua slot.
SERIES_COLOUR = {"beta1": "#2a78d6", "junctions": "#eb6834", "length": "#1baf7a"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=3)
    ax.yaxis.label.set_color(INK_MUTED)


@functools.lru_cache(maxsize=None)
def _run(run_dir, specimen_id):
    """One specimen's opened batch run.

    ``run_dir`` is a sensitivity threshold's folder holding one run per specimen, or None for
    the batch. Opening the run checks its placed ROI (open item 27). Cached, because an open
    places the ROI and the figures read each run several times.
    """
    specimen = get_specimen(specimen_id)
    return open_batch_run(specimen, None if run_dir is None else Path(run_dir) / specimen_id)


def _rows(run_dir, specimen_id):
    """One specimen's per-edge table, in file order. A missing table raises: a figure must not
    drop a specimen."""
    return tuple(_run(run_dir, specimen_id).edge_table().values())


def _column(run_dir, specimen_id, column):
    """One numeric column of a specimen's per-edge table as Python floats, in file order. A
    blank or non-finite cell raises in the reader rather than being skipped here."""
    return tuple(_run(run_dir, specimen_id).numeric_column(column).values())


def _degrees(rows):
    """Node degree on the MultiGraph: a parallel edge counts at both its ends (open item 40)."""
    degree = collections.Counter()
    for row in rows:
        degree[row["u"]] += 1
        degree[row["v"]] += 1
    return degree


def network_measures(run_dir=None):
    """Per-specimen loop, junction and vessel length density per mm3, from the per-edge table.

    beta-1 = E - V + C, the loop count H1 s1.1 names and the pipeline does not report. C = 1
    because GraphConfig.keep_largest_component_only is set, so the graph handed to
    morphometry is a single connected component by construction. Junctions are nodes of
    degree 3 or more; length is the sum of the edge polylines.
    """
    out = {}
    for specimen in SPECIMENS:
        rows = _rows(run_dir, specimen.specimen_id)
        degree = _degrees(rows)
        out[specimen.specimen_id] = {
            "beta1": (len(rows) - len(degree) + 1) / ROI_MM3,
            "junctions": sum(1 for d in degree.values() if d >= 3) / ROI_MM3,
            "length": sum(_column(run_dir, specimen.specimen_id, "length_um")) / ROI_MM3,
        }
    return out


@dataclasses.dataclass(frozen=True)
class GroupSummary:
    """Group means of one per-specimen quantity, and whether it separates by cohort."""

    wky: float
    shr: float
    split: CohortSplit

    @property
    def ratio(self):
        return self.shr / self.wky

    @property
    def text(self):
        state = "groups separate" if self.split.separated else "groups overlap"
        return (f"SHR {100 * (self.ratio - 1):+.1f}%  ·  {state}  ·  "
                f"exact p = {self.split.permutation_p:.2f}")


def group_summary(values_by_specimen, quantity="value"):
    groups = {s.specimen_id: s.group for s in SPECIMENS}
    wky = float(np.mean([v for s, v in values_by_specimen.items() if groups[s] == "WKY"]))
    shr = float(np.mean([v for s, v in values_by_specimen.items() if groups[s] == "SHR"]))
    return GroupSummary(wky, shr, assess_cohort_split(values_by_specimen, quantity, groups))


def sensitivity_runs():
    """Run folder per threshold: the frozen batch (None, the default run) and its two neighbours."""
    low, high = sensitivity_thresholds(cb_settings.FROZEN_THRESHOLD)
    return {low: SENSITIVITY_DIR / sensitivity_run_name(low),
            cb_settings.FROZEN_THRESHOLD: None,
            high: SENSITIVITY_DIR / sensitivity_run_name(high)}


def sensitivity_series():
    """Thresholds, and for each measure the GroupSummary at every threshold."""
    runs = sensitivity_runs()
    thresholds = sorted(runs)
    measures = {t: network_measures(runs[t]) for t in thresholds}
    series = {}
    for key, title, *_ in MEASURES:
        series[key] = [group_summary({s: m[key] for s, m in measures[t].items()}, title)
                       for t in thresholds]
    return thresholds, series


def fragmentation_onsets():
    """Each specimen's fragmentation onset from the threshold stage; None if it never fragments."""
    path = OUTPUT_DIR / "threshold_selection.json"
    data = json.loads(path.read_text())
    if "fragmentation_onset" not in data:
        raise KeyError(f"{path} has no 'fragmentation_onset'. Re-run cb_h1_batch.py "
                       "--stage threshold, which records it.")
    onsets = data["fragmentation_onset"]
    missing = [s.specimen_id for s in SPECIMENS if s.specimen_id not in onsets]
    if missing:
        raise KeyError(f"{path} has no fragmentation onset for {', '.join(missing)}.")
    return {s.specimen_id: (None if onsets[s.specimen_id] is None
                            else float(onsets[s.specimen_id])) for s in SPECIMENS}


def median_diameters(diameters):
    return {sid: float(np.median(values)) for sid, values in diameters.items()}


def junction_trim_shares():
    """Share of each specimen's edges the junction radius correction was applied to."""
    out = {}
    for specimen in SPECIMENS:
        rows = _rows(None, specimen.specimen_id)
        out[specimen.specimen_id] = (
            sum(1 for r in rows if r["edt_junction_trim"] == "trimmed") / len(rows))
    return out


def junction_exclusion_um():
    """The pipeline's junction exclusion distance, read from its config default.

    cb_h1_batch.py passes no --config, so the batch ran at this default.
    """
    from carotid_image_to_model import HaemodynamicsConfig
    fields = {f.name: f for f in dataclasses.fields(HaemodynamicsConfig)}
    return float(fields["edt_junction_proximity_exclusion_um"].default)


def _load_diameters():
    out = {}
    for specimen in SPECIMENS:
        out[specimen.specimen_id] = np.asarray(_column(None, specimen.specimen_id,
                                                       "edt_diameter_um"))
    return out


def _draw_groups(ax, values):
    """Per-specimen points with a group-mean rule; no bars, because n = 3."""
    for index, group in enumerate(("WKY", "SHR")):
        members = [s.specimen_id for s in SPECIMENS if s.group == group]
        points = np.array([values[m] for m in members])
        jitter = np.linspace(-0.10, 0.10, len(points))
        ax.scatter(index + jitter, points, s=88, color=COLOUR[group],
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.hlines(points.mean(), index - 0.28, index + 0.28,
                  color=COLOUR[group], linewidth=2, zorder=2)
        yield members, index, jitter, points


def figure_density(path):
    measures = network_measures()
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4), facecolor=SURFACE)

    for ax, (key, title, ylabel, scale, suffix) in zip(axes, MEASURES):
        _style(ax)
        values = {sid: m[key] / scale for sid, m in measures.items()}
        for members, index, jitter, points in _draw_groups(ax, values):
            for offset, value, name in zip(jitter, points, members):
                ax.annotate(name.split("-")[1], (index + offset, value),
                            textcoords="offset points", xytext=(11, -3),
                            fontsize=8, color=INK_MUTED)

        summary = group_summary({sid: m[key] for sid, m in measures.items()}, title)
        ax.set_title(f"{title}\n", fontsize=11, color=INK, loc="left")
        ax.text(0, 1.005, summary.text,
                transform=ax.transAxes, fontsize=9.5, color=INK_MUTED, va="bottom")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["WKY", "SHR"], fontsize=10, color=INK)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylabel(f"{ylabel}  ({suffix})")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    fig.suptitle("Carotid body network density, matched "
                 f"{ROI_MM3:.4f} mm³ region per specimen, threshold "
                 f"{cb_settings.FROZEN_THRESHOLD:.2f}",
                 fontsize=12.5, color=INK, x=0.040, ha="left", y=0.99)
    fig.text(0.040, 0.015,
             "One point per specimen (n = 3 per group); rule is the group mean. Exact two-sided "
             "permutation p on the difference in means. Preliminary: segmentation classifier "
             "not final.",
             fontsize=8.5, color=INK_MUTED)
    fig.tight_layout(rect=[0, 0.045, 1, 0.93])
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def figure_diameter(path, diameters):
    """ECDF per specimen, plus the quantisation grid a group gap has to be read against."""
    lo_um, hi_um = CAPILLARY_DIAMETER_RANGE_UM
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    _style(ax)
    for specimen in SPECIMENS:
        ordered = np.sort(diameters[specimen.specimen_id])
        ax.step(ordered, np.arange(1, len(ordered) + 1) / len(ordered),
                color=COLOUR[specimen.group], linewidth=2, alpha=0.85,
                where="post", zorder=3)
    for edge in np.arange(0, 26, VOXEL_UM):
        ax.axvline(edge, color=GRID, linewidth=0.8, zorder=1)
    ax.axvspan(lo_um, hi_um, color="#0b0b0b", alpha=0.045, zorder=0)
    ax.text((lo_um + hi_um) / 2, 0.62, f"expected\ncapillary\n{lo_um:g}–{hi_um:g} µm",
            fontsize=8.5, color=INK_MUTED,
            ha="center", va="center", linespacing=1.3, zorder=4,
            bbox=dict(boxstyle="round,pad=0.28", facecolor=SURFACE, edgecolor="none",
                      alpha=0.88))
    ax.set_xlim(0, 22); ax.set_ylim(0, 1)
    ax.set_xlabel("per-edge inscribed diameter (µm)", fontsize=9.5, color=INK_MUTED)
    ax.set_ylabel("cumulative fraction of edges")
    ax.set_title("Diameter distribution, one line per specimen\n", fontsize=11,
                 color=INK, loc="left")
    ax.text(0, 1.005, f"vertical rules mark the {VOXEL_UM:.2f} µm measurement step",
            transform=ax.transAxes, fontsize=9, color=INK_MUTED, va="bottom")
    for group, y in (("WKY", 0.93), ("SHR", 0.86)):
        ax.plot([15.4, 16.6], [y, y], color=COLOUR[group], linewidth=2)
        ax.text(17.0, y, group, fontsize=9.5, color=INK, va="center")

    ax = axes[1]
    _style(ax)
    medians = median_diameters(diameters)
    for _ in _draw_groups(ax, medians):
        pass
    split = group_summary(medians, "median diameter").split
    low, high = min(medians.values()), max(medians.values())
    centre = (low + high) / 2
    # Draw the step as a bounded interval, not a wash across the panel: the axis is opened
    # out so the band reads as a measured quantity the group gap has to be compared against.
    half_range = max(1.65 * VOXEL_UM, (high - low) / 2 + 0.3 * VOXEL_UM)
    ax.set_ylim(centre - half_range, centre + half_range)
    ax.add_patch(plt.Rectangle((-0.40, centre - VOXEL_UM / 2), 1.80, VOXEL_UM,
                               facecolor="#0b0b0b", alpha=0.06, edgecolor=GRID,
                               linewidth=1, zorder=0))
    ax.annotate("", xy=(1.52, centre + VOXEL_UM / 2), xytext=(1.52, centre - VOXEL_UM / 2),
                arrowprops=dict(arrowstyle="<->", color=INK_MUTED, linewidth=1.1))
    ax.text(1.62, centre, f"one {VOXEL_UM:.2f} µm\nmeasurement step", fontsize=8.5,
            color=INK_MUTED, va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["WKY", "SHR"], fontsize=10, color=INK)
    ax.set_xlim(-0.5, 2.9)
    ax.set_ylabel("median diameter (µm)")
    ax.set_title("Group separation against resolution\n", fontsize=11, color=INK, loc="left")
    subtitle = (f"gap {split.gap:.2f} µm = {split.gap / VOXEL_UM:.2f} of one step"
                if split.separated else "groups overlap")
    ax.text(0, 1.005, subtitle,
            transform=ax.transAxes, fontsize=9, color=INK_MUTED, va="bottom")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    in_window = sum(1 for m in medians.values() if lo_um <= m <= hi_um)
    state = (f"Medians separate completely by group ({split.direction})"
             if split.separated else "Medians overlap between groups")
    fig.suptitle("Per-edge vessel diameter", fontsize=12.5, color=INK, x=0.045,
                 ha="left", y=0.99)
    fig.text(0.045, 0.015,
             f"{state}; exact p = {split.permutation_p:.2f} (floor {split.floor_p:.2f} at "
             f"n = 3). {in_window} of {len(medians)} medians lie in the {lo_um:g}–{hi_um:g} µm "
             f"window.\nEDT diameters, not corrected for the half-voxel bias (reference open "
             f"item 42).",
             fontsize=8.5, color=INK_MUTED)
    fig.tight_layout(rect=[0, 0.075, 1, 0.90])
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR,
                        help="Folder to write the figures into (default: %(default)s). The "
                             "tables are always read from the batch and sensitivity runs.")
    args = parser.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    diameters = _load_diameters()
    density_path = out / "figure1_network_density.png"
    diameter_path = out / "figure2_diameter_distribution.png"
    figure_density(density_path)
    figure_diameter(diameter_path, diameters)
    sensitivity_path = out / "figure3_threshold_sensitivity.png"
    figure_sensitivity(sensitivity_path)
    degree_path = out / "figure7_node_degree.png"
    figure_degree(degree_path)
    length_path = out / "figure8_segment_length.png"
    figure_segment_length(length_path)
    for written in (density_path, diameter_path, sensitivity_path, degree_path, length_path):
        print(f"wrote {written}")


def figure_sensitivity(path):
    """Group ratio against threshold — robustness, and the predicted direction.

    Three series, so identity is carried by a direct label on each line rather than a legend
    box; the aqua slot sits below 3:1 on the light surface and the relief rule applies.
    An interval is dashed when its upper end reaches any specimen's fragmentation onset, where
    a single vessel begins to break into several edges and loops appear artefactually.
    """
    thresholds, series = sensitivity_series()
    onsets = fragmentation_onsets()
    known = [o for o in onsets.values() if o is not None]
    contaminated = [t for t in thresholds if any(o <= t for o in known)]
    step = min(np.diff(thresholds))

    fig, ax = plt.subplots(figsize=(8.2, 4.8), facecolor=SURFACE)
    _style(ax)

    ratios = np.array([[s.ratio for s in series[key]] for key, *_ in MEASURES])
    span = max(ratios.max(), 1.0) - min(ratios.min(), 1.0)
    y_low = min(ratios.min(), 1.0) - 0.12 * span
    y_high = max(ratios.max(), 1.0) + 0.45 * span

    if contaminated:
        n_fragmenting = sum(1 for o in known if o <= thresholds[-1])
        ax.axvspan(contaminated[0] - step / 2, contaminated[-1] + step / 2,
                   color="#0b0b0b", alpha=0.055, zorder=0)
        ax.text((contaminated[0] + contaminated[-1]) / 2, 0.97,
                f"fragmentation\ncontaminates\n({n_fragmenting} of {len(onsets)} specimens)",
                transform=ax.get_xaxis_transform(), fontsize=8.5, color=INK_MUTED,
                ha="center", va="top", linespacing=1.35)

    x_low, x_high = thresholds[0] - 0.35 * step, thresholds[-1] + 1.6 * step
    ax.axhline(1.0, color=GRID, linewidth=1.4, zorder=1)
    ax.text(x_low + 0.05 * step, 1.0 + 0.01 * span, "no difference", fontsize=8.5,
            color=INK_MUTED, va="bottom")

    for (key, label, *_), values in zip(MEASURES, ratios):
        colour = SERIES_COLOUR[key]
        for i in range(len(thresholds) - 1):
            dashed = any(o <= thresholds[i + 1] for o in known)
            ax.plot(thresholds[i:i + 2], values[i:i + 2], color=colour, linewidth=2.4,
                    zorder=3, linestyle=(0, (4, 2)) if dashed else "-")
        ax.scatter(thresholds, values, s=64, color=colour, edgecolor=SURFACE,
                   linewidth=2, zorder=4)
        ax.text(thresholds[-1] + 0.12 * step, values[-1], f"  {label}", fontsize=9.5,
                color=INK, va="center")

    ax.set_xticks(thresholds)
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds], fontsize=10, color=INK)
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)
    ax.set_xlabel("segmentation probability threshold  (higher = less inclusive)",
                  fontsize=9.5, color=INK_MUTED)
    ax.set_ylabel("group ratio, SHR / WKY")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    above = int((ratios > 1).sum())
    rising = all(np.all(np.diff(values) > 0) for values in ratios)
    if above == ratios.size and rising:
        title = "Threshold sensitivity: the direction holds, and the ratio grows as inclusion falls"
    elif above == ratios.size:
        title = "Threshold sensitivity: the direction holds at every threshold"
    else:
        title = "Threshold sensitivity of the SHR / WKY ratio"
    ax.set_title(f"{title}\n", fontsize=11.5, color=INK, loc="left")
    ax.text(0, 1.005, "solid = clean interval, both endpoints below every specimen's "
                      "fragmentation onset", transform=ax.transAxes, fontsize=9,
            color=INK_MUTED, va="bottom")
    splits = [s.split for key, *_ in MEASURES for s in series[key]]
    overlap = ("Groups overlap at every threshold" if not any(s.separated for s in splits)
               else f"Groups separate in {sum(s.separated for s in splits)} of {len(splits)}")
    fig.text(0.012, 0.015,
             f"SHR exceeds WKY in {above} of {ratios.size} comparisons. {overlap}; smallest "
             f"exact p {min(s.permutation_p for s in splits):.2f} (floor "
             f"{splits[0].floor_p:.2f} at n = 3 per group).",
             fontsize=8.5, color=INK_MUTED)
    fig.tight_layout(rect=[0, 0.055, 1, 0.94])
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def _degree_and_length():
    """Node degree distribution and segment lengths, per specimen, from the per-edge table."""
    degrees, lengths = {}, {}
    for specimen in SPECIMENS:
        rows = _rows(None, specimen.specimen_id)
        counter = _degrees(rows)
        histogram = collections.Counter(counter.values())
        total = len(counter)
        degrees[specimen.specimen_id] = {d: histogram.get(d, 0) / total for d in range(1, 7)}
        lengths[specimen.specimen_id] = np.array(_column(None, specimen.specimen_id,
                                                         "length_um"))
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
    exclusion = junction_exclusion_um()
    shares = junction_trim_shares()
    fig, ax = plt.subplots(figsize=(8.0, 4.6), facecolor=SURFACE)
    _style(ax)
    for specimen in SPECIMENS:
        ordered = np.sort(lengths[specimen.specimen_id])
        ax.step(ordered, np.arange(1, len(ordered) + 1) / len(ordered),
                color=COLOUR[specimen.group], linewidth=2, alpha=0.85, where="post", zorder=3)
    ax.axvline(2 * exclusion, color=INK, linewidth=1.4, linestyle=(0, (4, 2)), zorder=4)
    ax.text(2 * exclusion + 1.2, 0.12, "twice the junction exclusion:\nsegments left of this line\n"
                                       "cannot be trimmed at all",
            fontsize=8.5, color=INK_MUTED, va="bottom", linespacing=1.35)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 1)
    ax.set_xlabel("segment length (µm)", fontsize=9.5, color=INK_MUTED)
    ax.set_ylabel("cumulative fraction of segments")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Segment length distribution, one line per specimen\n", fontsize=11,
                 color=INK, loc="left")
    ax.text(0, 1.005, f"why the junction radius correction reaches only "
                      f"{100 * min(shares.values()):.0f}-{100 * max(shares.values()):.0f}% "
                      f"of edges",
            transform=ax.transAxes, fontsize=9, color=INK_MUTED, va="bottom")
    for group, y in (("WKY", 0.72), ("SHR", 0.62)):
        ax.plot([44, 47], [y, y], color=COLOUR[group], linewidth=2)
        ax.text(48, y, group, fontsize=9.5, color=INK, va="center")
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
