"""One marked-grid case drawn as four panels: the whole image's centrelines
seen down z, then the segmentation, dextran and claudin-5 in a window round
the vessel, each projected through the vessel's own thinnest axis over the
depth it spans. Shared by grid_av1.py and grid_sz1.py."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DROPBOX = Path("C:/Users/hd01/Dropbox")
STEM = "Composite_06082026_E14p5_clnd5_25x_940nm_1040_texasred3kkda_zstack_spot1_1p5z_MCAregion"
SEGMENTATION = DROPBOX / f"Claudin_{STEM}_Simple Segmentation.tiff"
RAW = DROPBOX / f"{STEM}.tif"

#: Half the side of a case's window, at least; it grows to hold the vessel.
HALF_WINDOW_UM = 60.0
#: The projection's depth either side of the vessel.
SLAB_PAD_UM = 8.0


@dataclass
class Scene:
    """What every case is drawn from: the graph, the volumes and every centreline."""

    G: object
    vs: np.ndarray
    seg: np.ndarray
    dextran: np.ndarray
    claudin: np.ndarray
    extent: np.ndarray
    #: ((u, v, key), polyline) for every edge.
    lines: list


def edge_key(row: dict) -> tuple:
    def node(text):
        return int(text) if text.lstrip("-").isdigit() else text

    return node(row["u"]), node(row["v"]), int(row["key"])


def load_scene(G) -> Scene:
    import tifffile

    from measures import polyline

    vs = np.asarray(G.graph.get("voxel_size_zyx") or G.graph.get("image_voxel_size_zyx"), float)
    seg = tifffile.imread(SEGMENTATION) == 1
    raw = tifffile.imread(RAW)
    if not (raw.ndim == 4 and raw.shape[1] == 2):
        raise SystemExit(f"raw image shape {raw.shape}: expected ZCYX with 2 channels")
    dextran, claudin = raw[:, 0], raw[:, 1]
    if seg.shape != dextran.shape:
        raise SystemExit(f"segmentation {seg.shape} and raw {dextran.shape} differ")
    lines = [((u, v, k), polyline(G, u, v, d)) for u, v, k, d in G.edges(keys=True, data=True)]
    return Scene(G, vs, seg, dextran, claudin, np.asarray(seg.shape) * vs, lines)


def draw_case(axrow, case_id: int, key: tuple, scene: Scene, highlight: set) -> dict:
    """Case *case_id* (the edge *key*) drawn into the four axes *axrow*,
    the edges in *highlight* in blue. Returns where it was drawn."""
    from matplotlib.patches import Rectangle

    from measures import polyline

    G, vs, seg = scene.G, scene.vs, scene.seg
    pts = polyline(G, key[0], key[1], G.edges[key])
    lo_v, hi_v = pts.min(axis=0), pts.max(axis=0)
    spread = hi_v - lo_v
    normal = int(np.argmin(spread / vs))  # project through the thinnest axis, in voxels
    across = [a for a in range(3) if a != normal]
    centre = 0.5 * (lo_v + hi_v)
    half = np.maximum(HALF_WINDOW_UM, 0.6 * spread[across].max() + 10.0)
    lo = centre.copy()
    hi = centre.copy()
    for a in across:
        lo[a], hi[a] = centre[a] - half, centre[a] + half
    lo[normal], hi[normal] = lo_v[normal] - SLAB_PAD_UM, hi_v[normal] + SLAB_PAD_UM
    lo_i = np.clip(np.floor(lo / vs).astype(int), 0, np.asarray(seg.shape) - 1)
    hi_i = np.clip(np.ceil(hi / vs).astype(int) + 1, 1, np.asarray(seg.shape))
    box = tuple(slice(a, b) for a, b in zip(lo_i, hi_i))
    names = "zyx"
    panels = [
        (seg[box].max(axis=normal).astype(float), "segmentation", "gray"),
        (scene.dextran[box].max(axis=normal).astype(float), "dextran", "magma"),
        (scene.claudin[box].max(axis=normal).astype(float), "claudin-5", "viridis"),
    ]
    row_ax, col_ax = across
    extent_px = [lo_i[col_ax] * vs[col_ax], hi_i[col_ax] * vs[col_ax],
                 hi_i[row_ax] * vs[row_ax], lo_i[row_ax] * vs[row_ax]]
    extent = scene.extent

    ax0 = axrow[0]
    for k2, p2 in scene.lines:
        if k2 != key and k2 not in highlight:
            ax0.plot(p2[:, 2], p2[:, 1], color="0.75", lw=0.4)
    for k2, p2 in scene.lines:
        if k2 in highlight and k2 != key:
            ax0.plot(p2[:, 2], p2[:, 1], color="tab:blue", lw=1.0)
    ax0.plot(pts[:, 2], pts[:, 1], color="red", lw=2.6)
    mid = pts[len(pts) // 2]
    ax0.plot([mid[2]], [mid[1]], "o", ms=9, mfc="none", mec="red", mew=1.5)
    ax0.add_patch(Rectangle((lo[2] if 2 in across else centre[2] - half, lo[1] if 1 in across else centre[1] - half),
                            (hi[2] - lo[2]) if 2 in across else 2 * half,
                            (hi[1] - lo[1]) if 1 in across else 2 * half,
                            fill=False, ec="yellow", lw=1))
    ax0.set_xlim(0, extent[2]); ax0.set_ylim(extent[1], 0)
    ax0.set_title(f"{case_id}", fontsize=22, loc="left", fontweight="bold")
    ax0.plot([10, 110], [extent[1] - 15] * 2, color="black", lw=3)
    ax0.text(60, extent[1] - 22, "100 um", color="black", ha="center", fontsize=8)
    ax0.set_xticks([]); ax0.set_yticks([])

    for col, (img, title, cmap) in enumerate(panels, start=1):
        ax = axrow[col]
        if title == "segmentation":
            ax.imshow(img, cmap=cmap, extent=extent_px, vmin=0, vmax=1.6)
        else:
            vmin, vmax = np.percentile(img, [1, 99.7])
            ax.imshow(img, cmap=cmap, extent=extent_px, vmin=vmin, vmax=max(vmax, vmin + 1))
        for k2, p2 in scene.lines:
            if col != 1 and k2 != key:
                continue
            sel = ((p2[:, normal] >= lo[normal] - 5) & (p2[:, normal] <= hi[normal] + 5))
            if not sel.any():
                continue
            q = np.where(sel[:, None], p2, np.nan)
            if k2 == key:
                ax.plot(q[:, col_ax], q[:, row_ax], color="red", lw=2.4 if col == 1 else 1.0,
                        alpha=1.0 if col == 1 else 0.7, ls="-" if col == 1 else "--")
            elif k2 in highlight:
                ax.plot(q[:, col_ax], q[:, row_ax], color="tab:blue", lw=1.4)
            else:
                # White: grey vanished against the grey segmentation (grid AV1).
                ax.plot(q[:, col_ax], q[:, row_ax], color="white", lw=0.8, alpha=0.9)
        ax.set_xlim(extent_px[0], extent_px[1]); ax.set_ylim(extent_px[2], extent_px[3])
        x0 = extent_px[0] + 0.06 * (extent_px[1] - extent_px[0])
        y0 = extent_px[2] - 0.06 * (extent_px[2] - extent_px[3])
        ax.plot([x0, x0 + 20], [y0, y0], color="white", lw=3)
        ax.text(x0 + 10, y0 - 0.03 * (extent_px[2] - extent_px[3]), "20 um", color="white",
                ha="center", fontsize=8)
        ax.set_title(f"{title}: {names[row_ax]} down, {names[col_ax]} across, "
                     f"through {names[normal]} {lo[normal]:.0f}-{hi[normal]:.0f} um", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    mid_vox = np.round(pts[len(pts) // 2] / vs).astype(int)
    axrow[3].text(1.0, -0.02, f"voxel (z, y, x) {tuple(int(x) for x in mid_vox)}",
                  transform=axrow[3].transAxes, ha="right", va="top", fontsize=8)
    return {"projected_through": names[normal], "mid_voxel_zyx": " ".join(str(int(x)) for x in mid_vox)}


def save_single(case_id: int, key: tuple, scene: Scene, highlight: set, path: Path) -> dict:
    """The case on its own, one row of four panels, for an interactive page."""
    import matplotlib.pyplot as plt

    fig, row = plt.subplots(1, 4, figsize=(16, 4.4))
    where = draw_case(row, case_id, key, scene, highlight)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return where
