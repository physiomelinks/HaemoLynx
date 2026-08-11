"""Offscreen renderings of the reconstruction and the analysed skeleton.

These are illustrations, not evidence. Each shows a thin slab of a 0.0266 mm3 region of one
carotid body, and a reader shown one WKY beside one SHR will see a difference that three
specimens per group cannot support. The captions say so, and the figures belong with the
measurement chain rather than with the results for the same reason.

**Why a slab and not the whole region.** At the frozen threshold the segmentation occupies
26-34% of the volume, so a 160-voxel cube viewed from outside is an opaque mass however the
transparency is set - the first attempt at these figures produced exactly that, and nothing
of the network was visible. A 14 um slab, about seven voxels, resolves individual vessels and
their branching while still being a real part of the measured volume rather than a cartoon.

**Constant tube radius.** Calibre is carried by colour alone. Scaling tube radius by the same
quantity double-encodes it, and on this data produced visible ballooning artefacts where a
short segment carried a large diameter.

Rendered through VTK's software path, so no display is required and the figures regenerate
with the rest of the analysis.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ImageLynx.specimens import get_specimen  # noqa: E402

pv.OFF_SCREEN = True

VTK = Path(__file__).resolve().parent / "outputs" / "cb_h1_paraview"
OUT = Path(__file__).resolve().parent / "outputs" / "cb_h1_batch"

# Carried from the plots so cohort identity reads the same throughout the report.
WKY, SHR = "#2a78d6", "#eb6834"
SURFACE = "#fcfcfb"
INK = "#131920"
GREY = "#aab4c0"
SLAB_UM = 14.0
PAIR = ("WKY-A", "SHR-B")


def slab(mesh, reference, thickness=SLAB_UM):
    """Clip to a slab through the middle of the region, viewed face-on."""
    b = np.array(reference.bounds)
    centre = (b[4] + b[5]) / 2.0
    box = (b[0], b[1], b[2], b[3], centre - thickness / 2, centre + thickness / 2)
    return mesh.clip_box(box, invert=False).extract_surface()


def tubes(lines, radius=0.75):
    if lines.n_cells == 0:
        return lines
    return lines.cell_data_to_point_data().tube(radius=radius, n_sides=6)


def face_on(plotter, zoom=1.28):
    plotter.camera_position = "xy"
    plotter.reset_camera()
    plotter.camera.zoom(zoom)


def caption(plotter, specimen_id):
    """Only the identifier is burnt in; everything descriptive lives in the figure caption."""
    colour = WKY if get_specimen(specimen_id).group == "WKY" else SHR
    plotter.add_text(specimen_id, position=(16, 16), font_size=13, color=colour)


def figure_reconstruction(path, size=(1620, 1120)):
    """All six specimens, not a chosen pair.

    A single WKY beside a single SHR is a cherry-pick however carefully the pair is chosen -
    the first version of this figure put WKY-A against SHR-B, which is the densest SHR
    specimen, and read as far more separated than the data supports. Showing all six removes
    the choice, and makes the overlap the rest of the report insists on directly visible:
    WKY-C carries a denser network than SHR-C.
    """
    order = (("WKY-A", "WKY-B", "WKY-C"), ("SHR-A", "SHR-B", "SHR-C"))
    plotter = pv.Plotter(off_screen=True, shape=(2, 3), window_size=size, border=False)
    for row, specimens in enumerate(order):
        for column, specimen_id in enumerate(specimens):
            plotter.subplot(row, column)
            plotter.set_background(SURFACE)
            surface = pv.read(VTK / f"{specimen_id}_surface.vtp")
            vessels = pv.read(VTK / f"{specimen_id}_vessels.vtp")
            colour = WKY if get_specimen(specimen_id).group == "WKY" else SHR
            plotter.add_mesh(slab(surface, surface), color=colour, opacity=0.30,
                             smooth_shading=True, show_scalar_bar=False)
            plotter.add_mesh(tubes(slab(vessels, surface), radius=0.68), color=INK,
                             smooth_shading=True, show_scalar_bar=False)
            caption(plotter, specimen_id)
            face_on(plotter, zoom=1.18)
    plotter.screenshot(path)
    plotter.close()


def figure_measured(path, specimen_id="SHR-B", size=(1560, 810)):
    vessels = pv.read(VTK / f"{specimen_id}_vessels.vtp")
    nodes = pv.read(VTK / f"{specimen_id}_nodes.vtp")
    surface = pv.read(VTK / f"{specimen_id}_surface.vtp")
    lines = slab(vessels, surface)

    plotter = pv.Plotter(off_screen=True, shape=(1, 2), window_size=size, border=False)

    plotter.subplot(0, 0)
    plotter.set_background(SURFACE)
    plotter.add_mesh(tubes(lines), scalars="edt_diameter_um", cmap="viridis", clim=(4.0, 14.0),
                     smooth_shading=True,
                     scalar_bar_args=dict(title="EDT diameter (um)", vertical=False,
                                          position_x=0.20, position_y=0.015, width=0.60,
                                          height=0.045, title_font_size=15,
                                          label_font_size=12, color=INK, n_labels=5))
    caption(plotter, specimen_id)
    face_on(plotter)

    plotter.subplot(0, 1)
    plotter.set_background(SURFACE)
    plotter.add_mesh(tubes(lines, radius=0.5), color=GREY, smooth_shading=True,
                     show_scalar_bar=False)
    b = np.array(surface.bounds)
    centre = (b[4] + b[5]) / 2.0
    keep = ((nodes.points[:, 2] >= centre - SLAB_UM / 2)
            & (nodes.points[:, 2] <= centre + SLAB_UM / 2)
            & (nodes.point_data["is_branch_node"] == 1))
    if keep.any():
        branch = pv.PolyData(nodes.points[keep])
        plotter.add_mesh(branch.glyph(geom=pv.Sphere(radius=1.9), orient=False, scale=False),
                         color=SHR, smooth_shading=True, show_scalar_bar=False)
    caption(plotter, specimen_id)
    face_on(plotter)

    plotter.screenshot(path)
    plotter.close()


def figure_skeleton_detail(path, specimen_id="WKY-A", size=(1560, 800)):
    """Raw skeleton against the centrelines actually measured.

    The difference is stub pruning and B-spline smoothing - the operators section 4
    describes - and is easier to see than to read.
    """
    skeleton = pv.read(VTK / f"{specimen_id}_skeleton.vtp")
    vessels = pv.read(VTK / f"{specimen_id}_vessels.vtp")
    surface = pv.read(VTK / f"{specimen_id}_surface.vtp")

    b = np.array(surface.bounds)
    centre = (b[4] + b[5]) / 2.0
    keep = ((skeleton.points[:, 2] >= centre - SLAB_UM / 2)
            & (skeleton.points[:, 2] <= centre + SLAB_UM / 2))
    skeleton_slab = pv.PolyData(skeleton.points[keep])

    plotter = pv.Plotter(off_screen=True, shape=(1, 2), window_size=size, border=False)

    plotter.subplot(0, 0)
    plotter.set_background(SURFACE)
    plotter.add_mesh(skeleton_slab, color=INK, point_size=3.4,
                     render_points_as_spheres=True, show_scalar_bar=False)
    caption(plotter, specimen_id)
    face_on(plotter)

    plotter.subplot(0, 1)
    plotter.set_background(SURFACE)
    plotter.add_mesh(tubes(slab(vessels, surface), radius=0.6), color=INK,
                     smooth_shading=True, show_scalar_bar=False)
    caption(plotter, specimen_id)
    face_on(plotter)

    plotter.screenshot(path)
    plotter.close()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for path, builder in ((OUT / "figure4_reconstruction.png", figure_reconstruction),
                          (OUT / "figure5_measured_network.png", figure_measured),
                          (OUT / "figure6_skeleton_detail.png", figure_skeleton_detail)):
        builder(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
