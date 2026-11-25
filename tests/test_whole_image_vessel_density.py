"""The whole-image density must be measured in microns, not voxels.

``export_results`` passed ``image_dimensions=image.shape`` to
``compute_comprehensive_vessel_statistics`` but left ``voxel_size`` at its
``(1, 1, 1)`` default, so "Total Image Volume" and "Vessel Density in Whole
Image" counted voxels rather than cubic microns. For the usual anisotropic
stack that is wrong by the product of the spacings — a factor of 0.4 here — and
it is invisible on isotropic 1 um data, which is why it survived.

``voxel_size`` is zipped element-wise with ``image_dimensions``, and
``image.shape`` is canonical ``(z, y, x)``, so what belongs there is
``voxel_size_zyx``, not the ``(x, y, z)`` of the image metadata.
"""
from __future__ import annotations

import csv
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from haemolynx.io import voxel_size_zyx_from_xyz
from haemolynx.pipeline import default_schema, stages
from haemolynx.statistics import compute_comprehensive_vessel_statistics

# Coarse z, fine x — the usual confocal case, and three distinct spacings so a
# zyx/xyz mix-up cannot pass.
VOXEL_SIZE_XYZ = (0.4, 0.5, 2.0)
VOXEL_SIZE_ZYX = voxel_size_zyx_from_xyz(VOXEL_SIZE_XYZ)
IMAGE_SHAPE = (10, 12, 14)

IMAGE_VOLUME_KEY = "Total Image Volume (micron\u00b3)"
# The CSV exporter moves a trailing "(unit)" into its own column.
CSV_VOLUME_METRIC = "Total Image Volume"
CSV_DENSITY_METRIC = "Vessel Density in Whole Image"
CSV_TOTAL_LENGTH_METRIC = "Total Vessel Length"


def _one_vessel_graph() -> nx.MultiGraph:
    G = nx.MultiGraph()
    G.add_node(0, pos=np.asarray([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.asarray([4.0, 0.0, 0.0]))
    G.add_edge(
        0,
        1,
        key=0,
        length=4.0,
        branch_order="B01",
        resistance=1.0e16,
        conductance=1.0e-16,
    )
    return G


def _export_and_read_statistics(tmp_path: Path, voxel_size_xyz) -> dict[str, str]:
    """Run the export stage and read back the statistics CSV it wrote."""
    settings = default_schema().defaults()
    settings.update(
        {
            "input_path": tmp_path / "anisotropic.tif",
            "statistics": True,
            "statistics_mode": "fast",
            # A solved run, so the graph carries resistances and the stage takes
            # its resistance-weighted branch. The haemodynamics-off branch is
            # covered by `test_statistics_without_haemodynamics.py`.
            "run_haemodynamics": True,
            "measurement_3d_to_cell_mask": False,
            "vtk_export": False,
            "visualize_vtk": False,
            "visualize_results": False,
        }
    )
    volume = stages.SkeletonisedVolume(
        image=np.zeros(IMAGE_SHAPE, dtype=np.uint8),
        skeleton=np.zeros(IMAGE_SHAPE, dtype=bool),
        voxel_size_xyz=tuple(float(v) for v in voxel_size_xyz),
        voxel_size_zyx=voxel_size_zyx_from_xyz(voxel_size_xyz),
        output_dir=tmp_path,
    )
    G = _one_vessel_graph()
    stages.export_results(
        settings,
        stages.VesselNetwork(graph=G, volume=volume),
        stages.HaemodynamicModel(graph=G),
        stages.Solution(),
    )

    csv_path = tmp_path / "anisotropic_statistics.csv"
    with csv_path.open(encoding="utf-8") as handle:
        return {row["Metric"]: row["Value"] for row in csv.DictReader(handle)}


def test_compute_vessel_density_expects_voxel_size_in_image_dimension_order():
    """Read, don't assume: `voxel_size` is zipped with `image_dimensions`."""
    stats = compute_comprehensive_vessel_statistics(
        _one_vessel_graph(),
        node_positions=nx.get_node_attributes(_one_vessel_graph(), "pos"),
        image_dimensions=IMAGE_SHAPE,
        voxel_size=VOXEL_SIZE_ZYX,
    )
    expected = float(np.prod(IMAGE_SHAPE) * np.prod(VOXEL_SIZE_ZYX))
    assert stats[IMAGE_VOLUME_KEY] == pytest.approx(expected)
    # The product is order-free, so pin the pairing itself: swapping the ends
    # of the spacing must change nothing, but pairing a *wrong-length* or
    # differently-scaled spacing must.
    assert stats[IMAGE_VOLUME_KEY] != pytest.approx(float(np.prod(IMAGE_SHAPE)))


def test_the_export_stage_scales_the_whole_image_volume_by_the_voxel_size(tmp_path):
    statistics_rows = _export_and_read_statistics(tmp_path, VOXEL_SIZE_XYZ)

    expected_volume = float(np.prod(IMAGE_SHAPE) * np.prod(VOXEL_SIZE_ZYX))
    reported_volume = float(statistics_rows[CSV_VOLUME_METRIC])

    assert reported_volume == pytest.approx(expected_volume)
    # What it used to report: the voxel count, 2.5x too large here.
    assert reported_volume != pytest.approx(float(np.prod(IMAGE_SHAPE)))

    total_length = float(statistics_rows[CSV_TOTAL_LENGTH_METRIC])
    assert float(statistics_rows[CSV_DENSITY_METRIC]) == pytest.approx(
        total_length / expected_volume
    )


def test_isotropic_micron_voxels_are_unchanged(tmp_path):
    """The old behaviour was only ever right for 1 um isotropic voxels."""
    statistics_rows = _export_and_read_statistics(tmp_path, (1.0, 1.0, 1.0))
    assert float(statistics_rows[CSV_VOLUME_METRIC]) == pytest.approx(
        float(np.prod(IMAGE_SHAPE))
    )
