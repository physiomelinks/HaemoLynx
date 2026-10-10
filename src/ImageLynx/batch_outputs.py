"""The one reader of a batch run: what ``cb_h1_batch.py`` wrote for one specimen.

A batch run is one specimen's output folder, from ``--stage run`` (``Specimen.batch_run_dir``)
or ``--stage sensitivity`` (opened by passing its folder). ``open_batch_run`` does the
one-time checks - the placed ROI against ``place_roi``, exactly one ``*_cache`` folder - and
returns a ``BatchRun``. The placement is kept on ``run.placement``, so nothing places the ROI
again. The run hands out each part only when asked:

- ``edge_table()``: ``per_edge_morphometry.csv`` keyed by integer ``(u, v, key)``, without
  loading the graph, so a driver that needs only the table stays fast;
- ``numeric_column(column)``: one edge-table column as a finite float per edge, the one place
  a cell becomes a number;
- ``graph()``: the cached network with ``assigned_diameter_um`` joined on from the edge table,
  one-to-one on forward ``(u, v, key)``, with no reverse-direction fallback;
- ``skeleton()`` and ``vessel_mask()``: boolean, shape-checked against the placed ROI;
- ``th_probabilities()`` and ``th_mask()``: the TH channel cropped to the placed ROI.

Drivers ask for parts here instead of knowing folder names, globs, CSV keys or HDF5 channels,
so a reading rule fixed once is fixed for every H1 and H2 number.
"""
from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np

from . import cb_settings
from .io.load import read_ilastik_probabilities
# ROI_RECORD_NAME is defined by the module that writes the record; it is a batch-run file too.
from .roi_placement import ROI_RECORD_NAME, check_output_roi, place_roi  # noqa: F401
from .specimens import TH_CHANNEL

EDGE_TABLE_NAME = "per_edge_morphometry.csv"
GRAPH_NAME = "network_graph.pkl"
SKELETON_NAME = "skeleton.npy"
VESSEL_MASK_NAME = "vessel_mask.npy"
# The pipeline's VTK exports of the network and the mask, in the run folder.
VESSELS_VTP_NAME = "resistance_network_vessels.vtp"
NODES_VTP_NAME = "resistance_network_nodes.vtp"
VESSEL_MASK_VTI_NAME = "resistance_network_vessel_mask.vti"
DIAMETER = "assigned_diameter_um"


def open_batch_run(specimen, run_dir=None) -> "BatchRun":
    """Open one specimen's batch run, refusing it unless it was cut at the placed ROI.

    ``run_dir`` defaults to ``specimen.batch_run_dir``; pass a sensitivity run's folder to
    open that instead. The placed-ROI check comes first, before any other file is read.
    """
    run_dir = Path(specimen.batch_run_dir if run_dir is None else run_dir)
    placement = place_roi(specimen, cb_settings.ROI_VOXELS)
    check_output_roi(run_dir, specimen, cb_settings.ROI_VOXELS, placement)

    caches = sorted(path for path in run_dir.glob("*_cache") if path.is_dir())
    if len(caches) != 1:
        raise ValueError(
            f"{specimen.specimen_id}: {run_dir} holds {len(caches)} *_cache folders "
            f"({[path.name for path in caches]}); a batch run has exactly one. A stale or "
            f"duplicate cache would otherwise be picked up silently."
        )
    return BatchRun(specimen, run_dir, placement, caches[0])


def _finite_or_none(cell):
    """The cell as a finite float, or None if it is blank, not a number, NaN or infinite."""
    try:
        value = float(cell)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


class BatchRun:
    """An opened batch run. Build it with ``open_batch_run``, which does the checks."""

    def __init__(self, specimen, run_dir: Path, placement, cache_dir: Path):
        self.specimen = specimen
        self.run_dir = run_dir
        self.placement = placement
        self.cache_dir = cache_dir

    def _where(self) -> str:
        return f"{self.specimen.specimen_id} ({self.run_dir})"

    def edge_table(self) -> dict:
        """Every edge-table row, in file order, keyed by integer ``(u, v, key)``.

        Each row keeps every column as the file wrote it (strings). A duplicate key raises:
        two rows for one edge would leave which diameter it has to file order.
        """
        path = self.run_dir / EDGE_TABLE_NAME
        if not path.exists():
            raise FileNotFoundError(f"{self._where()}: no {EDGE_TABLE_NAME}.")
        table = {}
        duplicates = []
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                edge = (int(row["u"]), int(row["v"]), int(row["key"]))
                if edge in table:
                    duplicates.append(edge)
                table[edge] = row
        if duplicates:
            raise ValueError(
                f"{self._where()}: {EDGE_TABLE_NAME} has more than one row for edges "
                f"{duplicates[:10]}.")
        return table

    def numeric_column(self, column: str) -> dict:
        """One edge-table column as a float per edge, keyed and ordered as ``edge_table()``.

        The one place an edge-table cell becomes a number. A blank, text, NaN or infinite cell
        raises, naming the column and edges, rather than becoming NaN, 0 or a dropped row.
        """
        table = self.edge_table()
        if not table:
            raise ValueError(f"{self._where()}: {EDGE_TABLE_NAME} has no rows.")
        columns = next(iter(table.values())).keys()
        if column not in columns:
            raise KeyError(f"{self._where()}: {EDGE_TABLE_NAME} has no column {column!r}; it has "
                           f"{sorted(columns)}.")
        values = {edge: _finite_or_none(row[column]) for edge, row in table.items()}
        bad = [edge for edge, value in values.items() if value is None]
        if bad:
            raise ValueError(
                f"{self._where()}: {len(bad)} rows of {EDGE_TABLE_NAME} have an empty or "
                f"non-finite {column}: {bad[:10]}.")
        return values

    def graph(self):
        """The cached network, unpickled afresh, with ``assigned_diameter_um`` on every edge.

        The cached graph carries no calibre, so it comes from the edge table under a strict
        one-to-one join on forward integer ``(u, v, key)``: every edge has a row, every row has
        an edge, and every diameter is a finite number (read through ``numeric_column``). Any
        break raises rather than leaving an edge without a calibre or putting one on the wrong
        edge. A fresh copy each call, because the rheology solve writes onto the graph it is
        given.
        """
        with (self.cache_dir / GRAPH_NAME).open("rb") as handle:
            G = pickle.load(handle)
        diameters = self.numeric_column(DIAMETER)

        edges = {(int(u), int(v), int(k)) for u, v, k in G.edges(keys=True)}
        missing = [edge for edge in sorted(edges) if edge not in diameters]
        extra = [edge for edge in diameters if edge not in edges]
        problems = []
        if missing:
            problems.append(f"{len(missing)} graph edges have no edge-table row: {missing[:10]}")
        if extra:
            problems.append(f"{len(extra)} edge-table rows match no graph edge: {extra[:10]}")
        if problems:
            raise ValueError(
                f"{self._where()}: the graph and {EDGE_TABLE_NAME} do not match one-to-one on "
                f"forward (u, v, key); " + "; ".join(problems) + ".")

        for u, v, k, data in G.edges(keys=True, data=True):
            data[DIAMETER] = diameters[(int(u), int(v), int(k))]
        return G

    def skeleton(self) -> np.ndarray:
        """The cached skeleton as booleans, the shape of the placed ROI."""
        return self._roi_array(SKELETON_NAME)

    def vessel_mask(self) -> np.ndarray:
        """The cached vessel mask as booleans, the shape of the placed ROI."""
        return self._roi_array(VESSEL_MASK_NAME)

    def _roi_array(self, name: str) -> np.ndarray:
        array = np.load(self.cache_dir / name)
        expected = tuple(int(n) for n in self.placement.size_zyx)
        if array.shape != expected:
            raise ValueError(
                f"{self._where()}: {name} has shape {array.shape}, but the placed ROI is "
                f"{expected}. It would misalign with the TH channel cropped at that box.")
        return array.astype(bool, copy=False)

    def th_probabilities(self) -> np.ndarray:
        """The TH channel's glomus probability, float32, cropped to the placed ROI.

        Read through the guarded Ilastik reader with the registry's TH channel index, so it
        gets the same class-axis, index and shape checks as the vessel read, and only the box
        is read from disk. The exports are 8-bit, so a maximum above 1.5 is divided by 255.
        """
        block = read_ilastik_probabilities(
            self.specimen.th_probabilities_path,
            vessel_class_index=TH_CHANNEL.target_index,
            expected_shape_zyx=self.specimen.shape_zyx,
            check_calibration=False,
            crop_zyx=self.placement.bounds,
        )
        if block.max() > 1.5:
            block = block / 255.0
        return block

    def th_mask(self, threshold=None) -> np.ndarray:
        """Glomus voxels of the placed ROI: ``p > threshold``, default ``TH_THRESHOLD``."""
        threshold = cb_settings.TH_THRESHOLD if threshold is None else threshold
        # Strict on purpose. Open item 17 made the vessel cuts inclusive because the vessel
        # sweep's thresholds land exactly on the map's hundredths levels; TH_THRESHOLD was not
        # chosen on that sweep, so the TH cut kept '>' and every published TH number uses it.
        return self.th_probabilities() > threshold
