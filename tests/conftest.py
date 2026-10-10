"""Pytest fixtures and configuration."""
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for tests

import csv
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import numpy as np
import networkx as nx

# The pipeline script lives in examples/, which pyproject's pythonpath does not cover. Several
# test modules import it, and until now each did its own sys.path surgery inside whichever test
# happened to need it first - so tests placed before that one silently skipped instead of
# running. Done once here, so importorskip means "genuinely unavailable".
_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))


@pytest.fixture
def small_binary_3d():
    """Small 3D binary volume for testing."""
    arr = np.zeros((10, 10, 10), dtype=bool)
    arr[4:6, 4:6, :] = True  # small rod
    return arr


@pytest.fixture
def tiny_skeleton():
    """Tiny skeleton for graph tests (linear path)."""
    skel = np.zeros((8, 8, 8), dtype=bool)
    for i in range(2, 6):
        skel[i, 4, 4] = True
    return skel


@pytest.fixture
def simple_graph():
    """Simple graph with 3 nodes, 2 edges, positions."""
    G = nx.Graph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([1.0, 0.0, 0.0]))
    G.add_node(2, pos=np.array([2.0, 0.0, 0.0]))
    G.add_edge(0, 1, resistance=1.0, length=1.0, voxels=[(0, 0, 0), (1, 0, 0)])
    G.add_edge(1, 2, resistance=1.0, length=1.0, voxels=[(1, 0, 0), (2, 0, 0)])
    return G


@pytest.fixture
def multigraph_with_branch_order():
    """MultiGraph with branch_order on edges."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([5.0, 0.0, 0.0]))
    G.add_edge(0, 1, resistance=1.0, length=5.0, branch_order="BO1", voxels=[(0, 0, 0), (5, 0, 0)])
    return G


# --- A stand-in batch run ---------------------------------------------------------------------
# One factory for every test that needs a batch run. It writes a tiny but complete run to
# ``tmp_path`` and hands it back through the real ``open_batch_run``, so a change to the reader
# breaks the tests that use it straight away, not only the full suite.

SHAPE = (12, 10, 9)          # the specimen's full volume, z y x
SIZE = (4, 6, 5)             # the placed ROI
CENTRE = (6, 4, 5)           # bounds z 4:8, y 1:7, x 3:8
EDGES = [(0, 1, 0, "12.5"), (1, 2, 0, "8.0"), (1, 2, 1, "6.25"), (2, 3, 0, "10.0")]
EDGE_TABLE_COLUMNS = ["u", "v", "key", "length_um", "assigned_diameter_um", "diameter_provenance"]


def placement(specimen_id, centre=CENTRE):
    """The placed ROI the stand-in run is cut at (or a moved one, for a refusal test)."""
    from ImageLynx.roi_placement import RoiPlacement, centre_to_offsets
    return RoiPlacement(specimen_id=specimen_id, centre_zyx=centre, size_zyx=SIZE,
                        offsets_zyx=centre_to_offsets(centre, SHAPE), peak_slice=centre[0],
                        source="test")


def write_edge_table(run_dir, rows):
    """Overwrite a stand-in run's edge table with ``(u, v, key, diameter)`` rows."""
    from ImageLynx import batch_outputs
    with (Path(run_dir) / batch_outputs.EDGE_TABLE_NAME).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(EDGE_TABLE_COLUMNS)
        for u, v, key, diameter in rows:
            writer.writerow([u, v, key, f"{10.0 * (u + 1):g}", diameter, "measured_edt"])


def _write_columns(run_dir, G, columns):
    """Write ``G``'s edges as an edge table with exactly the columns given."""
    from ImageLynx import batch_outputs
    names = list(columns)
    edges = set(G.edges(keys=True))
    for name in names:
        extra = sorted(set(columns[name]) - edges)
        if extra:
            raise ValueError(f"stand-in run: {name} given for edges not in the graph: {extra}")
    with (Path(run_dir) / batch_outputs.EDGE_TABLE_NAME).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["u", "v", "key", *names])
        for u, v, key in G.edges(keys=True):
            cells = []
            for name in names:
                if (u, v, key) not in columns[name]:
                    raise ValueError(f"stand-in run: no {name} given for edge {(u, v, key)}")
                cells.append(columns[name][(u, v, key)])
            writer.writerow([u, v, key, *cells])


def _glomus_probabilities():
    """A deterministic 0-255 ramp over the whole volume, so the crop position is checkable."""
    return (np.arange(np.prod(SHAPE)) % 256).astype(np.uint8).reshape(SHAPE)


@dataclass
class StandInRun:
    """The files of one stand-in batch run, and ``run``: the real reader opened on them."""
    specimen: Any
    run_dir: Path
    cache: Path
    graph: nx.MultiGraph
    skeleton: np.ndarray
    placed: Any
    glomus: np.ndarray
    th_path: Path

    @property
    def run(self):
        from ImageLynx.batch_outputs import open_batch_run
        return open_batch_run(self.specimen)


@pytest.fixture
def make_batch_run(tmp_path, monkeypatch):
    """Factory for a complete, consistent batch run for a stand-in specimen."""
    h5py = pytest.importorskip("h5py")
    from ImageLynx import batch_outputs, cb_settings
    from ImageLynx.roi_placement import roi_record, write_roi_record

    def make(specimen_id="TEST-A", G=None, columns=None):
        """The standard four-edge run, or one built from ``G`` and ``columns``.

        ``columns`` maps each edge-table column name to ``{(u, v, key): value}`` for every
        edge of ``G``; a missing value, or a key that is not an edge of ``G``, raises, as the
        reader's strict join would.
        """
        if (G is None) != (columns is None):
            raise ValueError("pass G and columns together, or neither")
        run_dir = tmp_path / specimen_id
        cache = run_dir / "TEST_vessels_ilastik_Probabilities_cache"
        cache.mkdir(parents=True)

        if G is None:
            G = nx.MultiGraph()
            for u, v, key, _ in EDGES:
                G.add_edge(u, v, key=key, length=10.0 * (u + 1))
            write_edge_table(run_dir, EDGES)
        else:
            _write_columns(run_dir, G, columns)
        with (cache / batch_outputs.GRAPH_NAME).open("wb") as handle:
            pickle.dump(G, handle)

        skeleton = np.zeros(SIZE, dtype=np.uint8)
        skeleton[2, 3, :] = 1
        np.save(cache / batch_outputs.SKELETON_NAME, skeleton)
        np.save(cache / batch_outputs.VESSEL_MASK_NAME, np.ones(SIZE, dtype=bool))

        th_path = tmp_path / f"{specimen_id}_TH_ilastik_Probabilities.h5"
        glomus = _glomus_probabilities()
        with h5py.File(th_path, "w") as handle:
            handle.create_dataset("exported_data", data=np.stack([glomus, 255 - glomus], axis=-1))

        specimen = SimpleNamespace(specimen_id=specimen_id, shape_zyx=SHAPE,
                                   batch_run_dir=run_dir, th_probabilities_path=th_path)
        placed = placement(specimen_id)
        monkeypatch.setattr(batch_outputs, "place_roi",
                            lambda s, size: placed if tuple(size) == tuple(cb_settings.ROI_VOXELS)
                            else pytest.fail(f"place_roi asked for {size}, not the frozen ROI"))
        write_roi_record(run_dir, roi_record(placed, SHAPE, centred=False))
        return StandInRun(specimen=specimen, run_dir=run_dir, cache=cache, graph=G,
                          skeleton=skeleton, placed=placed, glomus=glomus, th_path=th_path)

    return make


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "plotting: marks tests that create matplotlib figures"
    )
