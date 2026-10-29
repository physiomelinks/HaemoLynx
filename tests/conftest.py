"""Pytest fixtures and configuration."""
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for tests

import csv
import pickle
import subprocess
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

# scripts/ is not a package either; tests import its shared git helper, _worktree, from here.
_SCRIPTS = _EXAMPLES.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.append(str(_SCRIPTS))

import _worktree  # noqa: E402

# For the toy git repos the script tests build: no GIT_* variable a hook sets, and no
# COV_CORE_*/COVERAGE_* variable a coverage run sets, may reach a toy's git or pytest.
TOY_ENV = _worktree.clean_env("COV_CORE_", "COVERAGE_")


def toy_git(repo, *args):
    """git's stdout in toy repo `repo`, with a fixed author; raises on a non-zero exit."""
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo,
                          env=TOY_ENV, check=True, capture_output=True, text=True).stdout


def realdata_test(package):
    """A toy repo's test file that reads gitignored ``examples/images/sum.txt`` through a link.

    It imports ``add`` from ``<package>.calc``, is marked ``realdata``, and skips where the data
    is missing. Its folder must stay a real one (a tracked file keeps it in every worktree).
    """
    return f'''\
from pathlib import Path

import pytest

from {package}.calc import add

DATA = Path(__file__).resolve().parents[1] / "examples" / "images" / "sum.txt"


@pytest.mark.realdata
@pytest.mark.skipif(not DATA.exists(), reason="needs the gitignored data")
def test_add_matches_the_data():
    assert DATA.parent.is_dir() and not DATA.parent.is_symlink()  # a tracked file kept it real
    assert DATA.is_symlink(), "the data was copied, not linked"
    assert add(2, 3) == int(DATA.read_text())
'''


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


def cross_network(diameter=None, length=None, drop=None):
    """Centre node joined to one terminal on each of the six ROI faces, in micrometres.

    Node 7 hangs off the centre through a parallel pair, so the MultiGraph path is exercised,
    and node 8 is an interior dead end off node 7, on no face. ``drop`` removes one node
    before calibre and length are set. ``diameter`` and ``length`` are each
    ``f(n, (u, v, key))`` over the edges in order: ``diameter`` sets ``assigned_diameter_um`` on the edge, and ``length`` fills the returned
    ``length_um`` column, as ``BatchRun.numeric_column`` gives it (empty when not given).
    Returns ``(G, lengths)``.
    """
    G = nx.MultiGraph()
    mid, top = 148.0, 296.0
    G.add_node(0, pos=np.array([mid, mid, mid]))
    faces = [(0.0, mid, mid), (top, mid, mid), (mid, 0.0, mid),
             (mid, top, mid), (mid, mid, 0.0), (mid, mid, top)]
    for i, pos in enumerate(faces, start=1):
        G.add_node(i, pos=np.array(pos))
        G.add_edge(0, i)
    G.add_node(7, pos=np.array([mid, 60.0, 100.0]))
    G.add_node(8, pos=np.array([mid, 30.0, 100.0]))
    G.add_edge(0, 7)
    G.add_edge(0, 7)
    G.add_edge(7, 8)
    if drop is not None:
        G.remove_node(drop)
    lengths = {}
    for n, edge in enumerate(G.edges(keys=True)):
        if diameter is not None:
            G.edges[edge]["assigned_diameter_um"] = diameter(n, edge)
        if length is not None:
            lengths[edge] = length(n, edge)
    return G, lengths


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

    placements = {}

    def stub_place_roi(specimen, size):
        if tuple(size) != tuple(cb_settings.ROI_VOXELS):
            pytest.fail(f"place_roi asked for {size}, not the frozen ROI")
        return placements[specimen.specimen_id]

    monkeypatch.setattr(batch_outputs, "place_roi", stub_place_roi)

    def make(specimen_id="TEST-A", G=None, columns=None, run_dir=None):
        """The standard four-edge run, or one built from ``G`` and ``columns``.

        ``columns`` maps each edge-table column name to ``{(u, v, key): value}`` for every
        edge of ``G``; a missing value, or a key that is not an edge of ``G``, raises, as the
        reader's strict join would. ``run_dir`` defaults to ``tmp_path/<specimen_id>``; pass
        one to write the same specimen under more than one root. Each specimen made is placed
        at its own ROI, so runs of several specimens open side by side.
        """
        if (G is None) != (columns is None):
            raise ValueError("pass G and columns together, or neither")
        run_dir = tmp_path / specimen_id if run_dir is None else Path(run_dir)
        cache = run_dir / f"TEST_vessels_ilastik_Probabilities{batch_outputs.CACHE_SUFFIX}"
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
        placed = placements[specimen_id] = placement(specimen_id)
        write_roi_record(run_dir, roi_record(placed, SHAPE, centred=False))
        return StandInRun(specimen=specimen, run_dir=run_dir, cache=cache, graph=G,
                          skeleton=skeleton, placed=placed, glomus=glomus, th_path=th_path)

    return make


@pytest.fixture
def stub_h1_vtk_builders(monkeypatch):
    """Stub ``cb_h1_vtk``'s specimen list, ``stamp`` and its four builders, for ``main()`` tests.

    Returns ``stub(vessels=None, nodes=None, skeleton=None, surface=None, grid=None,
    events=None)``. Each builder returns what it is given (``build_surface`` returns
    ``(surface, grid)``), or a placeholder mesh for a part left as None: a builder never
    returns None, since a missing source file raises. The placeholder grid is a
    2x2x2 all-vessel mask, so ``verify`` runs on the placeholders. With ``events`` a list, each builder also appends its name
    (``"vessels"``, ``"nodes"``, ``"skeleton"``, ``"surface"``) when it is called. The one
    specimen is ``WKY-A``; ``stamp`` returns the mesh it is given. The test still sets
    ``OUTPUT``, ``open_batch_run`` and ``sys.argv`` itself.
    """
    import cb_h1_vtk
    pv = pytest.importorskip("pyvista")

    def stub(*, vessels=None, nodes=None, skeleton=None, surface=None, grid=None, events=None):
        def point():
            return pv.PolyData(np.zeros((1, 3)))

        if grid is None:
            grid = pv.ImageData(dimensions=(2, 2, 2))
            grid["vessel_mask"] = np.ones(grid.n_points, dtype=np.uint8)
        vessels = pv.Line((0, 0, 0), (1, 1, 1)) if vessels is None else vessels
        nodes, skeleton, surface = (point() if part is None else part
                                    for part in (nodes, skeleton, surface))

        def builder(name, value):
            def build(*args):
                if events is not None:
                    events.append(name)
                return value
            return build

        monkeypatch.setattr(cb_h1_vtk, "SPECIMENS",
                            [SimpleNamespace(specimen_id="WKY-A", group="WKY")])
        monkeypatch.setattr(cb_h1_vtk, "enrich_vessels", builder("vessels", vessels))
        monkeypatch.setattr(cb_h1_vtk, "build_nodes", builder("nodes", nodes))
        monkeypatch.setattr(cb_h1_vtk, "build_skeleton", builder("skeleton", skeleton))
        monkeypatch.setattr(cb_h1_vtk, "build_surface", builder("surface", (surface, grid)))
        monkeypatch.setattr(cb_h1_vtk, "stamp", lambda mesh, specimen: mesh)
        return cb_h1_vtk

    return stub


def write_ilp_project(path, lanes, label_names=("vessel", "background"), compute_in_2d=None):
    """Build a minimal Ilastik ``.ilp``: label names, one raw-data path per lane, label blocks.

    ``lanes`` is a sequence of ``(raw_data_path, blocks)``. ``blocks`` is a sequence of
    ``(z, labels_yx)`` pairs, each a 2D uint8 array written as one block at depth ``z``
    (0 is unlabelled). An empty ``blocks`` registers the lane without labelling it. The raw
    path is written as ``<raw_data_path>/data``. With ``compute_in_2d`` given, the feature
    selection is written too, every one of its seven features at that value.
    """
    import h5py

    with h5py.File(path, "w") as project:
        project["PixelClassification/LabelNames"] = np.array(
            [n.encode() for n in label_names], dtype="S32")
        for position, (raw_data_path, blocks) in enumerate(lanes):
            lane = f"lane{position:04d}"
            project[f"Input Data/infos/{lane}/Raw Data/filePath"] = \
                f"{raw_data_path}/data".encode()
            group = project.create_group(f"PixelClassification/LabelSets/labels{position:03d}")
            for index, (z, labels_yx) in enumerate(blocks):
                ny, nx = labels_yx.shape
                block = group.create_dataset(
                    f"block{index:04d}",
                    data=np.asarray(labels_yx, dtype=np.uint8).reshape(1, ny, nx, 1))
                block.attrs["blockSlice"] = f"[{z}:{z + 1},0:{ny},0:{nx},0:1]".encode()
        if compute_in_2d is not None:
            project["FeatureSelections/ComputeIn2d"] = np.array([compute_in_2d] * 7)
            project["FeatureSelections/SelectionMatrix"] = np.zeros((6, 7), dtype=bool)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "plotting: marks tests that create matplotlib figures"
    )
