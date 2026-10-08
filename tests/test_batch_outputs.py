"""The one batch-run reader every CB driver opens a batch run through.

Each test builds a tiny batch run in ``tmp_path`` - a four-edge graph in one ``*_cache``
folder, its edge table, a ``roi_placement.json``, skeleton and vessel-mask arrays and an 8-bit
TH export - and goes through ``open_batch_run`` and the parts it hands out, nothing else.
``place_roi`` is replaced by a fixed small box, since the real one reads the specimen's data.
"""
import csv
import pickle
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

import ImageLynx.batch_outputs as batch_outputs
from ImageLynx import cb_settings
from ImageLynx.batch_outputs import open_batch_run
from ImageLynx.roi_placement import (
    RoiPlacement, centre_to_offsets, roi_record, write_roi_record,
)

h5py = pytest.importorskip("h5py")

SHAPE = (12, 10, 9)          # the specimen's full volume, z y x
SIZE = (4, 6, 5)             # the placed ROI
CENTRE = (6, 4, 5)           # bounds z 4:8, y 1:7, x 3:8
EDGES = [(0, 1, 0, "12.5"), (1, 2, 0, "8.0"), (1, 2, 1, "6.25"), (2, 3, 0, "10.0")]
COLUMNS = ["u", "v", "key", "length_um", "assigned_diameter_um", "diameter_provenance"]


def _placement(specimen_id, centre=CENTRE):
    return RoiPlacement(specimen_id=specimen_id, centre_zyx=centre, size_zyx=SIZE,
                        offsets_zyx=centre_to_offsets(centre, SHAPE), peak_slice=centre[0],
                        source="test")


def _write_edge_table(run_dir, rows):
    with (run_dir / "per_edge_morphometry.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for u, v, key, diameter in rows:
            writer.writerow([u, v, key, f"{10.0 * (u + 1):g}", diameter, "measured_edt"])


def _glomus_probabilities():
    """A deterministic 0-255 ramp over the whole volume, so the crop position is checkable."""
    return (np.arange(np.prod(SHAPE)) % 256).astype(np.uint8).reshape(SHAPE)


@pytest.fixture
def batch_run(tmp_path, monkeypatch):
    """A complete, consistent batch run for a stand-in specimen."""
    run_dir = tmp_path / "TEST-A"
    cache = run_dir / "TEST_vessels_ilastik_Probabilities_cache"
    cache.mkdir(parents=True)

    G = nx.MultiGraph()
    for u, v, key, _ in EDGES:
        G.add_edge(u, v, key=key, length=10.0 * (u + 1))
    with (cache / "network_graph.pkl").open("wb") as handle:
        pickle.dump(G, handle)
    _write_edge_table(run_dir, EDGES)

    skeleton = np.zeros(SIZE, dtype=np.uint8)
    skeleton[2, 3, :] = 1
    np.save(cache / "skeleton.npy", skeleton)
    np.save(cache / "vessel_mask.npy", np.ones(SIZE, dtype=bool))

    th_path = tmp_path / "TEST_TH_ilastik_Probabilities.h5"
    glomus = _glomus_probabilities()
    with h5py.File(th_path, "w") as handle:
        handle.create_dataset("exported_data", data=np.stack([glomus, 255 - glomus], axis=-1))

    specimen = SimpleNamespace(specimen_id="TEST-A", shape_zyx=SHAPE, batch_run_dir=run_dir,
                               th_probabilities_path=th_path)
    placed = _placement(specimen.specimen_id)
    monkeypatch.setattr(batch_outputs, "place_roi",
                        lambda s, size: placed if tuple(size) == tuple(cb_settings.ROI_VOXELS)
                        else pytest.fail(f"place_roi asked for {size}, not the frozen ROI"))
    write_roi_record(run_dir, roi_record(placed, SHAPE, centred=False))
    return SimpleNamespace(specimen=specimen, run_dir=run_dir, cache=cache, graph=G,
                           skeleton=skeleton, placed=placed, glomus=glomus, th_path=th_path)


# --- Opening ---------------------------------------------------------------------------------

def test_opening_finds_the_run_its_placement_and_its_one_cache(batch_run):
    run = open_batch_run(batch_run.specimen)
    assert run.run_dir == batch_run.run_dir
    assert run.cache_dir == batch_run.cache
    assert run.placement == batch_run.placed


def test_the_edge_table_is_keyed_by_integer_edge_in_file_order(batch_run):
    table = open_batch_run(batch_run.specimen).edge_table()
    assert list(table) == [(0, 1, 0), (1, 2, 0), (1, 2, 1), (2, 3, 0)]
    # Every column is kept, as the file wrote it.
    assert table[(1, 2, 1)] == {"u": "1", "v": "2", "key": "1", "length_um": "20",
                                "assigned_diameter_um": "6.25",
                                "diameter_provenance": "measured_edt"}


def test_a_run_cut_anywhere_but_the_placed_roi_is_refused_before_anything_else(batch_run):
    """The ROI error wins even when the cache layout is also broken and the table is garbage."""
    moved = _placement(batch_run.specimen.specimen_id, centre=(7, 4, 5))
    write_roi_record(batch_run.run_dir, roi_record(moved, SHAPE, centred=False))
    (batch_run.run_dir / "second_cache").mkdir()
    (batch_run.run_dir / "per_edge_morphometry.csv").write_text("not,a\nvalid table")

    with pytest.raises(ValueError, match="place_roi now gives"):
        open_batch_run(batch_run.specimen)


def test_a_run_with_no_cache_folder_is_refused(batch_run):
    batch_run.cache.rename(batch_run.run_dir / "moved_away")
    with pytest.raises(ValueError, match="0 \\*_cache folders"):
        open_batch_run(batch_run.specimen)


def test_a_run_with_two_cache_folders_is_refused(batch_run):
    (batch_run.run_dir / "stale_cache").mkdir()
    with pytest.raises(ValueError, match="2 \\*_cache folders"):
        open_batch_run(batch_run.specimen)


def test_a_sensitivity_run_is_opened_by_passing_its_folder(batch_run, tmp_path):
    elsewhere = tmp_path / "t0.93" / "TEST-A"
    elsewhere.parent.mkdir()
    batch_run.run_dir.rename(elsewhere)
    run = open_batch_run(batch_run.specimen, run_dir=elsewhere)
    assert run.run_dir == elsewhere
    assert list(run.edge_table())[0] == (0, 1, 0)


def test_a_duplicate_edge_table_row_is_refused(batch_run):
    _write_edge_table(batch_run.run_dir, EDGES + [(1, 2, 1, "6.5")])
    with pytest.raises(ValueError, match=r"TEST-A.*more than one row.*\(1, 2, 1\)"):
        open_batch_run(batch_run.specimen).edge_table()


# --- Numeric columns -------------------------------------------------------------------------

def test_a_numeric_column_is_a_float_per_edge_in_file_order(batch_run):
    lengths = open_batch_run(batch_run.specimen).numeric_column("length_um")
    assert lengths == {(0, 1, 0): 10.0, (1, 2, 0): 20.0, (1, 2, 1): 20.0, (2, 3, 0): 30.0}
    assert list(lengths) == [(0, 1, 0), (1, 2, 0), (1, 2, 1), (2, 3, 0)]
    assert all(type(value) is float for value in lengths.values())


@pytest.mark.parametrize("cell", ["", "nan", "None", "inf", "-inf"],
                         ids=["blank", "nan", "text", "inf", "-inf"])
def test_a_cell_that_is_not_a_finite_number_is_refused_not_turned_into_nan(batch_run, cell):
    _write_edge_table(batch_run.run_dir, [EDGES[0], (1, 2, 0, cell), (1, 2, 1, cell), EDGES[3]])
    with pytest.raises(ValueError, match=r"2 rows .*empty or non-finite assigned_diameter_um"
                                         r".*\(1, 2, 0\), \(1, 2, 1\)") as raised:
        open_batch_run(batch_run.specimen).numeric_column("assigned_diameter_um")
    assert "TEST-A" in str(raised.value) and str(batch_run.run_dir) in str(raised.value)


def test_an_edge_table_with_no_rows_is_refused(batch_run):
    _write_edge_table(batch_run.run_dir, [])
    with pytest.raises(ValueError, match=r"TEST-A.*per_edge_morphometry.csv has no rows"):
        open_batch_run(batch_run.specimen).numeric_column("length_um")


def test_a_column_the_edge_table_lacks_is_refused(batch_run):
    with pytest.raises(KeyError, match=r"TEST-A.*no column 'fwhm_diameter_um'.*length_um"):
        open_batch_run(batch_run.specimen).numeric_column("fwhm_diameter_um")


# --- The graph and its diameters -------------------------------------------------------------

def test_the_graph_carries_each_rows_diameter_on_its_own_edge(batch_run):
    G = open_batch_run(batch_run.specimen).graph()
    got = {(u, v, k): d["assigned_diameter_um"] for u, v, k, d in G.edges(keys=True, data=True)}
    # Parallel edges (1, 2, 0) and (1, 2, 1) keep their own diameters.
    assert got == {(0, 1, 0): 12.5, (1, 2, 0): 8.0, (1, 2, 1): 6.25, (2, 3, 0): 10.0}
    assert G.edges[1, 2, 0]["length"] == 20.0     # the cached attributes are untouched


def test_each_graph_is_a_fresh_copy(batch_run):
    """The rheology solve writes onto the graph, so a second solve must not see the first's."""
    run = open_batch_run(batch_run.specimen)
    first = run.graph()
    first.edges[0, 1, 0]["flow_abs"] = 1.0
    assert "flow_abs" not in run.graph().edges[0, 1, 0]


@pytest.mark.parametrize("rows, message", [
    (EDGES[:-1], r"no edge-table row.*\(2, 3, 0\)"),
    (EDGES + [(3, 4, 0, "5.0")], r"no graph edge.*\(3, 4, 0\)"),
    ([(1, 0, 0, "12.5")] + EDGES[1:], r"no edge-table row.*\(0, 1, 0\)"),
    ([EDGES[0], (1, 2, 0, ""), *EDGES[2:]],
     r"empty or non-finite assigned_diameter_um.*\(1, 2, 0\)"),
    ([EDGES[0], (1, 2, 0, "nan"), *EDGES[2:]],
     r"empty or non-finite assigned_diameter_um.*\(1, 2, 0\)"),
    ([EDGES[0], (1, 2, 0, "None"), *EDGES[2:]],
     r"empty or non-finite assigned_diameter_um.*\(1, 2, 0\)"),
    ([EDGES[0], (1, 2, 0, "inf"), *EDGES[2:]],
     r"empty or non-finite assigned_diameter_um.*\(1, 2, 0\)"),
], ids=["missing row", "extra row", "reversed row", "empty diameter", "nan diameter",
        "text diameter", "infinite diameter"])
def test_the_join_is_strictly_one_to_one(batch_run, rows, message):
    """Forward (u, v, key) only: a row written the other way round matches nothing."""
    _write_edge_table(batch_run.run_dir, rows)
    with pytest.raises(ValueError, match=message) as raised:
        open_batch_run(batch_run.specimen).graph()
    assert "TEST-A" in str(raised.value) and str(batch_run.run_dir) in str(raised.value)


def test_the_edge_table_and_masks_never_unpickle_the_graph(batch_run):
    (batch_run.cache / "network_graph.pkl").write_bytes(b"not a pickle")
    run = open_batch_run(batch_run.specimen)
    run.edge_table()
    run.skeleton()
    run.vessel_mask()
    with pytest.raises(pickle.UnpicklingError):
        run.graph()


# --- ROI arrays ------------------------------------------------------------------------------

def test_the_skeleton_and_vessel_mask_come_back_boolean(batch_run):
    run = open_batch_run(batch_run.specimen)
    skeleton = run.skeleton()
    assert skeleton.dtype == bool and skeleton.sum() == SIZE[2]
    assert np.array_equal(skeleton, batch_run.skeleton.astype(bool))
    assert run.vessel_mask().dtype == bool and run.vessel_mask().all()


@pytest.mark.parametrize("part", ["skeleton", "vessel_mask"])
def test_an_array_not_the_shape_of_the_placed_roi_is_refused(batch_run, part):
    np.save(batch_run.cache / f"{part}.npy", np.zeros((4, 6, 6), dtype=bool))
    with pytest.raises(ValueError, match=r"TEST-A.*\(4, 6, 6\).*\(4, 6, 5\)"):
        getattr(open_batch_run(batch_run.specimen), part)()


# --- The TH channel --------------------------------------------------------------------------

def test_th_probabilities_are_the_glomus_channel_at_the_placed_roi_rescaled(batch_run):
    """Channel 0 (glomus, from the registry), cut at the placed box, 8-bit divided by 255."""
    got = open_batch_run(batch_run.specimen).th_probabilities()
    expected = batch_run.glomus[4:8, 1:7, 3:8].astype(np.float32) / 255.0
    assert got.dtype == np.float32
    assert np.array_equal(got, expected)


def test_the_th_channel_index_comes_from_the_registry(batch_run, monkeypatch):
    """Not a literal 0: with the registry saying channel 1, channel 1 is what is read."""
    glomus = batch_run.glomus
    with h5py.File(batch_run.th_path, "w") as handle:
        handle.create_dataset("exported_data", data=np.stack([255 - glomus, glomus], axis=-1))
    monkeypatch.setattr(batch_outputs, "TH_CHANNEL", SimpleNamespace(target_index=1))
    got = open_batch_run(batch_run.specimen).th_probabilities()
    assert np.array_equal(got, glomus[4:8, 1:7, 3:8].astype(np.float32) / 255.0)


def test_a_float_th_export_is_not_rescaled(batch_run):
    """The 255 rescale only fires above 1.5, so a [0, 1] export passes through as it is."""
    glomus = batch_run.glomus.astype(np.float32) / 255.0
    with h5py.File(batch_run.th_path, "w") as handle:
        handle.create_dataset("exported_data", data=np.stack([glomus, 1 - glomus], axis=-1))
    got = open_batch_run(batch_run.specimen).th_probabilities()
    assert np.array_equal(got, glomus[4:8, 1:7, 3:8])


def _th_export_with_one_value(batch_run, value):
    glomus = np.zeros(SHAPE, dtype=np.float32)
    glomus[5, 2, 4] = value                       # inside the placed box, at (1, 1, 1) of it
    glomus[5, 2, 5] = 0.9
    with h5py.File(batch_run.th_path, "w") as handle:
        handle.create_dataset("exported_data", data=np.stack([glomus, 1 - glomus], axis=-1))


def test_the_th_mask_is_a_strict_cut_at_th_threshold(batch_run):
    """A voxel at exactly TH_THRESHOLD is outside the mask; one just above is inside."""
    t = np.float32(cb_settings.TH_THRESHOLD)
    _th_export_with_one_value(batch_run, t)
    run = open_batch_run(batch_run.specimen)
    mask = run.th_mask()
    assert mask.dtype == bool
    assert not mask[1, 1, 1] and mask[1, 1, 2] and mask.sum() == 1

    _th_export_with_one_value(batch_run, np.nextafter(t, np.float32(1)))
    assert open_batch_run(batch_run.specimen).th_mask()[1, 1, 1]


def test_the_th_mask_takes_a_threshold_override(batch_run):
    _th_export_with_one_value(batch_run, 0.3)
    run = open_batch_run(batch_run.specimen)
    assert run.th_mask(threshold=0.25).sum() == 2
    assert run.th_mask(threshold=0.3).sum() == 1


# --- Where batch runs live -------------------------------------------------------------------

def test_each_specimens_batch_run_dir_is_where_the_batch_driver_writes_it():
    """The batch driver takes its output root from the registry, so the path is set once."""
    import cb_h1_batch
    from ImageLynx.specimens import BATCH_RUN_ROOT, SPECIMENS

    assert cb_h1_batch.OUTPUT_DIR is BATCH_RUN_ROOT
    for specimen in SPECIMENS:
        assert specimen.batch_run_dir == cb_h1_batch.OUTPUT_DIR / specimen.specimen_id


def test_the_batch_driver_does_not_spell_out_the_batch_run_root():
    import cb_h1_batch
    from pathlib import Path

    source = Path(cb_h1_batch.__file__).read_text()
    assert '"cb_h1_batch"' not in source
