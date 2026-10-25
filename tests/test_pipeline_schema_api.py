"""The pipeline's schema is part of the installed package, not example code.

`resolve_settings` cannot be called without a schema, so while the schema lived
in `examples/resistance_pipeline_schema.py` a pip-installed copy of HaemoLynx
had no way to configure a run at all. These tests pin the public route in --
`default_schema()` and `write_default_config()` -- and, in a subprocess with
only the package importable, prove it does not reach back into the repository.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from haemolynx.parsers import (  # noqa: E402
    ConfigError,
    Schema,
    load_config,
    parse_prerequisite,
    prerequisite_name,
)
from haemolynx.pipeline import default_schema, write_default_config  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"


# --- the schema itself -----------------------------------------------------


def test_default_schema_is_the_pipelines_settings():
    schema = default_schema()
    assert isinstance(schema, Schema)
    for name in (
        "input_path",
        "voxel_size_override_xyz",
        "inlet_p_bc",
        "vtk_output_prefix",
        "cut_network_at_large_vessel_volumes",
    ):
        assert name in schema, f"{name} missing from the pipeline schema"
    assert "cut_large_vessel_sample_densely" not in schema
    assert schema.title


def test_default_schema_is_stable_between_calls():
    """A GUI and a run must be configuring the same settings."""
    assert default_schema().names == default_schema().names


def test_the_schema_module_is_importable_without_the_examples_directory():
    """`import haemolynx.pipeline.schema` must not need a repository checkout."""
    import haemolynx.pipeline.schema as module

    assert Path(module.__file__).is_relative_to(SRC)
    assert module.SCHEMA is default_schema()


def test_vascular_community_weighting_choices_match_the_graph_module():
    """The schema's own choices tuple is a literal, matching every other
    `choice` setting's convention -- this pins it against
    `graph.communities.COMMUNITY_WEIGHTINGS` so the two cannot silently
    drift apart."""
    from haemolynx.graph.communities import COMMUNITY_WEIGHTINGS

    schema = default_schema()
    setting = schema["vascular_community_weighting"]
    assert setting.choices == COMMUNITY_WEIGHTINGS
    assert setting.default == "topology"
    assert schema["compute_vascular_communities"].default is False


def test_use_memmap_loading_defaults_off_and_lives_under_input_and_segmentation():
    """An existing config/run must not change behaviour just from upgrading
    -- the low-memory path is opt-in."""
    schema = default_schema()
    setting = schema["use_memmap_loading"]
    assert setting.default is False
    assert setting.kind == "bool"
    assert setting.section == "Input and segmentation"
    # An image run's setting: a synthetic network has no image to memory-map.
    assert setting.requires == ("network_source=segmented_image",)


def test_memmap_directory_defaults_unset_and_requires_use_memmap_loading():
    """An unset directory means 'use the OS temp directory' -- the setting
    only matters once use_memmap_loading is actually on."""
    schema = default_schema()
    setting = schema["memmap_directory"]
    assert setting.default is None
    assert setting.kind == "path"
    assert setting.section == "Input and segmentation"
    assert setting.requires == ("use_memmap_loading",)


def test_skeletonize_tiling_settings_default_off_and_chain_their_requires():
    """skeletonize_tile_large_components requires use_memmap_loading (the
    residual-case fallback only exists once memmap loading is already on);
    its own two children require it in turn, matching the same
    master-toggle-plus-children pattern use_thick_vessel_skeletonisation's
    own settings already use in this section."""
    schema = default_schema()

    toggle = schema["skeletonize_tile_large_components"]
    assert toggle.default is False
    assert toggle.kind == "bool"
    assert toggle.section == "Pipeline stages"
    assert toggle.requires == ("use_memmap_loading",)

    max_voxels = schema["skeletonize_tile_max_voxels"]
    assert max_voxels.default == 200_000_000
    assert max_voxels.kind == "int"
    assert max_voxels.requires == ("skeletonize_tile_large_components",)

    halo = schema["skeletonize_tile_halo_um"]
    assert halo.default == 30.0
    assert halo.kind == "float"
    assert halo.requires == ("skeletonize_tile_large_components",)


def test_all_diams_const_children_are_declared_immediately_after_it():
    """gui/_widget.py's Diameters tab is a flat Container in schema
    declaration order with no QGroupBox -- nesting a child under its parent
    checkbox is purely "declared right after it", so a child declared pages
    away (behind ~20 unrelated pericyte settings, as a regression once put
    them) reads as un-nested, or as having disappeared, even though
    `requires` still gates it correctly. This pins the six per-order
    diameter overrides directly after `all_diams_const` with nothing
    unrelated between them."""
    schema = default_schema()
    names = schema.names
    start = names.index("all_diams_const")
    children = (
        "manual_capillary_diameter_by_branch_order",
        "manual_arteriole_diameter_by_branch_order",
        "manual_venule_diameter_by_branch_order",
        "manual_large_arteriole_diameter_by_branch_order",
        "manual_large_venule_diameter_by_branch_order",
        "diameter_by_branch_order",
    )
    assert names[start + 1 : start + 1 + len(children)] == children
    for name in children:
        assert schema[name].requires == ("!all_diams_const",), name


def test_measurement_3d_to_cell_mask_is_declared_right_before_statistics():
    """On "9. Additional measurements" the 3D object-mask distance comes
    first, above Statistics: measurement_3d_to_cell_mask and its six
    children are one contiguous block (so the flat-Container nesting keeps
    the children under their toggle) ending right before `statistics`."""
    schema = default_schema()
    names = schema.names
    end = names.index("statistics")
    children = (
        "measurement_3d_to_cell_mask",
        "cell_mask_path",
        "cell_mask_h5_dataset_name",
        "measurement_3d_vessel_mask_path",
        "measurement_3d_vessel_mask_h5_dataset_name",
    )
    assert names[end - len(children) : end] == children


def test_no_path_default_points_inside_the_examples_directory():
    """Library defaults describe the user's working directory, not this repo."""
    inside_repo = [
        setting.name
        for setting in default_schema()
        if setting.kind == "path"
        and setting.default is not None
        and str(setting.default).startswith("examples/")
    ]
    assert not inside_repo, (
        "path defaults still point at this repository's examples/ tree: "
        f"{inside_repo}"
    )


# --- nested/gated settings must not be silently read while hidden ----------


def _non_default_probe_value(setting):
    """A value that differs from *setting*'s default, generic across kinds.

    Prefers a declared bound (minimum/maximum) over an arbitrary offset, so
    it stays inside a narrow range (a 0.0-1.0 fraction, say) instead of
    guessing a step size that happens to land out of bounds. Returns None
    when no value can be constructed generically (mapping/any/list kinds),
    which the caller skips rather than guesses at.
    """
    default = setting.default
    kind = setting.kind
    if kind == "bool":
        return not bool(default)
    if kind == "choice":
        for choice in setting.choices or ():
            if choice != default:
                return choice
        return None
    if kind in ("int", "float"):
        for bound in (setting.minimum, setting.maximum):
            if bound is not None and bound != default:
                return bound
        base = default if default is not None else 0
        return base + (1 if kind == "int" else 1.0)
    if kind in ("str", "path"):
        base = "" if default is None else str(default)
        return base + "_probe"
    return None


def _value_meeting(prerequisite: str) -> tuple[str, object]:
    """One ``(setting_name, value)`` pair under which *prerequisite* holds."""
    schema = default_schema()
    name, test, expected = parse_prerequisite(prerequisite)
    if test in ("equals", "not_equals"):
        choices = schema[name].choices or ()
        return name, expected if test == "equals" else next(c for c in choices if c != expected)
    if test == "off":
        return name, None if schema[name].kind == "path" else False
    return name, "_probe" if schema[name].kind == "path" else True


def _first_unmet_prerequisite(setting) -> dict[str, object]:
    """Values that make *setting*'s first prerequisite unmet -- ``all()``
    over requires means unmeeting just one is enough to make the whole
    setting ineffective. A ``!gate`` rule is unmet only while the gate is
    *in effect*, so the gate's own prerequisites are met alongside it."""
    prerequisite = setting.requires[0]
    name, test, expected = parse_prerequisite(prerequisite)
    if test == "equals":
        choices = default_schema()[name].choices or ()
        other = next((c for c in choices if c != expected), expected)
        return {name: other}
    if test == "not_equals":
        return {name: expected}
    if test == "off":
        values = dict(_value_meeting(rule) for rule in default_schema()[name].requires)
        values[name] = True
        return values
    return {name: None if default_schema()[name].kind == "path" else False}


def test_every_requires_prerequisite_names_a_real_setting():
    """A typo'd or renamed prerequisite is invisible to the ineffectiveness
    check below: ``is_prerequisite_met`` reads an unknown key as `False` via
    a plain dict `.get`, the same as a real setting deliberately turned off,
    so a stale `requires` entry would still "work" there by accident. This
    is the one check that actually looks at the name itself.
    """
    schema = default_schema()
    names = set(schema.names)
    bad = [
        (setting.name, prerequisite)
        for setting in schema
        for prerequisite in (setting.requires or ())
        if prerequisite_name(prerequisite) not in names
    ]
    assert bad == []


def test_every_gated_setting_is_ineffective_when_its_prerequisite_is_off():
    """Nested options must not silently be read once their prerequisite is
    off -- the GUI hides or greys them out on exactly that condition, and
    Schema.ineffective_settings is the general mechanism that is supposed to
    catch every such setting, not just the one hand-picked case in
    test_parsers_schema.py. This walks the real pipeline schema's 180+ gated
    settings and proves the claim for each one a probe value can be built
    for, so a `requires` tuple that stops matching what a stage actually
    reads (a typo, a renamed setting) shows up here instead of only as a
    silently-wrong run.
    """
    schema = default_schema()
    defaults = schema.defaults()
    checked = 0
    skipped = []

    for setting in schema:
        if not setting.requires:
            continue
        if setting.always_effective:
            # Deliberately exempt: genuinely read by a second, unconditional
            # feature too (see the setting's own `always_effective` usage
            # and help text for what that is), so "ineffective while its
            # own prerequisite is off" would be simply wrong for this one.
            skipped.append(setting.name)
            continue
        probe = _non_default_probe_value(setting)
        if probe is None:
            skipped.append(setting.name)
            continue
        unmet = _first_unmet_prerequisite(setting)
        resolved = {**defaults, **unmet, setting.name: setting.coerce(probe)}
        messages = schema.ineffective_settings(resolved)
        assert any(f"Setting '{setting.name}'" in message for message in messages), (
            f"{setting.name} requires {setting.requires!r}, but setting it to "
            f"{probe!r} while {unmet!r} did not get flagged "
            "as ineffective -- something may still read it while its own "
            "prerequisite is off."
        )
        checked += 1

    # A floor, not the exact count: proves the probe-value generator is
    # actually covering the bulk of the schema's gated settings rather than
    # skipping nearly all of them without anyone noticing.
    assert checked > 100, (
        f"only checked {checked} of the schema's gated settings "
        f"(skipped, no generic probe value: {skipped})"
    )


# --- schema bounds must agree with what the setting's own consumer accepts -


def test_cartwheel_hub_tangent_length_um_excludes_the_degenerate_zero_value():
    """A sample distance of exactly 0 always lands back on the node itself
    (graph.cartwheel_guard._point_along_polyline), giving a zero-length
    vector for every edge and silently turning detect_cartwheel_hubs into a
    no-op that never flags anything. The schema's own bound must exclude
    this value, not rely on the guard function to reject it."""
    setting = default_schema()["cartwheel_hub_tangent_length_um"]
    with pytest.raises(ConfigError, match="cartwheel_hub_tangent_length_um"):
        setting.coerce(0.0)


def test_thick_vessel_braid_factor_limit_zero_is_schema_legal_and_accepted():
    """Unlike the cartwheel setting above, 0.0 here has a sensible reading
    (flag every component) rather than a silent no-op, so the schema keeps
    accepting it -- see test_thick_vessel_braid_guard.py for the guard
    function's matching acceptance of this boundary."""
    setting = default_schema()["thick_vessel_braid_factor_limit"]
    assert setting.coerce(0.0) == 0.0


# --- the generated config file ---------------------------------------------


def test_write_default_config_writes_every_setting_with_its_documentation(tmp_path):
    path = write_default_config(tmp_path / "cfg.yaml")
    text = path.read_text(encoding="utf-8")
    schema = default_schema()

    written = yaml.safe_load(text)
    keys = {key for section in written.values() for key in section}
    assert keys == set(schema.names)
    for setting in schema:
        assert f"# {setting.help}" in text, f"{setting.name} lost its help comment"


def test_the_generated_config_loads_back_as_the_schema_defaults(tmp_path):
    path = write_default_config(tmp_path / "cfg.yaml")
    assert load_config(path, default_schema()) == default_schema().defaults()


def test_write_default_config_keeps_the_values_it_is_given(tmp_path):
    path = write_default_config(
        tmp_path / "cfg.yaml", values={"inlet_p_bc": 1234.0, "do_skeletonize": False}
    )
    settings = load_config(path, default_schema())
    assert settings["inlet_p_bc"] == 1234.0
    assert settings["do_skeletonize"] is False


def test_write_default_config_refuses_a_value_the_schema_rejects(tmp_path):
    with pytest.raises(ConfigError, match="not_a_setting"):
        write_default_config(tmp_path / "cfg.yaml", values={"not_a_setting": 1})


def test_write_default_config_accepts_an_extended_schema(tmp_path):
    """Extending the pipeline schema is how the whole-brain example works."""
    from haemolynx.parsers import Setting

    extended = Schema(
        list(default_schema())
        + [Setting(name="my_sweep", kind="bool", default=True, help="Sweep", section="Mine")]
    )
    path = write_default_config(tmp_path / "cfg.yaml", schema=extended)
    assert load_config(path, extended)["my_sweep"] is True


# --- the installed-package case --------------------------------------------


def isolated_import_env(pythonpath: Path) -> dict:
    """This process's environment with `pythonpath` as the only import root.

    Isolation here means one thing -- a known `PYTHONPATH` -- and it is spelled
    as a replacement rather than a hand-built environment on purpose. An
    interpreter cannot be started in an empty environment: on Windows one
    without `SYSTEMROOT` cannot initialise its socket layer and dies importing
    `_overlapped`, so an environment listing only the paths the test cares
    about tests nothing but the platform. What the environment must not do is
    smuggle a repository checkout in through an inherited `PYTHONPATH`; that
    the checkout really is out of reach is then asserted by the subprocess
    itself, which is the only place it can be observed.
    """
    return {**os.environ, "PYTHONPATH": str(pythonpath)}


def test_the_isolated_environment_replaces_only_the_import_path(monkeypatch):
    """Everything the platform needs to start an interpreter must survive."""
    monkeypatch.setenv("PYTHONPATH", str(REPO_ROOT))
    env = isolated_import_env(SRC)

    assert env["PYTHONPATH"] == str(SRC), "an inherited PYTHONPATH leaked through"
    assert {name: value for name, value in env.items() if name != "PYTHONPATH"} == {
        name: value for name, value in os.environ.items() if name != "PYTHONPATH"
    }


def test_a_bare_package_import_can_configure_a_run(tmp_path):
    """No repository, no examples/ on the path: still able to write a config.

    This is the failure the move exists to fix, so it is checked the only way
    that is honest -- in a fresh interpreter that cannot see this checkout.
    """
    script = textwrap.dedent(
        f"""
        import importlib.util
        import pathlib
        import sys

        repo_root = pathlib.Path({str(REPO_ROOT)!r})
        reachable = [
            entry
            for entry in sys.path
            if entry
            and pathlib.Path(entry).resolve() in (repo_root, repo_root / "examples")
        ]
        assert not reachable, "the repository is importable from: " + repr(reachable)
        for name in ("examples", "resistance_network_pipeline", "pipeline_presets"):
            assert importlib.util.find_spec(name) is None, name + " is importable"

        import haemolynx
        from haemolynx.parsers import load_config
        from haemolynx.pipeline import default_schema, write_default_config

        package_dir = pathlib.Path(haemolynx.__file__).resolve().parent
        assert package_dir == pathlib.Path({str(SRC)!r}).resolve() / "haemolynx", package_dir

        write_default_config("cfg.yaml")
        settings = load_config("cfg.yaml", default_schema())
        assert set(settings) == set(default_schema().names)
        print(len(default_schema()))
"""
    )
    (tmp_path / "run.py").write_text(script, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "run.py"],
        cwd=tmp_path,
        env=isolated_import_env(SRC),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert int(result.stdout.strip().splitlines()[-1]) == len(default_schema())
    assert (tmp_path / "cfg.yaml").is_file()


def test_a_setting_graph_building_reads_is_not_hidden_behind_skeletonising():
    """Regression (PLAN_bridging.md F8): the mask-supported bridging settings
    required do_skeletonize, so a run loading its saved skeleton hid them
    -- and graph building, which reads them, still did. Every setting graph
    building reads is checked, through its whole chain of prerequisites."""
    from haemolynx.gui.form import prerequisite_chain
    from haemolynx.pipeline import default_schema
    from haemolynx.pipeline.stages import graph_build_arguments

    schema = default_schema()
    read: set = set()

    class Recording(dict):
        def __getitem__(self, name):
            read.add(name)
            return super().__getitem__(name)

        def get(self, name, default=None):
            read.add(name)
            return super().get(name, default)

    graph_build_arguments(Recording(schema.defaults()))

    assert {"bridge_require_mask_support", "bridge_max_background_gap_um"} <= read
    hidden = {
        name for name in read
        if name in schema and any(rule.lstrip("!").split("=")[0].split("!")[0] == "do_skeletonize"
                                  for rule in prerequisite_chain(schema, name))
    }
    assert not hidden


#: The toggles that skip a stage, reusing what it saved: a setting another
#: stage reads must not hide behind one of them.
_STAGE_TOGGLES = {"do_skeletonize": "skeletonise", "do_graph_building": "build_network"}

#: Reads that only report a setting, deciding nothing: the empty-skeleton
#: error names the skeleton clean-up that may have emptied it.
_READ_ONLY_TO_REPORT = {
    "build_network": {"skeleton_min_branch_length", "skeleton_min_component_percent"},
}


def _settings_each_stage_reads() -> dict[str, set[str]]:
    """The settings each stage function of ``pipeline.stages`` reads, by
    name: ``settings["x"]`` and ``settings.get("x")`` in its body and in
    every function of the module it calls, and each
    ``prefixed_arguments(settings, prefix, parameters_of(f))``'s names.
    Names built at run time (``f"{role}_..."``) are not seen."""
    import ast
    import inspect

    from haemolynx.parsers import parameters_of
    from haemolynx.pipeline import stages

    functions = {
        node.name: node for node in ast.parse(inspect.getsource(stages)).body
        if isinstance(node, ast.FunctionDef)
    }

    def direct(function):
        reads, calls = set(), set()
        for node in ast.walk(function):
            if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                    and node.value.id == "settings" and isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, str)):
                reads.add(node.slice.value)
            if not isinstance(node, ast.Call):
                continue
            called = node.func
            if (isinstance(called, ast.Attribute) and called.attr == "get"
                    and isinstance(called.value, ast.Name) and called.value.id == "settings"
                    and node.args and isinstance(node.args[0], ast.Constant)):
                reads.add(node.args[0].value)
            if isinstance(called, ast.Name):
                calls.add(called.id)
                if called.id == "prefixed_arguments" and len(node.args) == 3:
                    prefix = node.args[1].value
                    target = eval(ast.unparse(node.args[2].args[0]), vars(stages))
                    reads.update(prefix + name for name in parameters_of(target))
        return reads, calls

    found = {name: direct(function) for name, function in functions.items()}

    def closure(name, seen):
        if name in seen or name not in found:
            return set()
        seen.add(name)
        reads, calls = found[name]
        return reads.union(*(closure(callee, seen) for callee in calls))

    return {stage: closure(stage, set()) for stage in (
        "segment", "skeletonise", "build_network", "assign_boundaries", "assign_diameters",
        "build_haemodynamic_model", "solve", "post_process", "run_perturbations", "export_results",
    )}


def test_no_setting_a_stage_reads_hides_behind_another_stages_toggle():
    """PLAN_bridging.md Phase 3: a setting a stage reads must be shown while
    that stage runs. With do_skeletonize off the skeleton is loaded, but
    graph building still splits the thick vessels (the region read back
    beside the skeleton) and post processing bridges edited vessels into
    them; with do_graph_building off the graph is loaded, but post
    processing still judges an edited vessel's stubs and loops. The panel
    reads every setting's whole chain of prerequisites (`form.
    prerequisite_chain`), so a row needing either toggle hid while a stage
    went on reading it."""
    from haemolynx.gui.form import prerequisite_chain
    from haemolynx.parsers import prerequisite_name
    from haemolynx.pipeline import default_schema

    schema = default_schema()
    reads = _settings_each_stage_reads()
    assert {"bridge_require_mask_support", "thick_vessel_fill_centreline_gaps"} <= reads["build_network"]
    assert "skeleton_closing_radius" in reads["skeletonise"], "the prefixed reads are seen"

    hidden = {
        (stage, name, toggle)
        for stage, names in reads.items()
        for name in names - _READ_ONLY_TO_REPORT.get(stage, set())
        if name in schema
        for toggle in {prerequisite_name(rule) for rule in prerequisite_chain(schema, name)}
        if _STAGE_TOGGLES.get(toggle, stage) != stage
    }
    assert not hidden


def test_with_a_saved_skeleton_and_graph_the_thick_vessel_split_stays_in_effect():
    """The rows the stages go on reading are in effect with both stages
    skipped; the thin-arm join and the braiding check, read only while
    skeletonising, are not."""
    from haemolynx.pipeline import default_schema

    schema = default_schema()
    values = schema.defaults()
    values.update(do_skeletonize=False, do_graph_building=False, use_thick_vessel_skeletonisation=True)

    ineffective = " ".join(schema.ineffective_settings(values))
    for name in ("use_thick_vessel_skeletonisation", "skeleton_thick_vessel_min_radius_um",
                 "thick_vessel_fill_centreline_gaps", "thick_bridge_centreline_tolerance_um",
                 "min_stub_length", "remove_short_loops_under_um"):
        assert name not in ineffective
    values.update(
        skeleton_thick_vessel_max_bridge_radius_multiple=1.0,
        detect_thick_vessel_braiding=not schema["detect_thick_vessel_braiding"].default,
    )
    ineffective = " ".join(schema.ineffective_settings(values))
    assert "skeleton_thick_vessel_max_bridge_radius_multiple" in ineffective
    assert "detect_thick_vessel_braiding" in ineffective


def test_bridging_settings_are_in_effect_with_a_saved_skeleton():
    from haemolynx.pipeline import default_schema

    schema = default_schema()
    values = schema.defaults()
    values.update(do_skeletonize=False, bridge_max_background_gap_um=1.0, bridge_min_mask_fraction=0.5)

    assert not [m for m in schema.ineffective_settings(values) if "bridge_" in m]
    values["bridge_require_mask_support"] = False
    assert any("bridge_max_background_gap_um" in m for m in schema.ineffective_settings(values))
