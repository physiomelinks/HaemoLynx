"""A perturbation as a named, typed settings override.

Two things are under test here, and they pull in opposite directions on
purpose. Reading a perturbation must never raise -- a hand-edited config has
to open in the panel so the user can see what is wrong with it -- while running
one must, so a mistyped setting name is not a re-solve that quietly does
nothing. So the reader collects `problems` and the pre-run checks turn them
into errors.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

yaml = pytest.importorskip("yaml")

from haemolynx.haemodynamics.perturbations import (  # noqa: E402
    INCOMPARABLE_OVERRIDES,
    PERTURBATION_TYPES,
    SETTINGS_FOR_TYPE,
    PerturbationSpec,
    is_usable_as_a_directory_name,
    perturbation_folder_name,
    perturbation_output_dir,
    perturbation_problems,
    perturbations_from_settings,
    perturbations_to_settings,
    settings_for_perturbation_type,
    visible_perturbation_settings,
)
from haemolynx.pipeline import default_schema  # noqa: E402
from haemolynx.pipeline.checks import check_perturbations  # noqa: E402

SCHEMA = default_schema()

TWO_ENTRIES = [
    {
        "name": "art_dilate_20",
        "type": "arteriole_diameter_change",
        "overrides": {"arteriole_diameter_change_percent": 20},
    },
    {
        "name": "higher_inlet",
        "type": "pressure_sweep",
        "overrides": {"inlet_pressure_min_pa": 5000, "inlet_pressure_max_pa": 7000},
    },
]


# --- what a type declares ----------------------------------------------------


def test_every_type_says_which_settings_it_reads():
    """The reveal table: a type with no entry here would show no options."""
    assert set(SETTINGS_FOR_TYPE) == set(PERTURBATION_TYPES)


def test_every_setting_a_type_reads_exists():
    """A renamed setting would otherwise leave a type with a row that is gone."""
    for perturbation_type, names in SETTINGS_FOR_TYPE.items():
        unknown = sorted(name for name in names if name not in SCHEMA)
        assert unknown == [], f"{perturbation_type} reads {unknown}, which do not exist"


def test_a_type_reveals_only_its_own_options():
    """Options are hidden until their type is chosen, not merely greyed out."""
    assert settings_for_perturbation_type("arteriole_diameter_change") == (
        "arteriole_diameter_change_percent",
    )
    assert settings_for_perturbation_type("none") == ()
    # A type nothing knows shows nothing rather than the last one's rows.
    assert settings_for_perturbation_type("typo") == ()


def test_combined_arteriole_pericyte_type_includes_arteriole_percent():
    """Single-shot combined type is arteriole % plus the pericyte knobs."""
    from haemolynx.haemodynamics.perturbations import PERICYTE_CONSTRICTION_SETTINGS

    combined = SETTINGS_FOR_TYPE["arteriole_and_pericyte_diameter_change"]
    assert combined[0] == "arteriole_diameter_change_percent"
    assert "arteriole_diameter_change_percent" in combined
    assert set(PERICYTE_CONSTRICTION_SETTINGS) <= set(combined)
    assert combined == (
        "arteriole_diameter_change_percent",
        *PERICYTE_CONSTRICTION_SETTINGS,
    )

def test_the_visible_settings_are_the_union_of_the_configured_types():
    specs = perturbations_from_settings({"perturbations": TWO_ENTRIES})

    visible = visible_perturbation_settings(specs)

    assert "arteriole_diameter_change_percent" in visible
    assert "inlet_pressure_min_pa" in visible
    assert "pericyte_mask_path" not in visible


def test_a_type_cannot_be_expressed_as_a_prerequisite():
    """Why the table above exists rather than `requires` on each setting.

    `Schema.__init__` refuses a prerequisite that is not a bool, and a
    perturbation's type is a choice -- so visibility has to be declared, the
    way `gui.boundary_picking` declares it for the selection methods.
    """
    from haemolynx.parsers import ConfigError, Schema, Setting

    with pytest.raises(ConfigError, match="not a bool"):
        Schema(
            [
                Setting("kind_of_thing", "choice", "none", "Which kind", "S",
                        choices=("none", "other")),
                Setting("scale", "float", 1.0, "How much", "S",
                        requires=("kind_of_thing",)),
            ]
        )


# --- reading and writing the list --------------------------------------------


def test_a_two_entry_list_round_trips_through_yaml_bytes_for_bytes():
    """The setting is edited by hand and by a panel, so both must agree."""
    specs = perturbations_from_settings({"perturbations": TWO_ENTRIES})
    assert [spec.name for spec in specs] == ["art_dilate_20", "higher_inlet"]
    assert [spec.problems for spec in specs] == [(), ()]

    once = yaml.safe_dump(perturbations_to_settings(specs), sort_keys=False)
    reloaded = perturbations_from_settings(yaml.safe_load(once))
    twice = yaml.safe_dump(perturbations_to_settings(reloaded), sort_keys=False)

    assert twice == once


def test_a_coerced_path_override_still_dumps(tmp_path):
    """A run's settings carry `Path`; `yaml.safe_dump` refuses one outright."""
    spec = PerturbationSpec.from_entry(
        {
            "name": "masked",
            "type": "pericyte_diameter_change",
            "overrides": {"pericyte_mask_path": tmp_path / "mask.tif"},
        }
    )

    dumped = yaml.safe_dump(perturbations_to_settings([spec]))

    assert "mask.tif" in dumped
    assert "\\" not in dumped, "a Windows path must dump the way Linux dumps it"


def test_a_single_entry_written_without_a_list_is_still_read():
    specs = perturbations_from_settings({"perturbations": TWO_ENTRIES[0]})

    assert [spec.name for spec in specs] == ["art_dilate_20"]


def test_no_perturbations_is_no_perturbations():
    assert perturbations_from_settings({}) == ()
    assert perturbations_from_settings({"perturbations": []}) == ()
    assert perturbations_from_settings({"perturbations": None}) == ()


# --- a malformed entry -------------------------------------------------------


def test_a_malformed_entry_reports_a_problem_instead_of_raising():
    """The panel must open on a config it cannot use, and say why."""
    specs = perturbations_from_settings({"perturbations": ["just a string"]})

    assert len(specs) == 1
    assert specs[0].problems
    assert "not a name/type/overrides entry" in specs[0].problems[0]


def test_an_entry_with_no_name_is_reported_and_still_named():
    """Output goes in a directory per perturbation, so it needs some name."""
    (spec,) = perturbations_from_settings(
        {"perturbations": [{"type": "none", "overrides": {}}]}
    )

    assert any("has no name" in problem for problem in spec.problems)
    assert spec.name


@pytest.mark.parametrize(
    "entry,expected",
    [
        ({"name": "a", "overrides": {}}, "has no type"),
        ({"name": "a", "type": "make_it_faster"}, "unknown type"),
        ({"name": "a", "type": "none", "overrides": 7}, "not a mapping"),
        ({"name": "a", "type": "none", "extra": 1}, "unexpected key"),
    ],
)
def test_each_way_of_writing_an_entry_wrong_says_which(entry, expected):
    (spec,) = perturbations_from_settings({"perturbations": [entry]})

    assert any(expected in problem for problem in spec.problems), spec.problems


def test_an_override_the_schema_does_not_know_is_a_problem_with_a_suggestion():
    (spec,) = perturbations_from_settings(
        {
            "perturbations": [
                {"name": "a", "type": "pressure_sweep",
                 "overrides": {"inlet_pressure_min": 5000}}
            ]
        }
    )

    (problem,) = spec.schema_problems(SCHEMA)
    assert "Unknown setting" in problem
    assert "inlet_pressure_min_pa" in problem, "the spelling suggestion is the point"


def test_an_override_the_schema_rejects_is_a_problem():
    (spec,) = perturbations_from_settings(
        {
            "perturbations": [
                {"name": "a", "type": "arteriole_diameter_change",
                 "overrides": {"arteriole_diameter_change_percent": "wider please"}}
            ]
        }
    )

    (problem,) = spec.schema_problems(SCHEMA)
    assert "arteriole_diameter_change_percent" in problem


def test_a_partial_override_is_not_validated_as_a_whole_config():
    """`Schema.validate` would raise IneffectiveSettingWarning on the fragment.

    A perturbation sets `pericyte_mask_path` while `use_pericyte_mask_constriction`
    lives in the run it is applied to, so validating the two-key fragment on its
    own would report a prerequisite that is in fact met. Each value is coerced
    on its own instead, exactly as the presets are.
    """
    (spec,) = perturbations_from_settings(
        {
            "perturbations": [
                {"name": "a", "type": "pericyte_diameter_change",
                 "overrides": {"pericyte_mask_path": "somewhere/mask.tif"}}
            ]
        }
    )

    import warnings

    from haemolynx.parsers import IneffectiveSettingWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error", IneffectiveSettingWarning)
        assert spec.schema_problems(SCHEMA) == ()

    assert spec.coerced_overrides(SCHEMA)["pericyte_mask_path"] == Path(
        "somewhere/mask.tif"
    )


def test_an_override_the_type_does_not_read_is_reported_as_unused():
    (spec,) = perturbations_from_settings(
        {
            "perturbations": [
                {"name": "a", "type": "arteriole_diameter_change",
                 "overrides": {"arteriole_diameter_change_percent": 20, "inlet_p_bc": 3000.0}}
            ]
        }
    )

    assert spec.unused_overrides() == ("inlet_p_bc",)
    assert spec.schema_problems(SCHEMA) == (), "it is a real setting, just unread"


def test_only_what_the_type_reads_is_applied():
    """What makes `unused_overrides` true rather than merely well-meant.

    Every perturbation re-solves the network, so an unread override left in the
    settings dict -- a boundary pressure, say -- would change the answer while
    being reported as having no effect.
    """
    (spec,) = perturbations_from_settings(
        {
            "perturbations": [
                {"name": "a", "type": "arteriole_diameter_change",
                 "overrides": {"arteriole_diameter_change_percent": 20, "inlet_p_bc": 3000.0}}
            ]
        }
    )

    assert spec.applied_overrides(SCHEMA) == {"arteriole_diameter_change_percent": 20}
    assert "inlet_p_bc" in spec.coerced_overrides(SCHEMA), "still read back for the report"


def test_a_well_formed_dilation_sweep_entry_is_accepted():
    entry = {
        "name": "dilation_sweep",
        "type": "pericyte_dilation_sweep",
        "overrides": {
            "pericyte_dilation_min_percent": 1,
            "pericyte_dilation_max_percent": 5,
            "pericyte_dilation_step_percent": 1,
            "constriction_by_branch_order": {"Art1": 1.0, "B01": 0.5, "Ven1": 1.0},
            "constriction_length_um": 20.0,
            "constriction_spacing_um": 80.0,
            "use_probabilistic_pericyte_constriction": True,
            "pericyte_constriction_probability": 0.5,
        },
    }
    (spec,) = perturbations_from_settings({"perturbations": [entry]})

    assert spec.problems == ()
    assert spec.schema_problems(SCHEMA) == ()
    assert spec.unused_overrides() == ()
    assert perturbation_problems({"perturbations": [entry]}, SCHEMA) == ()
    report = check_perturbations(_settings(perturbations=[entry]), SCHEMA)
    assert not report.errors
    applied = spec.applied_overrides(SCHEMA)
    assert applied["constriction_by_branch_order"] == {
        "Art1": 1.0,
        "B01": 0.5,
        "Ven1": 1.0,
    }
    assert applied["constriction_length_um"] == 20.0
    assert applied["constriction_spacing_um"] == 80.0
    assert applied["pericyte_constriction_probability"] == 0.5


def test_pericyte_sweep_types_expose_length_spacing_and_probability():
    from haemolynx.haemodynamics.perturbations import PERICYTE_ENTRY_GEOMETRY_SETTINGS

    for perturbation_type in (
        "pericyte_diameter_change",
        "arteriole_and_pericyte_diameter_change",
        "pericyte_dilation_sweep",
        "pressure_and_pericyte_sweep",
    ):
        reads = set(SETTINGS_FOR_TYPE[perturbation_type])
        missing = sorted(set(PERICYTE_ENTRY_GEOMETRY_SETTINGS) - reads)
        assert missing == [], f"{perturbation_type} is missing {missing}"


def test_every_focal_pericyte_type_exposes_constriction_by_branch_order():
    """Branch-order tone is a per-entry override on every focal-constriction type."""
    for perturbation_type in (
        "pericyte_diameter_change",
        "arteriole_and_pericyte_diameter_change",
        "pericyte_dilation_sweep",
        "pressure_and_pericyte_sweep",
        "pericyte_spacing_sweep",
        "pericyte_length_sweep",
    ):
        reads = set(SETTINGS_FOR_TYPE[perturbation_type])
        assert "constriction_by_branch_order" in reads, perturbation_type
        assert "pericyte_constriction_factor" in reads, perturbation_type


def test_a_well_formed_pressure_and_pericyte_sweep_is_accepted():
    entry = {
        "name": "both",
        "type": "pressure_and_pericyte_sweep",
        "overrides": {
            "pericyte_dilation_min_percent": 1,
            "pericyte_dilation_max_percent": 5,
            "inlet_pressure_min_pa": 4500,
            "inlet_pressure_max_pa": 5000,
            "constriction_length_um": 25.0,
            "constriction_spacing_um": 90.0,
            "use_probabilistic_pericyte_constriction": True,
            "pericyte_constriction_probability": 0.4,
        },
    }
    report = check_perturbations(_settings(perturbations=[entry]), SCHEMA)
    assert not report.errors
    (spec,) = perturbations_from_settings({"perturbations": [entry]})
    assert spec.unused_overrides() == ()
    assert spec.applied_overrides(SCHEMA)["constriction_spacing_um"] == 90.0

def test_a_well_formed_arteriole_diameter_sweep_is_accepted():
    entry = {
        "name": "art_sweep",
        "type": "arteriole_diameter_sweep",
        "overrides": {
            "arteriole_dilation_min_percent": 0,
            "arteriole_dilation_max_percent": 20,
            "arteriole_dilation_step_percent": 10,
        },
    }
    (spec,) = perturbations_from_settings({"perturbations": [entry]})
    assert spec.problems == ()
    assert spec.schema_problems(SCHEMA) == ()
    assert spec.unused_overrides() == ()
    assert perturbation_problems({"perturbations": [entry]}, SCHEMA) == ()


def test_a_well_formed_pressure_and_arteriole_sweep_is_accepted():
    entry = {
        "name": "art_and_p",
        "type": "pressure_and_arteriole_sweep",
        "overrides": {
            "arteriole_dilation_min_percent": 0,
            "arteriole_dilation_max_percent": 10,
            "inlet_pressure_min_pa": 4500,
            "inlet_pressure_max_pa": 5000,
        },
    }
    report = check_perturbations(_settings(perturbations=[entry]), SCHEMA)
    assert not report.errors


def test_a_well_formed_capillary_diameter_sweep_is_accepted():
    entry = {
        "name": "cap_sweep",
        "type": "capillary_diameter_sweep",
        "overrides": {
            "capillary_dilation_min_percent": 0,
            "capillary_dilation_max_percent": 20,
            "capillary_dilation_step_percent": 10,
        },
    }
    (spec,) = perturbations_from_settings({"perturbations": [entry]})
    assert spec.problems == ()
    assert spec.schema_problems(SCHEMA) == ()
    assert spec.unused_overrides() == ()
    assert perturbation_problems({"perturbations": [entry]}, SCHEMA) == ()


def test_a_well_formed_pressure_and_capillary_sweep_is_accepted():
    entry = {
        "name": "cap_and_p",
        "type": "pressure_and_capillary_sweep",
        "overrides": {
            "capillary_dilation_min_percent": 0,
            "capillary_dilation_max_percent": 10,
            "inlet_pressure_min_pa": 4500,
            "inlet_pressure_max_pa": 5000,
        },
    }
    report = check_perturbations(_settings(perturbations=[entry]), SCHEMA)
    assert not report.errors


def test_a_dilation_sweep_with_an_unread_override_is_rejected():
    entry = {
        "name": "dilation_sweep",
        "type": "pericyte_dilation_sweep",
        "overrides": {
            "pericyte_dilation_min_percent": 1,
            "inlet_pressure_min_pa": 4500,
        },
    }
    (spec,) = perturbations_from_settings({"perturbations": [entry]})

    assert spec.unused_overrides() == ("inlet_pressure_min_pa",)
    report = check_perturbations(_settings(perturbations=[entry]), SCHEMA)
    assert report.ok  # unread overrides are warnings, not errors
    assert any("inlet_pressure_min_pa" in warning for warning in report.warnings)


@pytest.mark.parametrize("name", INCOMPARABLE_OVERRIDES)
def test_the_blood_model_cannot_be_perturbed(name):
    """A law that changes every resistance cannot be one arm's difference."""
    (spec,) = perturbations_from_settings(
        {
            "perturbations": [
                {"name": "a", "type": "arteriole_diameter_change",
                 "overrides": {"arteriole_diameter_change_percent": 20, name: "pries"}}
            ]
        }
    )

    assert spec.incomparable_overrides() == (name,)
    # Reported once, as the refusal: not a second, milder line calling it unread.
    assert spec.unused_overrides() == ()
    assert any("may not change" in problem for problem in perturbation_problems(
        {"perturbations": [
            {"name": "a", "type": "arteriole_diameter_change",
             "overrides": {name: "pries"}}
        ]},
        SCHEMA,
    ))


@pytest.mark.parametrize("name", ("../elsewhere", "sub/dir", r"back\slash", ".."))
def test_a_name_that_is_a_path_is_a_problem(name):
    """A perturbation's name is used in the directory its output goes in."""
    (spec,) = perturbations_from_settings(
        {"perturbations": [{"name": name, "type": "none"}]}
    )

    assert any("directory name" in problem for problem in spec.problems), spec.problems


def test_an_ordinary_name_is_left_alone():
    (spec,) = perturbations_from_settings(
        {"perturbations": [{"name": "art_dilate_20", "type": "none"}]}
    )

    assert spec.problems == ()
    assert is_usable_as_a_directory_name("art_dilate_20")


# --- the pre-run checks ------------------------------------------------------


def _settings(**overrides):
    values = SCHEMA.defaults()
    values.update(overrides)
    return values


def test_a_well_formed_list_passes_preflight():
    report = check_perturbations(_settings(perturbations=TWO_ENTRIES), SCHEMA)

    assert report.ok, report.errors
    assert report.warnings == []
    assert any("art_dilate_20" in detail for _label, detail in report.passed)


def test_no_perturbations_checks_nothing():
    report = check_perturbations(_settings(), SCHEMA)

    assert report.ok and not report.passed and not report.warnings


def test_a_malformed_entry_is_an_error_before_the_run_starts():
    report = check_perturbations(_settings(perturbations=["nonsense"]), SCHEMA)

    assert not report.ok
    assert any("not a name/type/overrides entry" in error for error in report.errors)


def test_two_perturbations_with_one_name_is_an_error():
    """Each writes its own CSVs under its own name, so they cannot share one."""
    entries = [dict(TWO_ENTRIES[0]), dict(TWO_ENTRIES[0])]
    report = check_perturbations(_settings(perturbations=entries), SCHEMA)

    assert not report.ok
    assert any("repeats the name" in error for error in report.errors)


def test_an_unknown_type_is_an_error():
    report = check_perturbations(
        _settings(perturbations=[{"name": "a", "type": "wider"}]), SCHEMA
    )

    assert not report.ok
    assert any("unknown type" in error for error in report.errors)


def test_an_override_key_the_schema_does_not_have_is_an_error():
    report = check_perturbations(
        _settings(
            perturbations=[
                {"name": "a", "type": "none", "overrides": {"arteriole_diameter": 1.2}}
            ]
        ),
        SCHEMA,
    )

    assert not report.ok
    assert any("Unknown setting" in error for error in report.errors)


def test_an_unused_override_is_a_warning_and_not_a_failure():
    report = check_perturbations(
        _settings(
            perturbations=[
                {"name": "a", "type": "arteriole_diameter_change",
                 "overrides": {"arteriole_diameter_change_percent": 10, "inlet_p_bc": 3000.0}}
            ]
        ),
        SCHEMA,
    )

    assert report.ok
    assert any("does not read" in warning for warning in report.warnings)


def test_a_perturbation_naming_a_mask_that_is_not_there_fails_preflight(tmp_path):
    """The whole point of checking before the run: this path is read last.

    A perturbation re-solves the network after every stage has finished, so a
    mask it names that is not on disk would waste the entire run before saying
    so.
    """
    missing = tmp_path / "no_such_mask.tif"
    entry = {
        "name": "tighter",
        "type": "pericyte_diameter_change",
        "overrides": {
            "use_pericyte_mask_constriction": True,
            "pericyte_mask_path": str(missing),
        },
    }

    report = check_perturbations(_settings(perturbations=[entry]), SCHEMA)

    assert not report.ok
    assert any("pericyte_mask_path" in error for error in report.errors)


def test_a_perturbation_naming_a_mask_that_is_there_passes(tmp_path):
    mask = tmp_path / "mask.tif"
    mask.write_bytes(b"")
    entry = {
        "name": "tighter",
        "type": "pericyte_diameter_change",
        "overrides": {
            "use_pericyte_mask_constriction": True,
            "pericyte_mask_path": str(mask),
        },
    }

    report = check_perturbations(_settings(perturbations=[entry]), SCHEMA)

    assert report.ok, report.errors


def test_preflight_reports_a_bad_perturbation(capsys):
    """It has to be wired in, not merely available."""
    from haemolynx.pipeline import preflight

    report = preflight(
        _settings(perturbations=[{"name": "a", "type": "wider"}]), SCHEMA
    )

    assert not report.ok
    assert any("unknown type" in error for error in report.errors)


# --- where the output goes ---------------------------------------------------


def test_the_folder_name_is_the_user_name_underscore_the_type():
    """Exactly ``{name}_{type}``: the user-facing name and the type enum string."""
    assert perturbation_folder_name("strokeA", "constriction") == "strokeA_constriction"
    assert (
        perturbation_folder_name("art_dilate_20", "arteriole_diameter_change")
        == "art_dilate_20_arteriole_diameter_change"
    )
    spec = PerturbationSpec(name="strokeA", type="pericyte_diameter_change")
    assert spec.folder_name == "strokeA_pericyte_diameter_change"


def test_the_output_dir_falls_back_to_the_general_output_folder():
    values = _settings(vtk_output_prefix=Path("results/run_one/network"))

    assert perturbation_output_dir(values) == Path("results/run_one")


def test_the_output_dir_falls_back_to_cwd_when_the_run_has_no_prefix():
    assert perturbation_output_dir({}) == Path(".")


def test_the_perturbation_output_dir_setting_defaults_to_unset():
    """Unset at schema level; runtime resolution follows vtk_output_prefix."""
    assert SCHEMA["perturbation_output_dir"].default is None


def test_a_configured_output_dir_wins():
    values = _settings(
        vtk_output_prefix=Path("results/run_one/network"),
        perturbation_output_dir=Path("elsewhere"),
    )

    assert perturbation_output_dir(values) == Path("elsewhere")


# --- the public names --------------------------------------------------------


def test_the_perturbation_names_are_reachable_from_the_subpackage():
    import haemolynx.haemodynamics as haemodynamics

    for name in (
        "INCOMPARABLE_OVERRIDES",
        "PERTURBATION_TYPES",
        "PerturbationSpec",
        "is_usable_as_a_directory_name",
        "perturbation_folder_name",
        "perturbations_from_settings",
        "perturbations_to_settings",
        "perturbation_problems",
        "perturbation_output_dir",
        "settings_for_perturbation_type",
        "visible_perturbation_settings",
    ):
        assert hasattr(haemodynamics, name), name
        assert name in haemodynamics.__all__, name


def test_the_perturbation_module_imports_no_gui():
    """Pure, like the rest of what the panel reads: settings in, specs out."""
    import ast

    source = SRC_DIR / "haemolynx" / "haemodynamics" / "perturbations.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert not imported & {"napari", "magicgui", "qtpy", "haemolynx"}


def test_perturbation_problems_reads_both_kinds_at_once():
    """One list of lines, so the panel and the checks cannot disagree."""
    problems = perturbation_problems(
        {
            "perturbations": [
                {"name": "a", "type": "not_a_type", "overrides": {"nope": 1}},
                {"name": "a", "type": "none"},
            ]
        },
        SCHEMA,
    )

    assert any("unknown type" in problem for problem in problems)
    assert any("Unknown setting" in problem for problem in problems)
    assert any("repeats the name" in problem for problem in problems)
