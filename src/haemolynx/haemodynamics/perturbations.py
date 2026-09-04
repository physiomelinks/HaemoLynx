"""A perturbation: a named, typed settings override to re-solve a network for.

Once a run has a network with resistances on it, the question is usually not
"what does this network do" but "what does it do if". Each answer is a full
re-solve of the same baseline with a few settings changed -- a different inlet
pressure, wider arterioles, tighter pericytes -- so a perturbation is described
the same way a preset is: a name, and the settings it sets.

    perturbations:
      - name: art_dilate_20
        type: arteriole_diameter_change
        overrides:
          arteriole_diameter_change_percent: 20

The ``type`` is what a run dispatches on and what the panel reveals rows for:
:data:`SETTINGS_FOR_TYPE` says which settings each type reads, so an override
that type will not look at can be reported rather than silently ignored. It
cannot be expressed as a setting's ``requires`` -- a prerequisite must be a
bool and a type is a choice -- which is the same problem
:mod:`haemolynx.gui.boundary_picking` solves for the boundary selection
methods, and this follows it.

Every perturbation runs **from the same baseline**: two of them do not compose,
and neither changes the run that produced the network.

Nothing here executes anything, imports a GUI, or raises for a badly written
entry. A hand-edited config must not stop a panel from opening, so reading one
collects :attr:`PerturbationSpec.problems` and carries on; the run is where a
bad entry is an error, and :func:`haemolynx.pipeline.preflight` says so before
any work starts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "INCOMPARABLE_OVERRIDES",
    "PERTURBATION_TYPES",
    "PERICYTE_CONSTRICTION_SETTINGS",
    "PERICYTE_ENTRY_GEOMETRY_SETTINGS",
    "PERICYTE_DILATION_SWEEP_SETTINGS",
    "PERICYTE_SPACING_SWEEP_SETTINGS",
    "PERICYTE_LENGTH_SWEEP_SETTINGS",
    "ARTERIOLE_DILATION_SWEEP_SETTINGS",
    "CAPILLARY_DILATION_SWEEP_SETTINGS",
    "PRESSURE_SWEEP_SETTINGS",
    "SETTINGS_FOR_TYPE",
    "SWEEP_PERTURBATION_TYPES",
    "PerturbationSpec",
    "is_sweep_perturbation",
    "perturbation_folder_name",
    "perturbation_output_dir",
    "perturbation_problems",
    "perturbations_from_settings",
    "perturbations_to_settings",
    "is_usable_as_a_directory_name",
    "settings_for_perturbation_type",
    "visible_perturbation_settings",
]

#: What a perturbation can be. ``none`` does nothing and produces nothing: it
#: is the type an entry has before a user has chosen one.
PERTURBATION_TYPES: tuple[str, ...] = (
    "none",
    "pressure_sweep",
    "pressure_and_pericyte_sweep",
    "pericyte_dilation_sweep",
    "arteriole_diameter_change",
    "arteriole_diameter_sweep",
    "pressure_and_arteriole_sweep",
    "capillary_diameter_sweep",
    "pressure_and_capillary_sweep",
    "pericyte_spacing_sweep",
    "pericyte_length_sweep",
    "pericyte_diameter_change",
    "arteriole_and_pericyte_diameter_change",
)

#: Length, spacing, probability, base tone and per-order overrides every
#: pericyte-based perturbation can set on its own entry. Sweeps read these
#: through the merged settings; ``pericyte_diameter_change`` also needs the
#: probabilistic enable flag so the existing constriction-strategy path can
#: honour probability. ``constriction_by_branch_order`` replaces
#: ``pericyte_constriction_factor`` for listed orders only.
PERICYTE_ENTRY_GEOMETRY_SETTINGS: tuple[str, ...] = (
    "pericyte_constriction_factor",
    "constriction_by_branch_order",
    "constriction_length_um",
    "constriction_spacing_um",
    "use_probabilistic_pericyte_constriction",
    "pericyte_constriction_probability",
)

#: Settings that configure pericyte / constriction modelling. Declared under
#: Diameters and pericytes in the schema (so apply.py's section_values still
#: finds them) but claimed on the Perturbations tab: they are options of a
#: pericyte-based perturbation, not of the baseline diameter model.
#: ``do_pericyte_construction`` is *not* listed: the pipeline forces it False
#: on baseline and every merge, and typed pericyte paths call
#: ``set_resistances_for_constriction_strategy`` by type — so a checkbox would
#: lie. It is retabbed with the other legacy flags in progress.py.
#: Order matches ``_PERICYTE_SETTINGS_ON_PERTURBATIONS_TAB`` in progress.py.
PERICYTE_CONSTRICTION_SETTINGS: tuple[str, ...] = (
    "pericyte_constriction_factor",
    "constriction_by_branch_order",
    "constriction_length_um",
    "constriction_spacing_um",
    "use_pericyte_mask_constriction",
    "pericyte_mask_path",
    "pericyte_mask_h5_dataset_name",
    "pericyte_max_assignment_distance_um",
    "pericyte_min_diameter_um",
    "pericyte_max_diameter_um",
    "use_probabilistic_pericyte_constriction",
    "pericyte_constriction_probability",
    "pericyte_constriction_seed",
)

#: Dilation-percent axes for a pericyte dilation sweep.
PERICYTE_DILATION_SWEEP_SETTINGS: tuple[str, ...] = (
    "pericyte_dilation_min_percent",
    "pericyte_dilation_max_percent",
    "pericyte_dilation_step_percent",
)

#: Inter-pericyte spacing axis for a geometry sweep at fixed length and tone.
PERICYTE_SPACING_SWEEP_SETTINGS: tuple[str, ...] = (
    "constriction_spacing_min_um",
    "constriction_spacing_max_um",
    "constriction_spacing_step_um",
)

#: Constriction-length axis for a geometry sweep at fixed spacing and tone.
PERICYTE_LENGTH_SWEEP_SETTINGS: tuple[str, ...] = (
    "constriction_length_min_um",
    "constriction_length_max_um",
    "constriction_length_step_um",
)

#: Whole-branch diameter-percent axes for an arteriole dilation sweep.
ARTERIOLE_DILATION_SWEEP_SETTINGS: tuple[str, ...] = (
    "arteriole_dilation_min_percent",
    "arteriole_dilation_max_percent",
    "arteriole_dilation_step_percent",
)

#: Whole-branch diameter-percent axes for a passive capillary dilation sweep.
CAPILLARY_DILATION_SWEEP_SETTINGS: tuple[str, ...] = (
    "capillary_dilation_min_percent",
    "capillary_dilation_max_percent",
    "capillary_dilation_step_percent",
)

#: Inlet-pressure axes for a pressure sweep.
PRESSURE_SWEEP_SETTINGS: tuple[str, ...] = (
    "inlet_pressure_min_pa",
    "inlet_pressure_max_pa",
    "inlet_pressure_step_pa",
)

#: Which settings each type reads. This is the table the panel shows rows from
#: -- a type's options stay hidden until that type is chosen, rather than
#: greyed out -- and what tells a user that an override is doing nothing.
SETTINGS_FOR_TYPE: Mapping[str, tuple[str, ...]] = {
    "none": (),
    "pressure_sweep": PRESSURE_SWEEP_SETTINGS,
    "pressure_and_pericyte_sweep": (
        *PERICYTE_DILATION_SWEEP_SETTINGS,
        *PERICYTE_ENTRY_GEOMETRY_SETTINGS,
        *PRESSURE_SWEEP_SETTINGS,
    ),
    "pericyte_dilation_sweep": (
        *PERICYTE_DILATION_SWEEP_SETTINGS,
        *PERICYTE_ENTRY_GEOMETRY_SETTINGS,
    ),
    "arteriole_diameter_change": ("arteriole_diameter_change_percent",),
    "arteriole_diameter_sweep": ARTERIOLE_DILATION_SWEEP_SETTINGS,
    "pressure_and_arteriole_sweep": (
        *ARTERIOLE_DILATION_SWEEP_SETTINGS,
        *PRESSURE_SWEEP_SETTINGS,
    ),
    "capillary_diameter_sweep": CAPILLARY_DILATION_SWEEP_SETTINGS,
    "pressure_and_capillary_sweep": (
        *CAPILLARY_DILATION_SWEEP_SETTINGS,
        *PRESSURE_SWEEP_SETTINGS,
    ),
    "pericyte_spacing_sweep": (
        *PERICYTE_SPACING_SWEEP_SETTINGS,
        "pericyte_constriction_factor",
        "constriction_by_branch_order",
        "constriction_length_um",
        "pericyte_geometry_dilation_percent",
        "use_probabilistic_pericyte_constriction",
        "pericyte_constriction_probability",
    ),
    "pericyte_length_sweep": (
        *PERICYTE_LENGTH_SWEEP_SETTINGS,
        "pericyte_constriction_factor",
        "constriction_by_branch_order",
        "constriction_spacing_um",
        "pericyte_geometry_dilation_percent",
        "use_probabilistic_pericyte_constriction",
        "pericyte_constriction_probability",
    ),
    "pericyte_diameter_change": PERICYTE_CONSTRICTION_SETTINGS,
    # Union of arteriole_diameter_change and pericyte_diameter_change. Apply
    # order in stages._perturb_one is arteriole whole-branch % scale first,
    # then focal pericyte constrictions on the scaled graph.
    "arteriole_and_pericyte_diameter_change": (
        "arteriole_diameter_change_percent",
        *PERICYTE_CONSTRICTION_SETTINGS,
    ),
}

#: Settings a perturbation may not override, because changing one of them
#: makes its resistances incomparable with the baseline it is differenced
#: against. The viscosity law, the diameter basis and the haematocrit each
#: change *every* resistance in the network -- roughly doubling a capillary's
#: between the two laws -- so a perturbation that moved one of them would
#: report that change as its effect. A run picks one blood model for all of its
#: arms; comparing two models means two runs.
INCOMPARABLE_OVERRIDES: tuple[str, ...] = (
    "viscosity_law",
    "diameter_basis",
    "haematocrit",
)

#: Types that run a sweep helper and write a sweep CSV rather than one re-solve.
#: Derived from :data:`PERTURBATION_TYPES` by name so a new ``*_sweep`` type is
#: classified without editing a second table.
SWEEP_PERTURBATION_TYPES: frozenset[str] = frozenset(
    name for name in PERTURBATION_TYPES if "sweep" in name
)


def is_sweep_perturbation(perturbation_type: Any) -> bool:
    """Whether *perturbation_type* is a sweep (CSV grid / Alice curves).

    Uses the type name: every declared sweep includes ``sweep``, and non-sweep
    types do not. Unknown names with ``sweep`` in them are treated as sweeps so
    a new type gets the sweep export path (and slider-backed napari layer)
    rather than a single static re-solve layer.
    """
    text = str(perturbation_type)
    if text in SWEEP_PERTURBATION_TYPES:
        return True
    return "sweep" in text and text != "none"


def settings_for_perturbation_type(perturbation_type: Any) -> tuple[str, ...]:
    """The settings a perturbation of *perturbation_type* reads.

    An unknown type reads nothing, so a panel showing a hand-edited config
    shows no options rather than the previous type's.
    """
    return tuple(SETTINGS_FOR_TYPE.get(str(perturbation_type), ()))


def is_usable_as_a_directory_name(name: str) -> bool:
    """Whether *name* names one directory, and not a path to somewhere else.

    A perturbation's output goes in ``{name}_{type}``, so a name carrying a
    separator would write outside the run's output -- and ``..`` would write
    over the run's own files. That is the whole sanitisation: empty, ``.``,
    ``..``, and ``/`` ``\\`` ``:`` are refused; everything else is kept as typed.
    """
    stripped = name.strip()
    if not stripped or stripped in {".", ".."}:
        return False
    return not any(character in stripped for character in ("/", "\\", ":"))


def perturbation_folder_name(name: str, perturbation_type: str) -> str:
    """The subdirectory one perturbation writes into: ``{name}_{type}``.

    Uses the user-facing name and the type string as they are, joined by one
    underscore. The name is already required to be a single path component
    (see :func:`is_usable_as_a_directory_name`); the type is one of
    :data:`PERTURBATION_TYPES` and so is already filesystem-safe. Nothing else
    is rewritten.
    """
    return f"{name}_{perturbation_type}"


def visible_perturbation_settings(specs: Sequence["PerturbationSpec"]) -> set[str]:
    """Every setting the configured perturbations between them read."""
    wanted: set[str] = set()
    for spec in specs:
        wanted.update(settings_for_perturbation_type(spec.type))
    return wanted


def plain(value: Any) -> Any:
    """*value* as builtins, however deeply nested.

    The one boundary between this module and YAML. ``yaml.safe_dump`` refuses a
    ``np.float64`` or a ``Path`` outright, and a perturbation's overrides are
    ordinary settings values -- so one that has been coerced by the schema
    arrives as a ``Path``, and one picked up from a graph as a numpy scalar.
    Paths become forward-slashed strings for the same reason the config writer
    does it: a file written on Windows must match one written on Linux.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [plain(item) for item in value.tolist()]
    if isinstance(value, PurePath):
        return PurePath(value).as_posix()
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


@dataclass(frozen=True)
class PerturbationSpec:
    """One perturbation: what to call it, what kind it is, what it changes.

    Read leniently -- a malformed entry becomes a spec carrying a line in
    :attr:`problems` rather than an exception -- so a panel can report a
    hand-edited config while a run refuses it.
    """

    name: str
    type: str
    overrides: Mapping[str, Any] = field(default_factory=dict)
    #: One line per thing wrong with this entry that needs no schema to see.
    problems: tuple[str, ...] = ()

    @classmethod
    def from_entry(cls, entry: Any, *, index: int = 0) -> "PerturbationSpec":
        """One entry of the `perturbations` list, however badly written."""
        where = f"perturbations[{index}]"
        problems: list[str] = []
        if not isinstance(entry, Mapping):
            return cls(
                name=f"perturbation_{index + 1}",
                type="none",
                problems=(f"{where} is not a name/type/overrides entry: {entry!r}",),
            )

        unknown_keys = sorted(set(entry) - {"name", "type", "overrides"})
        if unknown_keys:
            problems.append(
                f"{where} has unexpected key(s) {unknown_keys}; an entry is "
                "name, type and overrides"
            )

        name = entry.get("name")
        if name is None or not str(name).strip():
            problems.append(f"{where} has no name")
            name = f"perturbation_{index + 1}"
        elif not is_usable_as_a_directory_name(str(name)):
            problems.append(
                f"{where} name {str(name)!r} cannot be a directory name, and a "
                "perturbation's name is used with its type as the directory "
                "its output goes in"
            )

        perturbation_type = entry.get("type")
        if perturbation_type is None:
            problems.append(
                f"{where} ('{name}') has no type; one of "
                f"{', '.join(PERTURBATION_TYPES)}"
            )
            perturbation_type = "none"
        elif str(perturbation_type) not in PERTURBATION_TYPES:
            problems.append(
                f"{where} ('{name}') has unknown type {perturbation_type!r}; one "
                f"of {', '.join(PERTURBATION_TYPES)}"
            )

        overrides = entry.get("overrides") or {}
        if not isinstance(overrides, Mapping):
            problems.append(
                f"{where} ('{name}') overrides is not a mapping of setting name "
                f"to value: {overrides!r}"
            )
            overrides = {}

        return cls(
            name=str(name),
            type=str(perturbation_type),
            overrides={str(key): value for key, value in overrides.items()},
            problems=tuple(problems),
        )

    @property
    def folder_name(self) -> str:
        """The subdirectory this perturbation writes into: ``{name}_{type}``."""
        return perturbation_folder_name(self.name, self.type)

    def to_entry(self) -> dict[str, Any]:
        """This perturbation as the entry a config file holds."""
        return {
            "name": str(self.name),
            "type": str(self.type),
            "overrides": {key: plain(value) for key, value in self.overrides.items()},
        }

    def unused_overrides(self) -> tuple[str, ...]:
        """Override keys this type does not read, so nothing would apply them.

        The refused keys are left out: they get an error of their own from
        :meth:`incomparable_overrides` rather than a second, milder line saying
        the same thing.
        """
        reads = set(settings_for_perturbation_type(self.type))
        return tuple(
            sorted(
                key
                for key in self.overrides
                if key not in reads and key not in INCOMPARABLE_OVERRIDES
            )
        )

    def incomparable_overrides(self) -> tuple[str, ...]:
        """Override keys that would make this perturbation's numbers a lie.

        See :data:`INCOMPARABLE_OVERRIDES`. Refused rather than warned about:
        the difference this perturbation reports is what a reader takes from
        its CSV months later, and a log line does not travel with the file.
        """
        return tuple(sorted(key for key in self.overrides if key in INCOMPARABLE_OVERRIDES))

    def schema_problems(self, schema) -> tuple[str, ...]:
        """What only the schema can see: an unknown key, or a value it rejects.

        Each override is coerced **on its own** rather than through
        ``Schema.validate``: a perturbation is a partial config, so its
        prerequisites are met by the run it is applied to and not by itself, and
        validating the fragment would raise `IneffectiveSettingWarning` for
        every override whose bool prerequisite lives outside it. This is the
        same reason `examples/pipeline_presets.py` checks its presets one value
        at a time.
        """
        problems: list[str] = []
        for key, value in self.overrides.items():
            try:
                setting = schema[key]
            except Exception as exc:  # ConfigError, with a spelling suggestion
                problems.append(f"perturbation '{self.name}' sets {exc}")
                continue
            try:
                setting.coerce(value)
            except Exception as exc:
                problems.append(f"perturbation '{self.name}': {exc}")
        return tuple(problems)

    def coerced_overrides(self, schema) -> dict[str, Any]:
        """The overrides this perturbation applies, as the schema's own types.

        Anything the schema rejects is left out -- `schema_problems` is what
        reports it -- so a caller gets the part of a bad entry that does work
        rather than nothing at all.
        """
        coerced: dict[str, Any] = {}
        for key, value in self.overrides.items():
            if key not in schema:
                continue
            try:
                coerced[key] = schema[key].coerce(value)
            except Exception:
                continue
        return coerced

    def applied_overrides(self, schema) -> dict[str, Any]:
        """The overrides a run actually puts on top of its settings.

        Only what this type reads: every perturbation re-solves the network, so
        a settings dict carrying an unread override -- `inlet_p_bc` on an
        arteriole dilation, say -- would change the answer while
        :meth:`unused_overrides` reported it as having no effect. Filtering here
        makes that report true, and keeps a perturbation to the one thing it
        says it is.
        """
        reads = set(settings_for_perturbation_type(self.type))
        return {
            key: value
            for key, value in self.coerced_overrides(schema).items()
            if key in reads
        }


def perturbations_from_settings(values: Mapping[str, Any]) -> tuple[PerturbationSpec, ...]:
    """Every perturbation the `perturbations` setting describes, in order."""
    raw = values.get("perturbations")
    if raw is None:
        return ()
    if isinstance(raw, Mapping):
        entries: list[Any] = [raw]
    elif isinstance(raw, (str, bytes)):
        # A single string cannot be an entry, and iterating one would report a
        # problem per character.
        return (
            PerturbationSpec(
                name="perturbation_1",
                type="none",
                problems=(f"perturbations is not a list of entries: {raw!r}",),
            ),
        )
    else:
        try:
            entries = list(raw)
        except TypeError:
            return (
                PerturbationSpec(
                    name="perturbation_1",
                    type="none",
                    problems=(f"perturbations is not a list of entries: {raw!r}",),
                ),
            )
    return tuple(
        PerturbationSpec.from_entry(entry, index=index)
        for index, entry in enumerate(entries)
    )


def perturbations_to_settings(
    specs: Sequence[PerturbationSpec],
) -> dict[str, list[dict[str, Any]]]:
    """The `perturbations` setting these specs describe, as plain YAML data."""
    return {"perturbations": [spec.to_entry() for spec in specs]}


def perturbation_problems(values: Mapping[str, Any], schema) -> tuple[str, ...]:
    """Everything wrong with the configured perturbations, as lines.

    What both the panel's report box and the pre-run checks read; neither
    raises, so they say the same thing in the two places a user meets it.
    """
    specs = perturbations_from_settings(values)
    problems: list[str] = []
    seen: dict[str, int] = {}
    for index, spec in enumerate(specs):
        problems.extend(spec.problems)
        problems.extend(spec.schema_problems(schema))
        refused = spec.incomparable_overrides()
        if refused:
            problems.append(
                f"perturbation '{spec.name}' sets {list(refused)}, which a "
                "perturbation may not change: it would move every resistance in "
                "the network, so the difference against the baseline would not "
                "be this perturbation's effect. Fix: set it for the whole run, "
                "or make it a second run"
            )
        first = seen.get(spec.name)
        if first is not None:
            problems.append(
                f"perturbations[{index}] repeats the name '{spec.name}' from "
                f"perturbations[{first}]; each one names its own output"
            )
        else:
            seen[spec.name] = index
    return tuple(problems)


def perturbation_output_dir(values: Mapping[str, Any]) -> Path:
    """Where perturbation output goes, configured or derived.

    Unset means the run's other output directory, which is what
    `vtk_output_prefix` names the parent of -- so a perturbation lands with
    the run it perturbs without anyone configuring a second path. Each
    perturbation still writes into a ``{name}_{type}`` subfolder of this root.
    """
    configured = values.get("perturbation_output_dir")
    if configured:
        return Path(configured)
    prefix = values.get("vtk_output_prefix")
    return Path(prefix).parent if prefix else Path(".")
