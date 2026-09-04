"""Declarative description of a pipeline's settings.

One :class:`Setting` per configurable value, carrying everything three
different front-ends need from a single declaration:

* the **YAML config file** — key, default, and the comment written above it
* the **command line** — flag name, type coercion, choices
* a **GUI** — section grouping, widget type, bounds, units, and which controls
  to disable when another setting is off

Nothing here imports the pipeline, so a schema can be introspected without
loading numpy, ilastik, or any image data — a GUI can render the form from
``Schema.describe()`` alone.
"""
from __future__ import annotations

import difflib
import warnings
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any, Iterable, Iterator, Mapping, Sequence

#: Widget/coercion kinds. Kept as strings rather than Python types so that
#: ``Schema.describe()`` is directly JSON-serialisable for a GUI.
KINDS = (
    "bool",
    "int",
    "float",
    "str",
    "path",
    "choice",
    "int_list",
    "float_list",
    "str_list",
    "mapping",
    "any",
)

_LIST_KINDS = {"int_list": int, "float_list": float, "str_list": str}


def is_prerequisite_met(prerequisite: str, values: Mapping[str, Any]) -> bool:
    """True when *prerequisite* holds in *values*.

    A leading ``!`` negates it, which is how a setting says it applies only
    while some feature is *off* — an input path that a run would otherwise
    generate for itself, for instance.
    """
    if prerequisite.startswith("!"):
        return not bool(values.get(prerequisite[1:], False))
    return bool(values.get(prerequisite, False))


def is_active(setting: "Setting", values: Mapping[str, Any]) -> bool:
    """True when every one of *setting*'s prerequisites is met."""
    return all(is_prerequisite_met(p, values) for p in setting.requires)


def section_key(section: str) -> str:
    """YAML heading for a section name."""
    return section.strip().lower().replace(" ", "_").replace("-", "_")


class ConfigError(ValueError):
    """A config value is missing, unknown, or fails its declared constraints."""


class IneffectiveSettingWarning(UserWarning):
    """A setting carries a non-default value that nothing will read.

    Not an error: leaving a path filled in while its feature is switched off is
    ordinary practice, and a config file that documents every setting should
    still load. It is worth saying out loud, though, because "I changed it and
    nothing happened" is otherwise silent.
    """


@dataclass(frozen=True)
class Setting:
    """One configurable value.

    Parameters
    ----------
    name:
        Key as it appears in the YAML file, ``lower_snake_case``.
    kind:
        One of :data:`KINDS`. Drives coercion, validation and widget choice.
    default:
        Value used when the key is absent. ``None`` means "unset"; combine with
        ``required=True`` to force the user to supply one.
    help:
        One sentence, imperative, no trailing full stop. Becomes the YAML
        comment and the GUI tooltip, so it must read well out of context.
    section:
        Grouping for both the YAML file and the GUI form.
    choices:
        Allowed values for ``kind="choice"``.
    minimum, maximum:
        Inclusive numeric bounds. A GUI can render these as a slider.
    unit:
        Physical unit shown next to the widget, e.g. ``"um"``.
    requires:
        Names of boolean settings that must all be true for this one to apply,
        each optionally negated with a leading ``!``. A GUI greys the control
        out when they are not met; a non-default value whose prerequisites are
        unmet warns, which is how the silent "setting had no effect" class of
        bug is caught.
    must_exist:
        For a path: the file or directory it names must be there before a run
        starts, whenever this setting is active. Checked by
        :func:`haemolynx.parsers.check_settings`, not at load, so a GUI can
        show the problem rather than refuse to open the file.
    advanced:
        Hide behind an "advanced" disclosure by default.
    """

    name: str
    kind: str
    default: Any
    help: str
    section: str
    choices: tuple[Any, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None
    unit: str | None = None
    requires: tuple[str, ...] = ()
    must_exist: bool = False
    advanced: bool = False
    #: Hint text a GUI shows in an empty (unset/``None``-default) box, e.g.
    #: ``"auto"`` for a setting whose blank value means "compute it from
    #: something else". Cosmetic only -- never becomes the value.
    placeholder: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ConfigError(
                f"Setting '{self.name}' has unknown kind '{self.kind}'. "
                f"Valid kinds: {', '.join(KINDS)}."
            )
        if not self.help:
            raise ConfigError(f"Setting '{self.name}' needs a help string.")
        if self.kind == "choice" and not self.choices:
            raise ConfigError(f"Choice setting '{self.name}' needs choices.")
        if self.choices is not None and self.kind != "choice":
            raise ConfigError(
                f"Setting '{self.name}' declares choices but is kind '{self.kind}'."
            )
        if self.kind not in {"int", "float"} and (
            self.minimum is not None or self.maximum is not None
        ):
            raise ConfigError(
                f"Setting '{self.name}' declares bounds but is kind '{self.kind}'."
            )
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ConfigError(
                f"Setting '{self.name}' has minimum {self.minimum} above maximum "
                f"{self.maximum}."
            )
        if self.must_exist and self.kind != "path":
            raise ConfigError(
                f"Setting '{self.name}' declares must_exist but is kind "
                f"'{self.kind}'; only a path can be checked for existence."
            )
        if self.placeholder is not None and self.default is not None:
            raise ConfigError(
                f"Setting '{self.name}' declares a placeholder but has a "
                f"non-None default; placeholder text is only shown in an "
                f"empty (unset) box."
            )
        if self.default is not None:
            # A default that its own rules reject is a schema bug, not a user
            # error, so surface it the moment the schema is imported.
            self.coerce(self.default)

    def coerce(self, value: Any) -> Any:
        """Return *value* converted to this setting's kind, or raise ConfigError."""
        if value is None:
            return None
        try:
            return self._coerce(value)
        except ConfigError:
            raise
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"Setting '{self.name}' expects {self.kind}, got {value!r} "
                f"({type(value).__name__}): {exc}"
            ) from exc

    def _coerce(self, value: Any) -> Any:
        kind = self.kind
        if kind == "any":
            return value
        if kind == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.strip().lower() in {
                "true", "false", "yes", "no", "1", "0", "on", "off",
            }:
                return value.strip().lower() in {"true", "yes", "1", "on"}
            raise ConfigError(
                f"Setting '{self.name}' expects true or false, got {value!r}."
            )
        if kind == "int":
            if isinstance(value, bool):
                raise ConfigError(f"Setting '{self.name}' expects an int, got a bool.")
            coerced = int(value)
            return self._check_bounds(coerced)
        if kind == "float":
            if isinstance(value, bool):
                raise ConfigError(f"Setting '{self.name}' expects a float, got a bool.")
            return self._check_bounds(float(value))
        if kind == "str":
            return str(value)
        if kind == "path":
            return Path(value).expanduser()
        if kind == "choice":
            assert self.choices is not None
            if value not in self.choices:
                allowed = ", ".join(repr(c) for c in self.choices)
                raise ConfigError(
                    f"Setting '{self.name}' got {value!r}; allowed values are {allowed}."
                )
            return value
        if kind in _LIST_KINDS:
            item_type = _LIST_KINDS[kind]
            if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
                raise ConfigError(
                    f"Setting '{self.name}' expects a list, got {value!r}."
                )
            return tuple(item_type(item) for item in value)
        if kind == "mapping":
            if not isinstance(value, Mapping):
                raise ConfigError(
                    f"Setting '{self.name}' expects a mapping, got {value!r}."
                )
            return dict(value)
        raise ConfigError(f"Setting '{self.name}' has unhandled kind '{kind}'.")

    def _check_bounds(self, value: float) -> float:
        if self.minimum is not None and value < self.minimum:
            raise ConfigError(
                f"Setting '{self.name}' is {value}, below its minimum {self.minimum}"
                f"{f' {self.unit}' if self.unit else ''}."
            )
        if self.maximum is not None and value > self.maximum:
            raise ConfigError(
                f"Setting '{self.name}' is {value}, above its maximum {self.maximum}"
                f"{f' {self.unit}' if self.unit else ''}."
            )
        return value

    def describe(self) -> dict[str, Any]:
        """JSON-friendly description, for building a GUI form."""
        return {
            "name": self.name,
            "kind": self.kind,
            "default": _jsonify(self.default),
            "help": self.help,
            "section": self.section,
            "choices": list(self.choices) if self.choices else None,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "unit": self.unit,
            "requires": list(self.requires),
            "must_exist": self.must_exist,
            "advanced": self.advanced,
            "placeholder": self.placeholder,
        }


def _copy_container(value: Any) -> Any:
    """Fresh list/dict so one run cannot mutate the next run's default."""
    if isinstance(value, list):
        return [_copy_container(item) for item in value]
    if isinstance(value, dict):
        return {key: _copy_container(item) for key, item in value.items()}
    return value


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        # Forward slashes on every platform, so a config generated on Windows
        # is byte-for-byte the one generated on Linux. `str()` here emitted
        # `examples\images\...` on Windows, which meant `regenerate_configs.py`
        # could not be run there without rewriting every committed config and
        # breaking CI. Windows accepts forward slashes, and `pathlib` reads
        # them back on either platform, so nothing is lost -- including the
        # drive letter of an absolute path, which becomes `C:/Users/...`.
        return PurePath(value).as_posix()
    if isinstance(value, tuple):
        return [_jsonify(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    return value


@dataclass(frozen=True)
class Schema:
    """An ordered collection of :class:`Setting`, grouped into sections."""

    settings: tuple[Setting, ...]
    title: str = ""
    description: str = ""
    _by_name: dict[str, Setting] = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        settings: Sequence[Setting],
        *,
        title: str = "",
        description: str = "",
    ) -> None:
        settings = tuple(settings)
        by_name: dict[str, Setting] = {}
        for setting in settings:
            if setting.name in by_name:
                raise ConfigError(f"Duplicate setting '{setting.name}' in schema.")
            by_name[setting.name] = setting
        for setting in settings:
            for prerequisite in setting.requires:
                prerequisite = prerequisite.lstrip("!")
                if prerequisite not in by_name:
                    raise ConfigError(
                        f"Setting '{setting.name}' requires '{prerequisite}', "
                        "which is not in the schema."
                    )
                if by_name[prerequisite].kind != "bool":
                    raise ConfigError(
                        f"Setting '{setting.name}' requires '{prerequisite}', "
                        "which is not a bool."
                    )
        for section in {setting.section for setting in settings}:
            slug = section_key(section)
            if slug in by_name:
                raise ConfigError(
                    f"Section '{section}' becomes YAML key '{slug}', which is also "
                    f"a setting name. Rename one of them, or the section heading "
                    "and the setting cannot be told apart when the file is read."
                )
        object.__setattr__(self, "settings", settings)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "_by_name", by_name)

    def __iter__(self) -> Iterator[Setting]:
        return iter(self.settings)

    def __len__(self) -> int:
        return len(self.settings)

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def __getitem__(self, name: str) -> Setting:
        try:
            return self._by_name[name]
        except KeyError:
            raise ConfigError(self._unknown_key_message(name)) from None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._by_name)

    def sections(self) -> dict[str, tuple[Setting, ...]]:
        """Settings grouped by section, preserving declaration order."""
        grouped: dict[str, list[Setting]] = {}
        for setting in self.settings:
            grouped.setdefault(setting.section, []).append(setting)
        return {name: tuple(items) for name, items in grouped.items()}

    def section_names(self, section: str) -> tuple[str, ...]:
        """Names of the settings in *section*, by title or by its YAML key."""
        wanted = section_key(section)
        for name, settings in self.sections().items():
            if name == section or section_key(name) == wanted:
                return tuple(setting.name for setting in settings)
        known = ", ".join(sorted(self.sections()))
        raise ConfigError(f"Unknown section '{section}'. Sections are: {known}.")

    def section_values(self, values: Mapping[str, Any], section: str) -> dict[str, Any]:
        """The part of *values* belonging to *section*.

        Lets a caller hand one group of settings to the stage that consumes it,
        instead of naming each one, and keeps the grouping identical to the one
        the config file and the GUI show.
        """
        return {
            name: values[name]
            for name in self.section_names(section)
            if name in values
        }

    def defaults(self) -> dict[str, Any]:
        """Defaults, coerced exactly like user-supplied values.

        Coercing here rather than storing raw literals means a path default is
        a ``Path`` whether or not the user overrode it, and each call returns
        fresh containers rather than sharing one mutable default between runs.
        """
        return {
            setting.name: _copy_container(setting.coerce(setting.default))
            for setting in self.settings
        }

    def _unknown_key_message(self, name: object) -> str:
        suggestions = difflib.get_close_matches(str(name), self.names, n=3, cutoff=0.6)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        return f"Unknown setting '{name}'.{hint}"

    def validate(
        self,
        values: Mapping[str, Any],
        *,
        fill_defaults: bool = True,
    ) -> dict[str, Any]:
        """Coerce and check *values*, returning a new dict.

        Rejects unknown keys (with a spelling suggestion), coerces every value
        to its declared kind, enforces choices and bounds, and reports a value
        whose ``requires`` prerequisites are switched off rather than letting it
        be silently ignored. Every problem is collected before raising, so one
        run of the checker tells the user everything that is wrong.
        """
        errors: list[str] = []
        resolved: dict[str, Any] = self.defaults() if fill_defaults else {}

        for key, value in values.items():
            if key not in self._by_name:
                errors.append(self._unknown_key_message(key))
                continue
            try:
                resolved[key] = self._by_name[key].coerce(value)
            except ConfigError as exc:
                errors.append(str(exc))

        for message in self._ineffective_messages(resolved):
            warnings.warn(message, IneffectiveSettingWarning, stacklevel=3)

        if errors:
            raise ConfigError(
                f"{len(errors)} configuration problem"
                f"{'' if len(errors) == 1 else 's'}:\n  - "
                + "\n  - ".join(errors)
            )
        return resolved

    def _ineffective_messages(self, resolved: Mapping[str, Any]) -> list[str]:
        messages: list[str] = []
        for setting in self.settings:
            if setting.name not in resolved or not setting.requires:
                continue
            value = resolved[setting.name]
            if value is None or value == setting.coerce(setting.default):
                continue
            unmet = [
                prerequisite
                for prerequisite in setting.requires
                if not is_prerequisite_met(prerequisite, resolved)
            ]
            if unmet:
                messages.append(
                    f"Setting '{setting.name}' is set to {value!r} but nothing will "
                    f"read it while {' and '.join(repr(u) for u in unmet)} "
                    f"{'is' if len(unmet) == 1 else 'are'} false."
                )
        return messages

    def ineffective_settings(self, values: Mapping[str, Any]) -> list[str]:
        """Settings carrying a value that nothing will read, as messages.

        A GUI greys these controls out instead; this is the text form for a
        command-line run or a preflight report.
        """
        return self._ineffective_messages(values)

    def describe(self) -> dict[str, Any]:
        """Whole-schema JSON description: the contract a GUI renders from."""
        return {
            "title": self.title,
            "description": self.description,
            "sections": [
                {
                    "name": section,
                    "settings": [setting.describe() for setting in items],
                }
                for section, items in self.sections().items()
            ],
        }

    def subset(self, names: Iterable[str]) -> "Schema":
        """A schema holding only *names*, for passing one section to a function."""
        wanted = list(names)
        return Schema(
            [self[name] for name in wanted],
            title=self.title,
            description=self.description,
        )
