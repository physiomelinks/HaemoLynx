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
from dataclasses import dataclass, field
from pathlib import Path
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


class ConfigError(ValueError):
    """A config value is missing, unknown, or fails its declared constraints."""


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
        Names of boolean settings that must all be true for this one to apply.
        A GUI greys the control out when they are not; validation rejects a
        non-default value whose prerequisites are off, which is how the silent
        "setting had no effect" class of bug is caught.
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
    advanced: bool = False

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
            "advanced": self.advanced,
        }


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
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

    def defaults(self) -> dict[str, Any]:
        """Defaults, coerced exactly like user-supplied values.

        Coercing here rather than storing raw literals means a path default is
        a ``Path`` whether or not the user overrode it, and each call returns
        fresh containers rather than sharing one mutable default between runs.
        """
        return {
            setting.name: setting.coerce(setting.default) for setting in self.settings
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

        for setting in self.settings:
            if setting.name not in resolved:
                continue
            value = resolved[setting.name]
            if not setting.requires:
                continue
            if value == setting.default or value is None:
                continue
            unmet = [
                prerequisite
                for prerequisite in setting.requires
                if not resolved.get(prerequisite, False)
            ]
            if unmet:
                errors.append(
                    f"Setting '{setting.name}' is set to {value!r} but has no effect "
                    f"while {' and '.join(repr(u) for u in unmet)} "
                    f"{'is' if len(unmet) == 1 else 'are'} false. Enable "
                    f"{' and '.join(unmet)}, or leave '{setting.name}' at its default."
                )

        if errors:
            raise ConfigError(
                f"{len(errors)} configuration problem"
                f"{'' if len(errors) == 1 else 's'}:\n  - "
                + "\n  - ".join(errors)
            )
        return resolved

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
