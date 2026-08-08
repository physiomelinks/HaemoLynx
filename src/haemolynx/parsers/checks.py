"""Pre-run checks derived from the schema.

Everything a run needs before it starts that the schema can state for itself:
paths that have to be there, and settings that a feature makes mandatory. These
are separate from :meth:`Schema.validate` because they touch the filesystem —
a config file should still load when a path is missing, so a GUI can show the
problem next to the field rather than refusing to open the file.

Checks that depend on how a particular pipeline names its files do not belong
here; see ``haemolynx.pipeline.checks``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schema import Schema, Setting, is_active

#: Loaders accept a zipped file in place of the plain one, so a missing path is
#: not missing if its `.zip` sibling is there.
ZIP_SUFFIX = ".zip"


@dataclass
class CheckReport:
    """The outcome of a set of pre-run checks."""

    passed: list[tuple[str, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_pass(self, label: str, detail: str) -> None:
        self.passed.append((label, detail))

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def extend(self, other: "CheckReport") -> "CheckReport":
        self.passed.extend(other.passed)
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        return self

    def print(self, title: str = "Pre-run checks") -> None:
        print(f"\n=== {title} ===")
        for label, detail in self.passed:
            print(f"[OK] {label}: {detail}")
        for message in self.warnings:
            print(f"[WARN] {message}")
        for message in self.errors:
            print(f"[ERROR] {message}")
        print("Checks passed." if self.ok else f"{len(self.errors)} problem(s) to fix.")


def resolve_existing_path(path: Path | str | None) -> tuple[bool, str]:
    """Whether *path* exists, accepting a ``.zip`` sibling; plus what was tried."""
    if path is None:
        return False, "not set"
    path = Path(path)
    if path.exists():
        return True, str(path)
    zipped = path.with_suffix(path.suffix + ZIP_SUFFIX)
    if zipped.exists():
        return True, f"{zipped} (zipped)"
    return False, f"checked: {path}, {zipped}"


def check_settings(
    schema: Schema, values: Mapping[str, Any], *, skip: Iterable[str] = ()
) -> CheckReport:
    """Check every active setting that declares ``must_exist``.

    A setting is active when its ``requires`` prerequisites are met, so a path
    the run would generate for itself is not demanded up front.
    """
    report = CheckReport()
    skipped = set(skip)
    for setting in schema:
        if setting.name in skipped or not setting.must_exist:
            continue
        if not is_active(setting, values):
            continue
        value = values.get(setting.name)
        if value is None:
            report.add_error(
                f"{setting.help} — '{setting.name}' is not set but is required "
                f"{_because(setting)}."
            )
            continue
        exists, detail = resolve_existing_path(value)
        if exists:
            report.add_pass(setting.name, detail)
        else:
            report.add_error(
                f"{setting.name}: {detail}. Fix: point '{setting.name}' at an "
                f"existing file ({setting.help.lower()})."
            )
    return report


def _because(setting: Setting) -> str:
    if not setting.requires:
        return "for every run"
    parts = [
        f"'{name[1:]}' is off" if name.startswith("!") else f"'{name}' is on"
        for name in setting.requires
    ]
    return "because " + " and ".join(parts)
