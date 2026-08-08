"""The napari panel: a settings form, the pre-run checks, and a run button.

Everything about *what* the form contains lives in :mod:`haemolynx.gui.form`,
which needs no GUI. This module is the part that cannot be tested without a
display: it turns those rows into magicgui widgets and wires the buttons up.

napari and magicgui are imported inside the functions, so importing this module
(and therefore the package) costs nothing when there is no GUI installed.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from haemolynx.gui.form import Field, sections_for, values_from
from haemolynx.parsers import dump_config, load_config
from haemolynx.pipeline import default_schema, preflight, resolve_settings, run_pipeline_stages

logger = logging.getLogger(__name__)


def _build_widget(field: Field):
    """One magicgui widget for one form row."""
    from magicgui.widgets import create_widget

    widget = create_widget(
        value=field.value,
        name=field.name,
        label=field.label,
        widget_type=field.widget_type,
        options=dict(field.options),
    )
    widget.tooltip = field.help
    return widget


def settings_widget():
    """The HaemoLynx panel, built from the settings schema.

    Every row comes from the schema, so a new setting appears here the moment
    it is declared -- there is no second list of settings to keep in step.
    """
    from magicgui.widgets import Container, PushButton, TextEdit

    schema = default_schema()
    grouped = sections_for(schema)
    fields = [field for rows in grouped.values() for field in rows]
    widgets_by_name = {field.name: _build_widget(field) for field in fields}
    fields_by_name = {field.name: field for field in fields}

    def current_values() -> dict[str, Any]:
        return {name: widget.value for name, widget in widgets_by_name.items()}

    def apply_prerequisites(*_args) -> None:
        """Grey out settings whose prerequisite is unmet, and say why."""
        values = current_values()
        for name, widget in widgets_by_name.items():
            field = fields_by_name[name]
            enabled = field.is_enabled(values)
            widget.enabled = enabled
            widget.tooltip = field.help if enabled else field.why_disabled(values)

    for widget in widgets_by_name.values():
        widget.changed.connect(apply_prerequisites)
    apply_prerequisites()

    report = TextEdit(label="", value="Ready.")
    report.read_only = True

    load_button = PushButton(text="Load config...")
    save_button = PushButton(text="Save config...")
    check_button = PushButton(text="Run checks")
    run_button = PushButton(text="Run pipeline")

    def on_load() -> None:
        from magicgui.widgets import FileEdit

        chooser = FileEdit(mode="r", filter="*.yaml")
        chooser.show()
        path = Path(str(chooser.value))
        if not path.is_file():
            return
        for name, value in load_config(path, schema).items():
            if name in widgets_by_name:
                widgets_by_name[name].value = value
        apply_prerequisites()
        report.value = f"Loaded {path}"

    def on_save() -> None:
        from magicgui.widgets import FileEdit

        chooser = FileEdit(mode="w", filter="*.yaml")
        chooser.show()
        path = Path(str(chooser.value))
        if not path.name:
            return
        dump_config(path, schema, values=current_values())
        report.value = f"Wrote {path}"

    def on_check() -> None:
        settings = resolve_settings(current_values(), schema=schema, config_path=None)
        result = preflight(settings, schema)
        lines = [f"FAILED: {message}" for message in result.errors]
        lines += [f"warning: {message}" for message in result.warnings]
        report.value = "\n".join(lines) if lines else "All checks passed."

    def on_run() -> None:
        from napari.qt.threading import thread_worker

        settings = resolve_settings(current_values(), schema=schema, config_path=None)
        if not preflight(settings, schema).ok:
            report.value = "Checks failed; nothing was run. Press 'Run checks' for detail."
            return

        @thread_worker
        def run() -> Any:
            return run_pipeline_stages(settings, schema)

        def finished(graph) -> None:
            if graph is None:
                report.value = "Finished, but the run produced no graph."
                return
            report.value = (
                f"Finished: {graph.number_of_nodes()} nodes, "
                f"{graph.number_of_edges()} vessels."
            )

        def failed(error: Exception) -> None:
            report.value = f"{type(error).__name__}: {error}"
            logger.exception("pipeline run failed", exc_info=error)

        worker = run()
        worker.returned.connect(finished)
        worker.errored.connect(failed)
        report.value = "Running..."
        worker.start()

    load_button.changed.connect(on_load)
    save_button.changed.connect(on_save)
    check_button.changed.connect(on_check)
    run_button.changed.connect(on_run)

    return Container(
        widgets=[
            *widgets_by_name.values(),
            load_button,
            save_button,
            check_button,
            run_button,
            report,
        ],
        labels=True,
        scrollable=True,
    )
