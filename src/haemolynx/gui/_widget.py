"""The napari panel: one tab per pipeline stage, then the run buttons.

What each tab contains lives in :mod:`haemolynx.gui.tabs`, and what each row
looks like in :mod:`haemolynx.gui.form`; neither needs a GUI, which is where
the testable logic is. This module is the Qt layer: it turns those rows into
magicgui widgets, stacks them into tabs, and wires the buttons.

napari, magicgui and Qt are imported inside the functions, so importing this
module -- and therefore the package -- costs nothing without a GUI installed.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from haemolynx.gui.form import Field
from haemolynx.gui.tabs import tabs_for
from haemolynx.parsers import dump_config, load_config
from haemolynx.pipeline import default_schema, preflight, resolve_settings, run_pipeline_stages

logger = logging.getLogger(__name__)


def _build_row(field: Field):
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


def _run_in_background(settings, schema, report, button):
    """Run the pipeline off the GUI thread, reporting back when it lands."""
    from napari.qt.threading import thread_worker

    @thread_worker
    def run():
        return run_pipeline_stages(settings, schema)

    def finished(graph) -> None:
        button.enabled = True
        if graph is None:
            report.value = "Finished, but the run produced no graph."
            return
        report.value = (
            f"Finished: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} vessels."
        )

    def failed(error: Exception) -> None:
        button.enabled = True
        report.value = f"{type(error).__name__}: {error}"
        logger.exception("pipeline run failed", exc_info=error)

    worker = run()
    worker.returned.connect(finished)
    worker.errored.connect(failed)
    button.enabled = False
    report.value = "Running..."
    worker.start()


def run_config_widget():
    """Run a config file as it stands, without opening the settings form.

    The same thing `python examples/resistance_network_pipeline.py --config
    my.yaml` does, for a config that is already how you want it.
    """
    from magicgui.widgets import Container, FileEdit, Label, PushButton, TextEdit

    schema = default_schema()
    chooser = FileEdit(mode="r", filter="*.yaml *.yml", label="Config file")
    report = TextEdit(value="Choose a config file, then press Run.")
    report.read_only = True
    run_button = PushButton(text="Run")

    def on_run() -> None:
        path = Path(str(chooser.value))
        if not path.is_file():
            report.value = f"Not a file: {path}"
            return
        try:
            settings = resolve_settings(schema=schema, config_path=path)
        except Exception as error:  # noqa: BLE001 - shown to the user, not raised
            report.value = f"{type(error).__name__}: {error}"
            return
        result = preflight(settings, schema)
        if not result.ok:
            report.value = "\n".join(f"FAILED: {message}" for message in result.errors)
            return
        _run_in_background(settings, schema, report, run_button)

    run_button.changed.connect(on_run)
    return Container(
        widgets=[
            Label(value="Run a config file exactly as it stands."),
            chooser,
            run_button,
            report,
        ],
        labels=True,
    )


def settings_widget():
    """The HaemoLynx panel: the pipeline's stages, in the order it runs them."""
    from magicgui.widgets import Container, Label, PushButton, TextEdit
    from qtpy.QtWidgets import QTabWidget, QVBoxLayout, QWidget

    schema = default_schema()
    tabs = tabs_for(schema)

    rows: dict[str, Any] = {}
    fields: dict[str, Field] = {}
    tab_widget = QTabWidget()
    for tab in tabs:
        for field in tab.fields:
            rows[field.name] = _build_row(field)
            fields[field.name] = field
        summary = Label(value=tab.stage.summary)
        page = Container(
            widgets=[summary, *(rows[field.name] for field in tab.fields)],
            labels=True,
            scrollable=True,
        )
        tab_widget.addTab(page.native, tab.stage.title)
        if tab.stage.call:
            index = tab_widget.count() - 1
            tab_widget.setTabToolTip(index, f"{tab.stage.call}(settings, ...)")

    def current_values() -> dict[str, Any]:
        return {name: widget.value for name, widget in rows.items()}

    def apply_prerequisites(*_args) -> None:
        """Grey out settings whose prerequisite is unmet, and say why."""
        values = current_values()
        for name, widget in rows.items():
            field = fields[name]
            enabled = field.is_enabled(values)
            widget.enabled = enabled
            widget.tooltip = field.help if enabled else field.why_disabled(values)

    for widget in rows.values():
        widget.changed.connect(apply_prerequisites)
    apply_prerequisites()

    report = TextEdit(value="Ready.")
    report.read_only = True

    load_button = PushButton(text="Load config...")
    save_button = PushButton(text="Save config...")
    check_button = PushButton(text="Run checks")
    run_button = PushButton(text="Run pipeline")

    def _settings() -> dict[str, Any]:
        return resolve_settings(current_values(), schema=schema, config_path=None)

    def on_load() -> None:
        from qtpy.QtWidgets import QFileDialog

        path, _filter = QFileDialog.getOpenFileName(
            None, "Open a HaemoLynx config", "", "YAML (*.yaml *.yml)"
        )
        if not path:
            return
        for name, value in load_config(Path(path), schema).items():
            if name in rows:
                rows[name].value = value
        apply_prerequisites()
        report.value = f"Loaded {path}"

    def on_save() -> None:
        from qtpy.QtWidgets import QFileDialog

        path, _filter = QFileDialog.getSaveFileName(
            None, "Save these settings", "config.yaml", "YAML (*.yaml *.yml)"
        )
        if not path:
            return
        dump_config(Path(path), schema, values=current_values())
        report.value = f"Wrote {path}"

    def on_check() -> None:
        result = preflight(_settings(), schema)
        lines = [f"FAILED: {message}" for message in result.errors]
        lines += [f"warning: {message}" for message in result.warnings]
        report.value = "\n".join(lines) if lines else "All checks passed."

    def on_run() -> None:
        settings = _settings()
        if not preflight(settings, schema).ok:
            report.value = "Checks failed; nothing was run. Press 'Run checks' for detail."
            return
        _run_in_background(settings, schema, report, run_button)

    load_button.changed.connect(on_load)
    save_button.changed.connect(on_save)
    check_button.changed.connect(on_check)
    run_button.changed.connect(on_run)

    buttons = Container(
        widgets=[load_button, save_button, check_button, run_button],
        layout="horizontal",
        labels=False,
    )

    panel = QWidget()
    layout = QVBoxLayout(panel)
    layout.addWidget(tab_widget)
    layout.addWidget(buttons.native)
    layout.addWidget(report.native)
    return panel
