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
from haemolynx.gui.layers import input_for_layer
from haemolynx.gui.tabs import tabs_for
from haemolynx.parsers import dump_config, load_config
from haemolynx.pipeline import default_schema, preflight, resolve_settings, run_pipeline_stages

logger = logging.getLogger(__name__)


def _create_widget(**kwargs):
    """magicgui's `create_widget`, imported only when a panel is built."""
    from magicgui.widgets import create_widget

    return create_widget(**kwargs)


def _export_dir(values: dict[str, Any]) -> Path:
    """Where a layer with no file behind it gets written: beside the outputs."""
    prefix = values.get("vtk_output_prefix")
    return Path(prefix).parent if prefix else Path.cwd()


def _build_row(field: Field):
    """One magicgui widget for one form row."""
    widget = _create_widget(
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


def settings_widget(napari_viewer=None):
    """The HaemoLynx panel: the pipeline's stages, in the order it runs them.

    The panel can run on a layer that is already open, which needs the viewer.
    napari injects that only into a class -- `_get_widget_viewer_param` returns
    nothing for a plain function -- and defining a QWidget subclass would mean
    importing Qt when this module is imported, which the library must not do.
    So the viewer is asked for instead: `napari.current_viewer()` is the one
    building this panel. Pass *napari_viewer* to override that, as a test or a
    script would.
    """
    import napari
    from magicgui.widgets import Container, Label, PushButton, TextEdit
    from qtpy.QtWidgets import QScrollArea, QTabWidget, QVBoxLayout, QWidget

    viewer = napari_viewer if napari_viewer is not None else napari.current_viewer()

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
        )
        # A plain QScrollArea rather than `Container(scrollable=True)`: the
        # magicgui one reports the full height of its contents, so a tab with
        # 39 rows stretches the whole napari window instead of scrolling.
        # QScrollArea's own size hint ignores the widget inside it, which is
        # exactly what keeps the panel a sensible size. Its scrollbars default
        # to appearing only when needed, vertically and horizontally.
        scroller = QScrollArea()
        scroller.setWidgetResizable(True)
        scroller.setWidget(page.native)
        tab_widget.addTab(scroller, tab.stage.title)
        if tab.stage.call:
            index = tab_widget.count() - 1
            tab_widget.setTabToolTip(index, f"{tab.stage.call}(settings, ...)")

    def current_values() -> dict[str, Any]:
        """What the panel says, in the settings' own terms.

        Read back through the field rather than straight off the widget: an
        empty picker is *unset*, not the working directory, and an empty box is
        unset rather than zero.
        """
        return {
            name: fields[name].to_setting_value(widget.value)
            for name, widget in rows.items()
        }

    def use_layer(layer) -> None:
        """Point the run at *layer*: its own file, or its array written out."""
        try:
            chosen = input_for_layer(layer, _export_dir(current_values()))
        except ValueError as error:
            report.value = str(error)
            return
        if chosen.needs_export:
            import tifffile

            target = Path(chosen.settings["input_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(target, layer.data)
        for name, value in chosen.settings.items():
            if name in rows:
                rows[name].value = value
        apply_prerequisites()
        report.value = chosen.note

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

    layer_row: Any = None
    if viewer is not None:
        layer_picker = _create_widget(
            annotation=napari.layers.Image, label="Use open layer"
        )
        use_button = PushButton(text="Use this layer as the input")
        layer_row = Container(
            widgets=[layer_picker, use_button], layout="horizontal", labels=True
        )

        applying = False

        def adopt(layer) -> None:
            """Point the run at *layer*, without re-entering through `changed`."""
            nonlocal applying
            if applying or not isinstance(layer, napari.layers.Image):
                return
            applying = True
            try:
                if layer_picker.value is not layer:
                    layer_picker.value = layer
                use_layer(layer)
            finally:
                applying = False

        def on_layer_added(event) -> None:
            """A dropped image becomes the input, panel already open or not."""
            adopt(getattr(event, "value", None))

        # Both orders have to work: open the panel with an image already there,
        # or drop one in while it is open.
        layer_picker.changed.connect(lambda *_: adopt(layer_picker.value))
        use_button.changed.connect(lambda *_: use_layer(layer_picker.value))
        viewer.layers.events.inserted.connect(on_layer_added)

        images = [
            layer for layer in viewer.layers if isinstance(layer, napari.layers.Image)
        ]
        active = viewer.layers.selection.active
        if isinstance(active, napari.layers.Image):
            adopt(active)
        elif images:
            adopt(images[-1])

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
    if layer_row is not None:
        layout.addWidget(layer_row.native)
    layout.addWidget(tab_widget)
    layout.addWidget(buttons.native)
    layout.addWidget(report.native)
    return panel
