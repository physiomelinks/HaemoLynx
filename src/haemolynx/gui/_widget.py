"""The napari panel: one tab per pipeline stage, then the run buttons.

What each tab contains lives in :mod:`haemolynx.gui.tabs`, what each row looks
like in :mod:`haemolynx.gui.form`, and what the progress bars read in
:mod:`haemolynx.gui.progress`; none of the three needs a GUI, which is where
the testable logic is. This module is the Qt layer: it turns those rows into
magicgui widgets, stacks them into tabs, wires the buttons, and moves the bars
as the run reports back.

napari, magicgui and Qt are imported inside the functions and methods here, so
importing this module -- and therefore the package -- costs nothing without a
GUI installed.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from haemolynx.gui.form import Field
from haemolynx.gui.layers import input_for_layer
from haemolynx.gui.results import (
    NODES,
    TEXT_COLUMNS,
    VESSELS,
    ResultLayers,
    colour_cycle_for,
)
from haemolynx.gui.progress import ProgressDisplay
from haemolynx.gui.tabs import tabs_for
from haemolynx.parsers import dump_config, load_config
from haemolynx.pipeline import default_schema, preflight, resolve_settings, run_pipeline_stages
from haemolynx.pipeline.progress import STAGE_STARTED, ProgressEvent

logger = logging.getLogger(__name__)

#: What a layer looks like when the user asks for no colouring at all.
UNCOLOURED = "#cccccc"

#: Settings the panel starts with switched off, because they put a run's
#: results somewhere other than napari: `show_plots_in_ide` makes the 3D graph
#: open in a web browser (plotly's `fig.show()`), and `interactive_plots`
#: blocks on windows of its own. They are ordinary rows, so anyone who wants a
#: browser tab can tick them back on -- they are just the wrong default for a
#: panel whose whole point is that the viewer is already open.
#:
#: `hold_ide_plots_open` is deliberately NOT here: it only does anything while
#: `show_plots_in_ide` is on, so setting it as well earns an
#: IneffectiveSettingWarning on a panel nobody has touched.
DISPLAY_SETTINGS_OFF_IN_NAPARI = {
    "show_plots_in_ide": False,
    "interactive_plots": False,
}


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


class ProgressBars:
    """The panel's two progress bars: one across the stages, one within one.

    What they should read is :class:`haemolynx.gui.progress.ProgressDisplay`,
    which needs no GUI and is tested without one; this only copies that onto
    the widgets. Every method here touches Qt, so all of them must be called on
    the GUI thread -- `_run_in_background` is what gets a run's events there.
    """

    def __init__(self) -> None:
        # Plain QProgressBars rather than magicgui's: these are read, never
        # edited, and a bar needs `setFormat` to name the stage it is on.
        from qtpy.QtWidgets import QProgressBar, QVBoxLayout, QWidget

        self.display = ProgressDisplay()
        self.stage_bar = QProgressBar()
        self.step_bar = QProgressBar()
        self.native = QWidget()
        layout = QVBoxLayout(self.native)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.stage_bar)
        layout.addWidget(self.step_bar)
        self._refresh()

    def start(self) -> None:
        """A run has begun: show an empty bar rather than nothing at all."""
        self.display.start()
        self._refresh()

    def show_event(self, event: ProgressEvent) -> None:
        """One event from the run, already marshalled onto the GUI thread."""
        self.display.update(event)
        self._refresh()

    def finish(self, message: str = "Finished") -> None:
        self.display.finish(message)
        self._refresh()

    def fail(self, message: str = "Failed") -> None:
        self.display.fail(message)
        self._refresh()

    def _refresh(self) -> None:
        for bar, state in (
            (self.stage_bar, self.display.stages),
            (self.step_bar, self.display.steps),
        ):
            # Range 0..0 is Qt's own "no end in sight": the bar animates rather
            # than filling, which is the honest reading for an unknown total.
            bar.setRange(0, state.total)
            bar.setValue(min(state.value, state.total) if state.total else 0)
            bar.setTextVisible(bool(state.text))
            bar.setFormat(state.text or "%p%")
            bar.setVisible(state.visible)


#: Marks a layer as one of ours, so a re-run updates its own work and never a
#: layer the user made that happens to share a name.
OURS = "haemolynx"


def _is_ours(layer) -> bool:
    return bool(getattr(layer, "metadata", {}).get(OURS))


def _apply_layers(viewer, group, report=None) -> None:
    """Put one stage's layers in the viewer. Runs on the GUI thread."""
    for spec in group.layers:
        try:
            _add_or_update(viewer, spec)
        except Exception:  # noqa: BLE001 - one bad layer must not stop the rest
            logger.exception("could not show layer %s", spec.name)
    for name, column in group.recolour:
        layer = viewer.layers[name] if name in viewer.layers else None
        if layer is not None and _is_ours(layer):
            _colour_layer(layer, column)
    if group.ndisplay is not None:
        viewer.dims.ndisplay = group.ndisplay
    if report is not None and group.note:
        report.value = f"{group.title}: {group.note}"


def _colour_layer(layer, column: str | None, kind: str = "continuous",
                  cycle=(), limits=None) -> None:
    """Colour a layer by one of its feature columns."""
    attribute = "edge_color" if layer.__class__.__name__ == "Vectors" else "face_color"
    if column is None:
        # No opinion: a stage that does not name a colouring means "leave what
        # is there", not "blank it". Most stages after build_network have
        # nothing to say about colour, and clearing on each would throw away
        # the previous stage's colouring every time.
        return
    if column in {"", "none"}:
        # An explicit "no colouring", which has to be a real branch: leaving the
        # layer as it was would make picking "none" a control that does nothing.
        setattr(layer, attribute, UNCOLOURED)
        _record_colour(layer, None)
        return
    if column not in getattr(layer, "features", {}):
        return
    # Drop to a flat colour before naming the new column. A layer keeps
    # whichever colour mode the last colouring left it in, and neither mode
    # survives meeting the other kind of column:
    #
    #   cycle mode + a column holding NaN  -> `KeyError: nan`, from
    #     `CategoricalColormap.map`, which decides membership with `np.isin`.
    #     NaN is never equal to itself, so the value is filed under a key that
    #     can never be found again and the next lookup raises.
    #   colormap mode + a text column      -> `TypeError: cannot cast O to
    #     float64`, from trying to interpolate the strings.
    #
    # A run walks straight into the first: the diameters stage colours by
    # `branch_order`, which is text and leaves the layer cycling, and the solve
    # then colours by `flow_abs`, which is full of NaN because `set_edge_flows`
    # skips every edge with no conductance. No user interaction required.
    #
    # Setting the mode instead of the colour does not work -- changing mode
    # re-maps the column that is still active, which is the same crash.
    setattr(layer, attribute, UNCOLOURED)
    if kind == "categorical" and cycle:
        setattr(layer, f"{attribute}_cycle", [colour for _label, colour in cycle])
        setattr(layer, attribute, column)
    else:
        setattr(layer, f"{attribute}_colormap", "viridis")
        setattr(layer, attribute, column)
        if limits is not None:
            _set_contrast_limits(layer, attribute, limits)
    _record_colour(layer, column)


def _set_contrast_limits(layer, attribute: str, limits) -> None:
    """Give the colormap the range of the column it is showing.

    The attribute is not the obvious one. Colours live on `edge_color` and
    `face_color`, but their range lives on `edge_contrast_limits` and
    `face_contrast_limits` -- no `_color`. That matters more than it looks,
    because a napari layer accepts `setattr` of a name it does not have: the
    value lands on a stray attribute, nothing raises, and the real limits keep
    whatever the previous colouring left them at. Colouring by `segment_id`
    (0..9) and then by `flow_abs` (0..1.5e-13) therefore mapped every vessel to
    the bottom of the colormap, and the network came out a single flat colour
    while looking, from the outside, exactly as though it had worked.

    So the name is checked rather than tried: a wrong one is a bug to hear
    about, not a condition to pass over.
    """
    name = f"{attribute.replace('_color', '')}_contrast_limits"
    if not hasattr(layer, name):
        logger.warning(
            "%s has no %s: colouring will use whatever range was set before.",
            type(layer).__name__, name,
        )
        return
    try:
        setattr(layer, name, tuple(float(v) for v in limits))
    except (ValueError, TypeError):
        # A degenerate range (every value identical) is not worth a failure.
        logger.debug("could not set %s to %r", name, limits)


def _record_colour(layer, column: str | None) -> None:
    """Remember what a layer is coloured by, on the layer itself.

    napari keeps the chosen column somewhere different for each layer type and
    each napari version, so asking the layer is fragile. We set the colouring,
    so we can just write down what we set -- and the panel needs it to show the
    user what they are already looking at.
    """
    tag = getattr(layer, "metadata", {}).get(OURS)
    if isinstance(tag, dict):
        tag["colour_by"] = column


def _current_colour(viewer, layer_name: str) -> str:
    """What that layer is coloured by now, as a combo-box value."""
    layers = getattr(viewer, "layers", {})
    if layer_name not in layers:
        return "none"
    tag = getattr(layers[layer_name], "metadata", {}).get(OURS)
    column = tag.get("colour_by") if isinstance(tag, dict) else None
    return column or "none"


def _refresh_colour_choices(viewer, choosers) -> None:
    """Re-offer what the layers can now be coloured by.

    The combo boxes are built before a run, when the only honest answer is
    "none" -- a column cannot be offered until the stage that fills it has run.
    Nothing rebuilt them as stages landed, so `flow_abs` and node `pressure`,
    which arrive at the very last stage, could never be selected: the features
    were on the layers and the dropdown did not know.

    A choice the user has made is kept whenever the column still exists; when
    they have not chosen, the box follows the stage's own default colouring, so
    it always names what is actually on screen.

    "Has the user chosen?" is a flag rather than "is the value still 'none'":
    "none" is itself a choice one can make, and inferring it from the value
    would quietly overrule anyone who picked it at the next stage.
    """
    for layer_name, chooser in choosers:
        if chooser is None or viewer is None:
            continue
        choices = _colour_choices(viewer, layer_name)
        chosen = getattr(chooser, "_haemolynx_chosen", False)
        keep = chooser.value if chosen and chooser.value in choices else None
        if keep is None:
            keep = _current_colour(viewer, layer_name)
        if keep not in choices:
            keep = "none"
        if list(chooser.choices) == choices and chooser.value == keep:
            continue
        with chooser.changed.blocked():
            chooser.choices = choices
            chooser.value = keep


def _add_or_update(viewer, spec) -> None:
    """Add *spec*, or update the layer of ours already carrying its name."""
    import pandas as pd  # noqa: F401  (napari builds features through pandas)

    existing = viewer.layers[spec.name] if spec.name in viewer.layers else None
    if existing is not None and not _is_ours(existing):
        # Someone else's layer happens to share the name. Never overwrite it.
        spec = replace(spec, name=f"{spec.name} (HaemoLynx)")
        existing = viewer.layers[spec.name] if spec.name in viewer.layers else None

    if existing is not None and existing.__class__.__name__.lower() == _CLASS_FOR[spec.kind]:
        existing.data = spec.data
        if spec.features:
            existing.features = dict(spec.features)
        _colour_layer(existing, spec.colour_by, spec.colour_kind,
                      spec.colour_cycle, spec.contrast_limits)
        return

    if existing is not None:
        viewer.layers.remove(existing)

    adder = getattr(viewer, f"add_{spec.kind}")
    options = dict(spec.options)
    if spec.features:
        options["features"] = dict(spec.features)
    layer = adder(spec.data, name=spec.name, scale=spec.scale,
                  visible=spec.visible, metadata={OURS: {"kind": spec.kind}}, **options)
    _colour_layer(layer, spec.colour_by, spec.colour_kind,
                  spec.colour_cycle, spec.contrast_limits)


#: Spec kind -> the napari class name it becomes, for "is this the same sort of
#: layer I already have?".
_CLASS_FOR = {
    "image": "image", "labels": "labels", "points": "points",
    "vectors": "vectors", "shapes": "shapes",
}


def _colour_choices(viewer, layer_name: str) -> list[str]:
    """Columns that layer can be coloured by, or just "none" before a run.

    Read off the layer itself rather than guessed, so a quantity only appears
    once the stage that produces it has run.
    """
    choices = ["none"]
    if viewer is None or layer_name not in getattr(viewer, "layers", {}):
        return choices
    layer = viewer.layers[layer_name]
    if not _is_ours(layer):
        return choices
    for column in getattr(layer, "features", {}):
        if column not in {"u", "v", "key", "edge_index", "node_id"}:
            choices.append(column)
    return choices


def _clear_our_layers(viewer) -> int:
    """Remove every layer this plugin added. Leaves the user's alone."""
    ours = [layer for layer in list(viewer.layers) if _is_ours(layer)]
    for layer in ours:
        viewer.layers.remove(layer)
    return len(ours)


#: Built on first use: defining a QObject subclass registers a Qt meta-object,
#: and one per run would be one per press of the button.
_PROGRESS_BRIDGE_CLASS = None


def _progress_bridge():
    """A QObject whose signal carries progress from the run's thread to the GUI.

    The pipeline reports through a callback that fires deep inside a stage, so
    there is nothing the thread worker could `yield` at the moment a step lands
    -- the events have to cross threads by themselves. A Qt signal is how that
    is done safely: emitting from the worker thread posts to the receiving
    object's event loop, and this object is made on the GUI thread, so every
    slot connected to it runs there. Touching a widget from the worker thread
    instead is what crashes or freezes a Qt application.
    """
    global _PROGRESS_BRIDGE_CLASS
    if _PROGRESS_BRIDGE_CLASS is None:
        from qtpy.QtCore import QObject, Signal

        class ProgressBridge(QObject):
            event = Signal(object)
            #: A StageLayers, already built on the run's thread. Converting
            #: here rather than in the slot is what stops an earlier stage
            #: being drawn with a later stage's numbers.
            layers = Signal(object)

        _PROGRESS_BRIDGE_CLASS = ProgressBridge
    return _PROGRESS_BRIDGE_CLASS()


def _run_in_background(
    settings, schema, report, button, bars=None, viewer=None, results=None,
    after_layers=None,
):
    """Run the pipeline off the GUI thread, reporting back as it goes.

    With *viewer* and *results*, each stage's output is turned into layers as it
    finishes and shown in the viewer the run was launched from. *after_layers*
    is called on the GUI thread once each stage's layers are in place, so the
    panel can offer whatever that stage just made available.
    """
    from napari.qt.threading import thread_worker

    bridge = _progress_bridge()
    show_layers = viewer is not None and results is not None

    def progressed(event: ProgressEvent) -> None:
        if bars is not None:
            bars.show_event(event)
        if event.kind == STAGE_STARTED:
            report.value = f"Running {event.title} ({event.index + 1}/{event.total})..."

    bridge.event.connect(progressed)
    if show_layers:
        def apply(group) -> None:
            _apply_layers(viewer, group, report)
            # Right here, and not at the end of the run: a quantity becomes
            # selectable the moment the stage that fills it lands.
            if after_layers is not None:
                after_layers()

        bridge.layers.connect(apply)

    def produced(stage: str, output) -> None:
        """Build this stage's layers here, on the run's thread.

        Eagerly, because every stage after `build_network` writes onto the same
        graph: convert later and the viewer shows a later stage's numbers under
        this stage's name. And guarded, because a fault in drawing a run must
        never end it -- an eight-hour whole-brain run least of all.
        """
        try:
            group = results.stage_finished(stage, output)
        except Exception:  # noqa: BLE001 - reported, never raised at the run
            logger.exception("could not build layers for stage %s", stage)
            return
        bridge.layers.emit(group)

    @thread_worker
    def run():
        # `bridge` is captured here, which is also what keeps it alive for as
        # long as the run that emits through it.
        return run_pipeline_stages(
            settings,
            schema,
            progress=bridge.event.emit,
            on_stage_output=produced if show_layers else None,
        )

    def finished(graph) -> None:
        button.enabled = True
        if graph is None:
            if bars is not None:
                bars.finish("Finished, no graph")
            report.value = "Finished, but the run produced no graph."
            return
        if bars is not None:
            bars.finish()
        report.value = (
            f"Finished: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} vessels."
        )

    def failed(error: Exception) -> None:
        button.enabled = True
        if bars is not None:
            bars.fail(f"Failed: {type(error).__name__}")
        report.value = f"{type(error).__name__}: {error}"
        logger.exception("pipeline run failed", exc_info=error)

    worker = run()
    worker.returned.connect(finished)
    worker.errored.connect(failed)
    button.enabled = False
    if bars is not None:
        bars.start()
    report.value = "Running..."
    worker.start()


def run_config_widget():
    """Run a config file as it stands, without opening the settings form.

    The same thing `python examples/resistance_network_pipeline.py --config
    my.yaml` does, for a config that is already how you want it.
    """
    from magicgui.widgets import Container, FileEdit, Label, PushButton, TextEdit
    from qtpy.QtWidgets import QVBoxLayout, QWidget

    schema = default_schema()
    chooser = FileEdit(mode="r", filter="*.yaml *.yml", label="Config file")
    report = TextEdit(value="Choose a config file, then press Run.")
    report.read_only = True
    run_button = PushButton(text="Run")
    bars = ProgressBars()

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
        _run_in_background(settings, schema, report, run_button, bars)

    run_button.changed.connect(on_run)
    form = Container(
        widgets=[
            Label(value="Run a config file exactly as it stands."),
            chooser,
            run_button,
        ],
        labels=True,
    )
    # A QWidget rather than one more Container: the progress bars are plain
    # QProgressBars, which a magicgui Container will not hold.
    panel = QWidget()
    # What a test would otherwise have to press a button and wait for a run to see.
    panel._haemolynx_progress = bars
    layout = QVBoxLayout(panel)
    layout.addWidget(form.native)
    layout.addWidget(bars.native)
    layout.addWidget(report.native)
    return panel


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
    from magicgui.widgets import CheckBox, ComboBox, Container, Label, PushButton, TextEdit
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

    for name, value in DISPLAY_SETTINGS_OFF_IN_NAPARI.items():
        if name in rows:
            rows[name].value = value

    for widget in rows.values():
        widget.changed.connect(apply_prerequisites)
    apply_prerequisites()

    report = TextEdit(value="Ready.")
    report.read_only = True
    bars = ProgressBars()

    layer_row: Any = None
    if viewer is not None:
        def image_layers(_widget=None):
            """The viewer's image layers, as (label, layer) choices.

            Explicitly, rather than through magicgui's napari type
            registration: that finds its own viewer, which in a test -- and in
            any second viewer -- is not this one, and the combo comes up empty
            with "is not a valid choice. must be in ()".
            """
            return [
                (layer.name, layer)
                for layer in viewer.layers
                if isinstance(layer, napari.layers.Image)
            ]

        from magicgui.widgets import ComboBox

        layer_picker = ComboBox(
            choices=image_layers, label="Use open layer", nullable=True
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
                layer_picker.reset_choices()
                if layer_picker.value is not layer:
                    layer_picker.value = layer
                use_layer(layer)
            finally:
                applying = False

        def on_layer_added(event) -> None:
            """A dropped image becomes the input, panel already open or not."""
            layer_picker.reset_choices()
            adopt(getattr(event, "value", None))

        def on_layer_removed(_event) -> None:
            layer_picker.reset_choices()

        # Both orders have to work: open the panel with an image already there,
        # or drop one in while it is open.
        layer_picker.changed.connect(lambda *_: adopt(layer_picker.value))
        use_button.changed.connect(lambda *_: use_layer(layer_picker.value))
        viewer.layers.events.inserted.connect(on_layer_added)
        viewer.layers.events.removed.connect(on_layer_removed)

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
    clear_button = PushButton(text="Clear layers")

    # What a run puts in the viewer, and how it is coloured. These are panel
    # controls rather than settings: a config file is read by CLI runs too,
    # where "show it in napari" means nothing.
    show_results = CheckBox(value=True, text="Show each stage in the viewer")
    show_steps = CheckBox(value=False, text="Show each topology step")
    colour_vessels = ComboBox(
        label="Colour vessels by", choices=_colour_choices(viewer, VESSELS)
    )
    colour_nodes = ComboBox(
        label="Colour nodes by", choices=_colour_choices(viewer, NODES)
    )
    view = SimpleNamespace(results=None)

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
        results = None
        if show_results.value and viewer is not None:
            results = ResultLayers(show_steps=bool(show_steps.value))
            view.results = results
        _run_in_background(
            settings, schema, report, run_button, bars,
            viewer=viewer if show_results.value else None,
            results=results,
            after_layers=refresh_colours,
        )

    def on_clear() -> None:
        if viewer is None:
            return
        removed = _clear_our_layers(viewer)
        view.results = None
        refresh_colours()  # the columns went with the layers
        report.value = f"Removed {removed} HaemoLynx layer(s)."

    def refresh_colours() -> None:
        """Offer whatever the layers now carry, as soon as they carry it."""
        _refresh_colour_choices(
            viewer, ((VESSELS, colour_vessels), (NODES, colour_nodes))
        )

    def on_colour_changed(*_args) -> None:
        """Recolour what is already there; no geometry is rebuilt."""
        if viewer is None:
            return
        for name, chooser in ((VESSELS, colour_vessels), (NODES, colour_nodes)):
            if name not in viewer.layers:
                continue
            layer = viewer.layers[name]
            if not _is_ours(layer):
                continue
            column = chooser.value
            kind = "categorical" if column in TEXT_COLUMNS else "continuous"
            cycle = ()
            if kind == "categorical" and column in getattr(layer, "features", {}):
                cycle = colour_cycle_for(layer.features[column])
            _colour_layer(layer, column, kind, cycle)

    load_button.changed.connect(on_load)
    save_button.changed.connect(on_save)
    check_button.changed.connect(on_check)
    run_button.changed.connect(on_run)
    clear_button.changed.connect(on_clear)
    def chosen_by_hand(chooser):
        """Note that this box now holds a choice, not just a default."""
        def handler(*_args) -> None:
            chooser._haemolynx_chosen = True
            on_colour_changed()

        return handler

    colour_vessels.changed.connect(chosen_by_hand(colour_vessels))
    colour_nodes.changed.connect(chosen_by_hand(colour_nodes))

    buttons = Container(
        widgets=[load_button, save_button, check_button, run_button, clear_button],
        layout="horizontal",
        labels=False,
    )
    view_controls = Container(
        widgets=[show_results, show_steps, colour_vessels, colour_nodes],
        labels=True,
    )

    panel = QWidget()
    # What the panel would send to a run, and what a run would report back,
    # for a test that cannot press buttons and wait.
    panel._haemolynx_values = current_values
    panel._haemolynx_progress = bars
    panel._haemolynx_view = view
    panel._haemolynx_colour = {"vessels": colour_vessels, "nodes": colour_nodes}
    panel._haemolynx_refresh_colours = refresh_colours
    panel._haemolynx_show_results = show_results
    layout = QVBoxLayout(panel)
    if layer_row is not None:
        layout.addWidget(layer_row.native)
    layout.addWidget(tab_widget)
    layout.addWidget(view_controls.native)
    layout.addWidget(buttons.native)
    layout.addWidget(bars.native)
    layout.addWidget(report.native)
    return panel
