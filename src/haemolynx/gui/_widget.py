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
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from haemolynx.gui.form import Field, display_value_for
from haemolynx.gui.layers import input_for_layer
from haemolynx.gui.results import (
    NODES,
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
            continue
        if spec.name in viewer.layers:
            try:
                _attach_colour_scale(viewer, viewer.layers[spec.name])
                _refresh_layer_controls(viewer, viewer.layers[spec.name])
            except Exception:  # noqa: BLE001 - a missing colour bar is survivable
                logger.debug("could not attach a colour bar to %s",
                             spec.name, exc_info=True)
    for name, column in group.recolour:
        layer = viewer.layers[name] if name in viewer.layers else None
        if layer is not None and _is_ours(layer):
            _colour_layer(layer, column)
    if group.ndisplay is not None:
        viewer.dims.ndisplay = group.ndisplay
    if report is not None and group.note:
        report.value = f"{group.title}: {group.note}"


def _colour_attribute(layer) -> str:
    """Where a layer keeps its colour: vessels on the edge, points on the face."""
    return "edge_color" if layer.__class__.__name__ == "Vectors" else "face_color"


def _colour_layer(layer, column: str | None, kind: str = "continuous",
                  cycle=(), limits=None) -> None:
    """Colour a layer by one of its feature columns."""
    attribute = _colour_attribute(layer)
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
        # After the column, and through the same path the Fit buttons use: the
        # range has to be applied *and* the colours re-mapped against it. Set
        # before, and the assignment above maps with the old range; set with a
        # plain setattr, and nothing re-maps at all -- which is how `flow_abs`
        # came to be the selected colouring and not the one on screen.
        if limits is None:
            limits = _data_range(layer, column)
        if limits is not None:
            _apply_contrast_limits(layer, *limits)
    _record_colour(layer, column)


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


def _active_column(layer) -> str | None:
    """Which feature the layer is coloured by right now, or None.

    Read off the layer rather than remembered, because the choice is made in
    napari's own layer controls: anything we noted when we last set a colouring
    goes stale the moment the user picks something on the left.

    `color_properties` is private, so fall back to what we recorded if a napari
    version moves it -- a stale answer beats no colour bar.
    """
    manager = getattr(layer, "_edge", None) or getattr(layer, "_face", None)
    properties = getattr(manager, "color_properties", None)
    name = getattr(properties, "name", None)
    if name:
        return str(name)
    tag = getattr(layer, "metadata", {}).get(OURS)
    return tag.get("colour_by") if isinstance(tag, dict) else None


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


#: Identifiers rather than quantities: colouring by one shows nothing.
NOT_WORTH_COLOURING_BY = frozenset({"u", "v", "key", "edge_index", "node_id"})

#: How wide and tall the colour bar is drawn, in pixels.
COLORBAR_SIZE = (150, 12)


def _colorbar_pixmap(colormap_name: str = "viridis", size=COLORBAR_SIZE):
    """The colormap as a strip you can actually look at.

    napari draws no colour bar for a Points or Vectors layer coloured by a
    feature -- the contrast slider in the layer controls belongs to Image
    layers -- so there is nothing on screen saying which end is which. It does
    ship the pieces to draw one.
    """
    from napari.utils.colormaps import ensure_colormap
    from napari.utils.colormaps.colorbars import make_colorbar
    from qtpy.QtGui import QImage, QPixmap

    width, height = size
    bar = np.ascontiguousarray(
        make_colorbar(ensure_colormap(colormap_name), size=(height, width),
                      horizontal=True)
    )
    image = QImage(bar.data, width, height, 4 * width, QImage.Format_RGBA8888)
    return QPixmap.fromImage(image.copy())


def _is_text_column(layer, column: str | None) -> bool:
    """Whether a column holds labels rather than numbers, asked of the data.

    Not of a list of names. `TEXT_COLUMNS` names the text columns the results
    module writes, and `role` -- the one on the boundary nodes -- is not among
    them, so choosing it tried to map "starting" and "output" onto a colormap
    and raised `could not convert string to float`. Any column any layer ever
    carries has a dtype; that is the honest question.
    """
    values = getattr(layer, "features", {}).get(column) if column else None
    if values is None:
        return False
    kind = np.asarray(values).dtype.kind
    if kind in "OUS":
        # Object arrays can still be numbers stored the long way round.
        return not all(isinstance(v, (int, float, np.number)) or v is None
                       for v in np.asarray(values).ravel()[:100])
    return False


def _format_limit(value: float) -> str:
    """Short enough to read, wide enough for a flow of 1e-16."""
    if value is None or not np.isfinite(value):
        return ""
    if value == 0:
        return "0"
    return f"{value:.4g}"


def _data_range(layer, column: str | None, low_percentile=0.0, high_percentile=100.0):
    """The range of the column being shown, ignoring the values it has not got.

    Percentiles rather than only min and max because a flow distribution is
    long-tailed: on a real run a handful of vessels carry orders of magnitude
    more than the rest, and against the full range everything else is one
    colour at the bottom of the map.
    """
    if column in {None, "", "none"} or column not in getattr(layer, "features", {}):
        return None
    if _is_text_column(layer, column):
        return None
    try:
        values = np.asarray(layer.features[column], dtype=float)
    except (TypeError, ValueError):
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    low = float(np.percentile(finite, low_percentile))
    high = float(np.percentile(finite, high_percentile))
    if high <= low:
        high = low + abs(low) * 1e-6 + 1e-30
    return low, high


def _contrast_limits_attribute(layer) -> str:
    return f"{_colour_attribute(layer).replace('_color', '')}_contrast_limits"


def _apply_contrast_limits(layer, low: float, high: float) -> bool:
    """Set the range the colormap spans, and get the canvas to show it.

    Setting the limits alone changes the colours on the model and tells nobody:
    `ColorManager.contrast_limits` is a plain field, so no `edge_color` event is
    emitted and the canvas keeps drawing the buffer it already has. The change
    only appeared once you picked a different feature and came back -- because
    that assignment does fire the event.

    So re-assign the column afterwards, which is exactly what going away and
    coming back does, and the view updates when the range is set.
    """
    name = _contrast_limits_attribute(layer)
    if not hasattr(layer, name) or not (np.isfinite(low) and np.isfinite(high)):
        return False
    if high <= low:
        return False
    try:
        setattr(layer, name, (float(low), float(high)))
    except (ValueError, TypeError):
        return False
    column = _active_column(layer)
    if column:
        try:
            setattr(layer, _colour_attribute(layer), column)
        except (KeyError, TypeError, ValueError):
            logger.debug("could not repaint %s after rescaling", layer.name)
    return True


def _viewer_colorbar(layer, create: bool = True):
    """The overlay that draws a colour bar in the canvas, made if need be.

    napari registers one for Points -- `face_colorbar`, hidden by default --
    and none at all for Vectors, so the vessels need theirs adding. The overlay
    reads the layer's colour manager, so it stays right as the colouring and
    the range change.
    """
    overlays = getattr(layer, "_overlays", None)
    if overlays is None:
        return None
    name = "edge_colorbar" if _colour_attribute(layer) == "edge_color" \
        else "face_colorbar"
    if name not in overlays:
        if not create:
            # Only looking. Making one here would register an overlay -- and
            # fire the event the canvas builds visuals from -- for a layer
            # nobody has asked to show a bar for.
            return None
        from napari.components.overlays import ColorBarOverlay

        manager = f"_{name.split('_')[0]}"
        overlays[name] = ColorBarOverlay(colormanager_attribute=manager)
    return overlays[name]


@contextmanager
def _blocked(widget):
    """Change a Qt widget without its own signal coming back at us."""
    previous = widget.blockSignals(True)
    try:
        yield
    finally:
        widget.blockSignals(previous)


class _ColourScale:
    """A colour bar for one layer, with the range it spans, editable.

    napari draws neither for a Points or Vectors layer coloured by a feature:
    the contrast slider in the layer controls belongs to Image layers, so a
    feature colouring arrives with no legend and no way to rescale it except
    from the console. That hurts most on the quantities this plugin exists to
    show -- flows span orders of magnitude, and against their own full range
    almost every vessel sits at the bottom of the colormap.

    Every control here reads and writes the layer directly, so it stays true
    whether the colouring was changed from this panel or from napari's own
    dropdown on the left.
    """

    def __init__(self, viewer, layer_name: str) -> None:
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import (
            QCheckBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout,
            QWidget,
        )

        self._viewer = viewer
        self._layer_name = layer_name
        self._column: str | None = None
        self._connected = None
        self.shown = False

        self.native = QWidget()
        outer = QVBoxLayout(self.native)
        outer.setContentsMargins(0, 2, 0, 2)
        outer.setSpacing(2)

        self.heading = QLabel("no colouring")
        self.heading.setToolTip(
            "The range the colours span. Type a number, or let it fit the data."
        )
        outer.addWidget(self.heading)

        row = QHBoxLayout()
        row.setSpacing(4)
        self.low = QLineEdit()
        self.high = QLineEdit()
        for box in (self.low, self.high):
            box.setFixedWidth(74)
            box.setAlignment(Qt.AlignRight)
            box.editingFinished.connect(self._apply_typed)
        self.bar = QLabel()
        self.bar.setFixedHeight(COLORBAR_SIZE[1])
        self.bar.setPixmap(_colorbar_pixmap())
        row.addWidget(self.low)
        row.addWidget(self.bar, 1)
        row.addWidget(self.high)
        outer.addLayout(row)

        buttons = QHBoxLayout()
        buttons.setSpacing(4)
        # Full range and a trimmed one. The trimmed one is the useful default
        # on real data: a handful of vessels carry most of the flow, and
        # including them flattens everything else to a single colour.
        self.full_button = QPushButton("Fit all")
        self.full_button.setToolTip("Span the smallest and largest value")
        self.full_button.clicked.connect(lambda: self.autoscale(0.0, 100.0))
        self.trim_button = QPushButton("Fit 1-99%")
        self.trim_button.setToolTip(
            "Ignore the extreme 1% at each end, so the bulk of the data spreads"
        )
        self.trim_button.clicked.connect(lambda: self.autoscale(1.0, 99.0))
        buttons.addWidget(self.full_button)
        buttons.addWidget(self.trim_button)
        buttons.addStretch(1)
        outer.addLayout(buttons)

        self.in_viewer = QCheckBox("Show colour bar in the viewer")
        self.in_viewer.setToolTip(
            "Draw the scale in the canvas, beside the data it describes"
        )
        self.in_viewer.toggled.connect(self._show_in_viewer)
        outer.addWidget(self.in_viewer)
        self.native.setVisible(False)

    # -- state ------------------------------------------------------------

    def _layer(self):
        layers = getattr(self._viewer, "layers", {}) if self._viewer else {}
        if self._layer_name not in layers:
            return None
        layer = layers[self._layer_name]
        return layer if _is_ours(layer) else None

    def follow_the_layer(self) -> None:
        """Show whatever the layer is coloured by, however it got that way.

        Also (re)connect to the layer's colour event, so a colouring chosen in
        napari's controls on the left moves this bar too. Connecting is cheap
        and idempotent-ish: the layer is replaced on a re-run, so the previous
        connection dies with it.
        """
        layer = self._layer()
        if layer is not None and layer is not self._connected:
            events = getattr(layer, "events", None)
            attribute = _colour_attribute(layer)
            signal = getattr(events, attribute, None) if events else None
            if signal is not None:
                signal.connect(lambda *_a: self.follow_the_layer())
            self._connected = layer
        column = _active_column(layer) if layer is not None else None
        changed = column != self._column
        self.refresh(column)
        if changed and self.shown:
            # A colouring chosen in napari's own dropdown is applied with
            # whatever range the last one used, so a column of flows lands on
            # a scale of branch orders and every vessel comes out one colour.
            # Fit it, exactly as the button does.
            self.autoscale(0.0, 100.0)

    def refresh(self, column: str | None) -> None:
        """Show the range of *column*, or hide if there is nothing to show."""
        layer = self._layer()
        self._column = None if column in {None, "", "none"} else column
        usable = (
            layer is not None
            and self._column is not None
            and not _is_text_column(layer, self._column)
        )
        # Recorded as well as applied: Qt reports a child of an unshown window
        # as invisible whatever we set, so `isVisible()` cannot tell a test
        # whether the bar was hidden on purpose.
        self.shown = bool(usable)
        self.native.setVisible(self.shown)
        if not usable:
            return

        self.heading.setText(str(self._column))
        overlay = None
        try:
            overlay = _viewer_colorbar(layer, create=False)
        except Exception:  # noqa: BLE001 - private napari ground
            logger.debug("no colour bar overlay available", exc_info=True)
        if overlay is not None and self.in_viewer.isChecked() != bool(overlay.visible):
            with _blocked(self.in_viewer):
                self.in_viewer.setChecked(bool(overlay.visible))
        limits = getattr(layer, _contrast_limits_attribute(layer), None)
        if limits is None:
            limits = _data_range(layer, self._column)
        if limits is not None:
            self.low.setText(_format_limit(float(limits[0])))
            self.high.setText(_format_limit(float(limits[1])))

    # -- actions ----------------------------------------------------------

    def autoscale(self, low_percentile: float, high_percentile: float) -> bool:
        layer = self._layer()
        found = _data_range(layer, self._column, low_percentile, high_percentile) \
            if layer is not None else None
        if found is None or not _apply_contrast_limits(layer, *found):
            return False
        self.low.setText(_format_limit(found[0]))
        self.high.setText(_format_limit(found[1]))
        return True

    def _show_in_viewer(self, wanted: bool) -> None:
        """Put the colour bar in the canvas, or take it away again."""
        layer = self._layer()
        if layer is None:
            return
        try:
            overlay = _viewer_colorbar(layer)
        except Exception:  # noqa: BLE001 - private napari ground
            logger.debug("no colour bar overlay available", exc_info=True)
            return
        if overlay is not None:
            overlay.visible = bool(wanted)

    def _apply_typed(self) -> None:
        layer = self._layer()
        if layer is None or self._column is None:
            return
        try:
            low, high = float(self.low.text()), float(self.high.text())
        except ValueError:
            self.refresh(self._column)  # unreadable: put back what is real
            return
        if not _apply_contrast_limits(layer, low, high):
            self.refresh(self._column)


class _FeatureChooser:
    """The dropdown napari gives Vectors and does not give Points.

    `QtVectorsControls` has an "edge feature:" box; `QtPointsControls` has a
    colour swatch and no way to colour by a column at all, so the node layer
    arrives with `pressure`, `degree` and `node_id` on it and nothing to pick
    between them.

    Rebuilt from the layer whenever it is refreshed rather than filled once,
    because napari's own box is filled in its constructor and never updated --
    the bug that kept flow out of the vessels list -- and there is no reason to
    repeat it here.
    """

    def __init__(self, viewer, layer_name: str) -> None:
        from qtpy.QtWidgets import QComboBox

        self._viewer = viewer
        self._layer_name = layer_name
        self.native = QComboBox()
        self.native.currentTextChanged.connect(self._chosen)

    def _layer(self):
        layers = getattr(self._viewer, "layers", {}) if self._viewer else {}
        layer = layers[self._layer_name] if self._layer_name in layers else None
        return layer if layer is not None and _is_ours(layer) else None

    def refresh(self) -> None:
        """Offer the columns the layer holds now, keeping the current one."""
        layer = self._layer()
        if layer is None:
            return
        columns = [
            name for name in getattr(layer, "features", {})
            if name not in NOT_WORTH_COLOURING_BY
        ]
        active = _active_column(layer)
        if list(self._items()) == columns and self.native.currentText() == (active or ""):
            return
        with _blocked(self.native):
            self.native.clear()
            self.native.addItems(columns)
            if active in columns:
                self.native.setCurrentIndex(columns.index(active))

    def _items(self):
        return (self.native.itemText(i) for i in range(self.native.count()))

    def _chosen(self, column: str) -> None:
        layer = self._layer()
        if layer is None or not column:
            return
        text = _is_text_column(layer, column)
        cycle = colour_cycle_for(layer.features[column]) if text else ()
        _colour_layer(layer, column, "categorical" if text else "continuous", cycle)


def _layer_controls(viewer, layer):
    """The controls napari shows on the left for *layer*, if they can be found.

    Entirely private API: a plugin has no supported way to add a row to another
    layer type's controls, and the alternative -- registering our own controls
    class -- would change every Vectors and Points layer in the session, not
    just ours. So this reaches in, and every caller treats failure as normal:
    the colour bar is worth having, and not worth breaking a run over.
    """
    window = getattr(viewer, "window", None)
    # `_qt_viewer`, not `qt_viewer`: the public spelling is deprecated and warns
    # on every access, and this is private ground either way.
    qt_viewer = getattr(window, "_qt_viewer", None) if window else None
    container = getattr(qt_viewer, "controls", None)
    widgets = getattr(container, "widgets", None)
    if widgets is None:
        return None
    try:
        return widgets[layer]
    except (KeyError, TypeError):
        return None


def _attach_colour_scale(viewer, layer) -> bool:
    """Put a colour bar in this layer's controls, once.

    napari gives a feature colouring no legend and no way to rescale it: the
    contrast slider in the layer controls belongs to Image layers. So the range
    is invisible and unreachable on exactly the quantities this plugin exists to
    show -- and flows span orders of magnitude, so against their full range
    nearly every vessel sits at one end of the colormap.

    Keyed on the controls widget rather than the layer, because napari builds a
    fresh one whenever a layer is removed and re-added.
    """
    from qtpy.QtWidgets import QLabel

    controls = _layer_controls(viewer, layer)
    if controls is None or getattr(controls, "_haemolynx_scale", None) is not None:
        return controls is not None
    layout = controls.layout()
    if not hasattr(layout, "addRow"):
        return False
    if _colour_attribute(layer) == "face_color":
        # Vectors already has "edge feature:"; Points has nothing.
        chooser = _FeatureChooser(viewer, layer.name)
        layout.addRow(QLabel("node feature:"), chooser.native)
        controls._haemolynx_feature = chooser
        chooser.refresh()
    scale = _ColourScale(viewer, layer.name)
    layout.addRow(QLabel("colour range:"), scale.native)
    controls._haemolynx_scale = scale
    scale.follow_the_layer()
    return True


def _refresh_layer_controls(viewer, layer) -> None:
    """Let our additions catch up with whatever the stage just changed."""
    controls = _layer_controls(viewer, layer)
    for attribute in ("_haemolynx_feature", "_haemolynx_scale"):
        widget = getattr(controls, attribute, None)
        if widget is None:
            continue
        if hasattr(widget, "follow_the_layer"):
            widget.follow_the_layer()
        else:
            widget.refresh()


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
    settings, schema, report, button, bars=None, viewer=None, results=None):
    """Run the pipeline off the GUI thread, reporting back as it goes.

    With *viewer* and *results*, each stage's output is turned into layers as it
    finishes and shown in the viewer the run was launched from.
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
        bridge.layers.connect(lambda group: _apply_layers(viewer, group, report))

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


#: What the About panel says. Written here rather than in the widget so it can
#: be checked without a display -- and so the two questions it answers stay
#: answered: where the colour controls are (napari's own layer controls, not
#: this plugin's panel) and what a config file is for.
ABOUT_TEXT = """\
HaemoLynx {version}

Turns 3D microvascular microscopy into a NetworkX graph with haemodynamic
edge weights, VTK exports and network statistics.

Pipeline settings runs the pipeline: one tab per stage, in the order they
run. Point it at the image layer you have open, or load a config file.

Colours are not set here. Select a vessel or node layer and use napari's
own layer controls on the left: "edge feature:" and "node feature:" choose
the quantity, "colour range:" sets the scale.

A config file is the same YAML the command line takes, so a run set up
here repeats with:

    python examples/resistance_network_pipeline.py --config my.yaml

Settings in this build: {settings}
Docs and issues: https://github.com/physiomelinks/HaemoLynx
"""


def about_text() -> str:
    """The About panel's text, filled in for this installation."""
    import haemolynx

    return ABOUT_TEXT.format(
        version=getattr(haemolynx, "__version__", "unknown"),
        settings=len(list(default_schema())),
    )


def about_widget():
    """What HaemoLynx is, and the two things people ask the panel that it
    cannot answer: where the colour controls live, and what a config is for."""
    from magicgui.widgets import TextEdit

    report = TextEdit(value=about_text())
    report.read_only = True
    return report.native


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

    #: What a loaded config said each path setting was, before its FileEdit
    #: made it absolute. Empty until a config is opened.
    loaded_paths: dict[str, Any] = {}

    def current_values() -> dict[str, Any]:
        """What the panel says, in the settings' own terms.

        Read back through the field rather than straight off the widget: an
        empty picker is *unset*, not the working directory, and an empty box is
        unset rather than zero. A path a loaded config gave is handed back as
        that config wrote it, for as long as the row still names the same file
        -- see `load_config_file`. Picking a different file in the row replaces
        it, because then the absolute path is what the user chose.
        """
        values = {
            name: fields[name].to_setting_value(widget.value)
            for name, widget in rows.items()
        }
        for name, original in loaded_paths.items():
            current = values.get(name)
            if current is None:
                continue
            try:
                unchanged = Path(current) == Path(original).resolve()
            except (TypeError, ValueError, OSError):
                continue
            if unchanged:
                values[name] = original
        return values

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
    view = SimpleNamespace(results=None)

    def _settings() -> dict[str, Any]:
        return resolve_settings(current_values(), schema=schema, config_path=None)

    def load_config_file(path: Path | str) -> None:
        """Put a config file's settings into the form.

        Only reads the file. Nothing it names is opened -- an `input_path`
        pointing at an image that is not on this machine still loads, because
        the image a run works on is the layer open in napari, and a config is
        routinely written on one machine and read on another. The paths are
        checked when a run is about to start, by "Run checks" and by the run
        itself, which is where a missing file is worth stopping for.

        A FileEdit stores whatever it is given as an absolute path, so putting
        a config's relative `classifiers/nerve_classifier.ilp` into one turns
        it into `/home/you/wherever/classifiers/nerve_classifier.ilp`. Left
        alone that rewrites the config: every relative path in it becomes
        specific to this machine, "Save config..." writes those back, and the
        settings then differ from their defaults, so a run warns that fourteen
        of them are set while nothing reads them. What the file said is kept
        here, and `current_values` hands it back for any row still naming the
        same file.
        """
        loaded = load_config(Path(path), schema)
        loaded_paths.clear()
        for name, value in loaded.items():
            if name in rows:
                if schema[name].kind == "path" and value is not None:
                    loaded_paths[name] = value
                rows[name].value = display_value_for(schema[name], value)
        apply_prerequisites()
        report.value = f"Loaded {path}"

    def on_load() -> None:
        from qtpy.QtWidgets import QFileDialog

        path, _filter = QFileDialog.getOpenFileName(
            None, "Open a HaemoLynx config", "", "YAML (*.yaml *.yml)"
        )
        if not path:
            return
        load_config_file(path)

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
        )

    def on_clear() -> None:
        if viewer is None:
            return
        removed = _clear_our_layers(viewer)
        view.results = None
        report.value = f"Removed {removed} HaemoLynx layer(s)."

    load_button.changed.connect(on_load)
    save_button.changed.connect(on_save)
    check_button.changed.connect(on_check)
    run_button.changed.connect(on_run)
    clear_button.changed.connect(on_clear)
    buttons = Container(
        widgets=[load_button, save_button, check_button, run_button, clear_button],
        layout="horizontal",
        labels=False,
    )
    view_controls = Container(
        widgets=[show_results, show_steps],
        labels=True,
    )

    panel = QWidget()
    # What the panel would send to a run, and what a run would report back,
    # for a test that cannot press buttons and wait.
    panel._haemolynx_values = current_values
    panel._haemolynx_progress = bars
    panel._haemolynx_view = view
    panel._haemolynx_show_results = show_results
    panel._haemolynx_load_config = load_config_file
    panel._haemolynx_report = lambda: report.value
    panel._haemolynx_rows = lambda: rows
    layout = QVBoxLayout(panel)
    if layer_row is not None:
        layout.addWidget(layer_row.native)
    layout.addWidget(tab_widget)
    layout.addWidget(view_controls.native)
    layout.addWidget(buttons.native)
    layout.addWidget(bars.native)
    layout.addWidget(report.native)
    return panel
