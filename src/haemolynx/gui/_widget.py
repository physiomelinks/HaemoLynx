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
import math
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from haemolynx.gui.form import (
    Field,
    SHARED_ILASTIK_SETTINGS,
    SHARED_ILASTIK_SETTING_SET,
    display_value_for,
    shared_ilastik_host,
)
from haemolynx.gui.layers import input_for_layer, voxel_size_xyz_from_scale
from haemolynx.gui.log_view import LogView, VERBOSE_LEVEL
from haemolynx.gui.run_log import DEFAULT_LEVEL, attach
from haemolynx.gui.results import (
    BRANCH_HOVER,
    NODES,
    VESSELS,
    ResultLayers,
    colour_cycle_for,
    filter_points_by_z,
    filter_vectors_by_z,
    is_z_depth_filtered_layer,
)
from haemolynx.gui.progress import ProgressDisplay
from haemolynx.gui.run_state import (
    ALREADY_RUNNING,
    CANCELLED,
    FINISHED_FIRST,
    RunCancelled,
    RunState,
    clear_message,
)
from haemolynx.gui.stage_checkpoints import (
    SKIP_FOR_RESUME,
    StageCheckpoints,
    can_revert_from,
    discard_cached_artefacts_for_settings,
    previous_tab,
    restore_message,
)
from haemolynx.gui.tabs import tabs_for
from haemolynx.parsers import dump_config, load_config
from haemolynx.pipeline import default_schema, preflight, resolve_settings, run_pipeline_stages
from haemolynx.pipeline.progress import STAGE_STARTED, ProgressEvent, log_progress

logger = logging.getLogger(__name__)

#: What a layer looks like when the user asks for no colouring at all.
UNCOLOURED = "#cccccc"

#: The same grey, as RGBA, for the places that build a colour array by hand.
UNCOLOURED_RGBA = (0.8, 0.8, 0.8, 1.0)

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

#: What napari calls the log window's dock.
LOG_DOCK_NAME = "HaemoLynx run log"


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


def _scale_layer_from_its_file(layer, path) -> tuple[float, float, float] | None:
    """Give *layer* the voxel size its own file describes, if it has none.

    napari's readers do not apply a TIFF's resolution tags, so a stack opened
    by dragging it in sits at one unit per voxel while everything this plugin
    draws is in microns -- node `pos` and edge `voxels` are physical already.
    On the nerve stack that is 2.029 um of z drawn as 1, so the image ends up
    at 58% of its depth and the vessels do not lie on the vessels.

    Only ever fills a gap: a layer whose scale someone has already set is left
    alone, and so is a file whose tags say nothing. Returns the scale applied,
    or None if it left the layer as it was.
    """
    from haemolynx.io import read_voxel_size_xyz, voxel_size_zyx_from_xyz

    if path is None or voxel_size_xyz_from_scale(getattr(layer, "scale", None)):
        return None
    found = read_voxel_size_xyz(path)
    if found is None:
        return None
    scale = voxel_size_zyx_from_xyz(found[0])
    try:
        layer.scale = scale
    except Exception:  # noqa: BLE001 - a layer that will not take a scale is survivable
        logger.debug("Could not scale %s", getattr(layer, "name", "?"), exc_info=True)
        return None
    return scale


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

    def reset(self) -> None:
        """Both bars back to nothing: the run was stopped on purpose."""
        self.display.reset()
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

#: Session-wide checkbox selection for the branch-hover metrics panel.
#: ``None`` means "not yet chosen" -- the panel then defaults to every
#: metric the current graph can offer. Survives layer rebuilds within a
#: napari session so toggling mid-run does not reset after the next stage.
_branch_hover_session_selected: tuple[str, ...] | None = None

#: Keys stashed on a branch-hover LayerSpec that must not reach napari.
_BRANCH_HOVER_OPTION_KEYS = frozenset(
    {"branch_hover_available", "branch_hover_selected"}
)


def _is_ours(layer) -> bool:
    return bool(getattr(layer, "metadata", {}).get(OURS))


def _store_z_filter_cache(
    layer,
    data: Any = None,
    features: Mapping[str, np.ndarray] | None = None,
    *,
    segment_owner: Any = None,
) -> None:
    """Remember unfiltered graph geometry for the view-only Z depth filter."""
    kind = layer.__class__.__name__.lower()
    if not is_z_depth_filtered_layer(layer.name, kind):
        return
    if data is None:
        data = layer.data
    if features is None:
        features = dict(getattr(layer, "features", {}))
    tag = dict(getattr(layer, "metadata", {}).get(OURS) or {})
    cache: dict[str, Any] = {
        "data": np.asarray(data),
        "features": {name: np.asarray(values) for name, values in features.items()},
    }
    owner = segment_owner
    if owner is None:
        owner = tag.get("segment_owner")
    if owner is not None:
        cache["segment_owner"] = np.asarray(owner)
    tag["z_filter_full"] = cache
    metadata = dict(getattr(layer, "metadata", {}) or {})
    metadata[OURS] = tag
    layer.metadata = metadata


def _apply_z_filter(
    viewer,
    z_min: float,
    z_max: float,
    *,
    z_extent: float | None = None,
) -> None:
    """Redraw graph Vectors/Points layers filtered to a physical Z band."""
    full_range = (
        z_extent is not None
        and z_min <= 0.0
        and z_max >= z_extent - max(1e-6, abs(z_extent) * 1e-9)
    )
    for layer in viewer.layers:
        if not _is_ours(layer):
            continue
        kind = layer.__class__.__name__.lower()
        if not is_z_depth_filtered_layer(layer.name, kind):
            continue
        tag = getattr(layer, "metadata", {}).get(OURS) or {}
        cache = tag.get("z_filter_full")
        if cache is None:
            _store_z_filter_cache(layer)
            tag = getattr(layer, "metadata", {}).get(OURS) or {}
            cache = tag.get("z_filter_full")
        if cache is None:
            continue
        if full_range:
            data = cache["data"]
            features = cache["features"]
            segment_owner = cache.get("segment_owner")
        elif kind == "vectors":
            data, features, segment_owner = filter_vectors_by_z(
                cache["data"],
                cache["features"],
                z_min,
                z_max,
                segment_owner=cache.get("segment_owner"),
            )
        else:
            data, features = filter_points_by_z(
                cache["data"], cache["features"], z_min, z_max
            )
            segment_owner = cache.get("segment_owner")
        _set_z_filtered_layer_data(
            viewer, layer, kind, data, features, segment_owner
        )


def _set_z_filtered_layer_data(
    viewer,
    layer,
    kind: str,
    data: Any,
    features: Mapping[str, np.ndarray],
    segment_owner: Any,
) -> None:
    """Write filtered geometry; recreate when shrinking avoids stale Vectors draw."""
    old_count = len(np.asarray(getattr(layer, "data", ())))
    new_count = len(np.asarray(data))
    adder = getattr(viewer, f"add_{kind}", None)
    if kind in {"vectors", "points"} and new_count < old_count and adder is not None:
        name = layer.name
        visible = layer.visible
        scale = getattr(layer, "scale", None)
        metadata = dict(getattr(layer, "metadata", {}) or {})
        if segment_owner is not None:
            ours = dict(metadata.get(OURS) or {})
            ours["segment_owner"] = np.asarray(segment_owner)
            metadata[OURS] = ours
        add_kwargs: dict[str, Any] = {
            "name": name,
            "visible": visible,
            "metadata": metadata,
        }
        if scale is not None:
            add_kwargs["scale"] = scale
        if features:
            add_kwargs["features"] = dict(features)
        if kind == "vectors":
            for key in ("vector_style", "edge_width", "length", "out_of_slice_display"):
                if hasattr(layer, key):
                    add_kwargs[key] = getattr(layer, key)
            colour_attr, colour = "edge_color", getattr(layer, "edge_color", None)
        else:
            for key in ("size", "out_of_slice_display"):
                if hasattr(layer, key):
                    add_kwargs[key] = getattr(layer, key)
            colour_attr, colour = "face_color", getattr(layer, "face_color", None)
        viewer.layers.remove(layer)
        new_layer = adder(data, **add_kwargs)
        if colour is not None:
            setattr(new_layer, colour_attr, colour)
        return
    layer.data = data
    if features:
        layer.features = features
    if segment_owner is not None:
        metadata = dict(getattr(layer, "metadata", {}) or {})
        ours = dict(metadata.get(OURS) or {})
        ours["segment_owner"] = np.asarray(segment_owner)
        metadata[OURS] = ours
        layer.metadata = metadata


def _sync_z_depth_slider(slider, results: ResultLayers | None) -> None:
    """Set slider range from the image stack; hide until skeletonise has run."""
    extent = results.image_z_extent_um() if results is not None else None
    if extent is None or extent <= 0.0:
        slider.setEnabled(False)
        slider.setVisible(False)
        return
    step = float(results._voxel_size_zyx[0]) if results is not None else 1.0  # noqa: SLF001
    slider.setVisible(True)
    slider.setEnabled(True)
    slider.blockSignals(True)
    try:
        slider.setRange(0.0, extent)
        if hasattr(slider, "setSingleStep"):
            slider.setSingleStep(max(step, 1e-6))
        lo, hi = slider.value()
        if hi <= lo or hi > extent or lo < 0.0:
            lo, hi = 0.0, extent
        else:
            lo = max(0.0, min(lo, extent))
            hi = max(lo, min(hi, extent))
        slider.setValue((lo, hi))
    finally:
        slider.blockSignals(False)


def _park_z_depth_host(host, parking) -> None:
    """Take *host* out of napari's layer controls without destroying it."""
    try:
        host.setParent(parking)
    except RuntimeError:
        return


def _reparent_z_depth_slider(
    viewer,
    *,
    slider,
    host,
    parking,
    controls_holder: dict[str, Any],
    results: ResultLayers | None,
) -> None:
    """Mount the shared Z depth row at the top of the active layer's controls."""
    _park_z_depth_host(host, parking)
    controls_holder["controls"] = None

    if viewer is None:
        host.setVisible(False)
        return

    active = viewer.layers.selection.active
    if active is None or not _is_ours(active):
        host.setVisible(False)
        return

    controls = _layer_controls(viewer, active)
    if controls is None:
        host.setVisible(False)
        return

    layout = controls.layout()
    if layout is None or not hasattr(layout, "insertRow"):
        host.setVisible(False)
        return

    layout.insertRow(0, host)
    controls_holder["controls"] = controls

    _sync_z_depth_slider(slider, results)
    host.setVisible(slider.isVisible())


#: Flow-direction arrow scale matches ``flow_arrow_scale`` schema bounds.
_ARROW_LENGTH_MIN = 0.1
_ARROW_LENGTH_MAX = 5.0


def _vectors_layer_uses_arrow_length(layer) -> bool:
    """Triangle-style HaemoLynx Vectors layers scale heads via ``length``."""
    if layer is None or not _is_ours(layer):
        return False
    if layer.__class__.__name__ != "Vectors":
        return False
    return getattr(layer, "vector_style", None) == "triangle"


def _stored_arrow_length(layer) -> float | None:
    tag = getattr(layer, "metadata", {}).get(OURS) or {}
    if tag.get("arrow_length_user_set"):
        value = tag.get("arrow_length")
        if value is not None:
            return float(value)
    return None


def _remember_arrow_length(layer, length: float) -> None:
    metadata = dict(getattr(layer, "metadata", {}) or {})
    tag = dict(metadata.get(OURS) or {})
    tag["arrow_length"] = float(length)
    tag["arrow_length_user_set"] = True
    metadata[OURS] = tag
    layer.metadata = metadata


def _sync_arrow_length_slider(slider, layer, host=None) -> None:
    """Show the slider only for flow-direction layers; mirror ``layer.length``."""
    if not _vectors_layer_uses_arrow_length(layer):
        slider.setEnabled(False)
        slider.setVisible(False)
        if host is not None:
            host.setVisible(False)
        return
    if host is not None:
        host.setVisible(True)
    slider.setVisible(True)
    slider.setEnabled(True)
    value = _stored_arrow_length(layer)
    if value is None:
        value = float(getattr(layer, "length", 1.0))
    value = max(_ARROW_LENGTH_MIN, min(_ARROW_LENGTH_MAX, value))
    slider.blockSignals(True)
    try:
        slider.setValue(value)
    finally:
        slider.blockSignals(False)


def _form_layout_row_for_widget(layout, widget) -> int:
    from qtpy.QtWidgets import QFormLayout

    for row in range(layout.rowCount()):
        for role in (
            QFormLayout.SpanningRole,
            QFormLayout.LabelRole,
            QFormLayout.FieldRole,
        ):
            item = layout.itemAt(row, role)
            if item is not None and item.widget() is widget:
                return row
    return -1


def _reparent_arrow_length_slider(
    viewer,
    *,
    slider,
    host,
    parking,
    controls_holder: dict[str, Any],
    z_depth_host=None,
) -> None:
    """Mount the shared arrow-size row under the Z depth row when both apply."""
    _park_z_depth_host(host, parking)
    controls_holder["controls"] = None

    if viewer is None:
        host.setVisible(False)
        return

    active = viewer.layers.selection.active
    if not _vectors_layer_uses_arrow_length(active):
        host.setVisible(False)
        return

    controls = _layer_controls(viewer, active)
    if controls is None:
        host.setVisible(False)
        return

    layout = controls.layout()
    if layout is None or not hasattr(layout, "insertRow"):
        host.setVisible(False)
        return

    insert_at = 0
    if (
        z_depth_host is not None
        and z_depth_host.isVisible()
        and z_depth_host.parentWidget() is controls
    ):
        z_row = _form_layout_row_for_widget(layout, z_depth_host)
        if z_row >= 0:
            insert_at = z_row + 1
    layout.insertRow(insert_at, host)
    controls_holder["controls"] = controls

    _sync_arrow_length_slider(slider, active, host)


def _apply_vectors_length(existing, spec) -> None:
    """Apply spec length unless the user already chose one on the slider."""
    if "length" not in spec.options:
        return
    stored = _stored_arrow_length(existing)
    existing.length = stored if stored is not None else spec.options["length"]


def _maybe_store_z_filter_cache(layer, spec) -> None:
    if is_z_depth_filtered_layer(spec.name, spec.kind):
        _store_z_filter_cache(
            layer,
            spec.data,
            spec.features,
            segment_owner=getattr(spec, "segment_owner", None),
        )


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
            try:
                _attach_sweep_sliders(viewer, viewer.layers[spec.name], spec)
            except Exception:  # noqa: BLE001 - missing sliders are survivable
                logger.debug("could not attach sweep sliders to %s",
                             spec.name, exc_info=True)
            try:
                _attach_branch_hover_controls(viewer, viewer.layers[spec.name])
            except Exception:  # noqa: BLE001 - missing hover panel is survivable
                logger.debug("could not attach branch-hover controls to %s",
                             spec.name, exc_info=True)
    for name, column in group.recolour:
        layer = viewer.layers[name] if name in viewer.layers else None
        if layer is not None and _is_ours(layer):
            _colour_layer(layer, column)
    if group.ndisplay is not None:
        viewer.dims.ndisplay = group.ndisplay
    reparent = getattr(viewer, "_haemolynx_reparent_z_depth_slider", None)
    if reparent is not None:
        try:
            reparent()
        except Exception:  # noqa: BLE001 - missing controls are survivable
            logger.debug("could not reparent Z depth slider", exc_info=True)
    reparent_arrow = getattr(viewer, "_haemolynx_reparent_arrow_length_slider", None)
    if reparent_arrow is not None:
        try:
            reparent_arrow()
        except Exception:  # noqa: BLE001 - missing controls are survivable
            logger.debug("could not reparent arrow length slider", exc_info=True)
    after = getattr(viewer, "_haemolynx_after_layers_applied", None)
    if after is not None:
        try:
            after()
        except Exception:  # noqa: BLE001 - missing Z filter is survivable
            logger.debug("could not re-apply Z depth filter", exc_info=True)
    if report is not None and group.note:
        report.value = f"{group.title}: {group.note}"


def _colour_attribute(layer) -> str:
    """Where a layer keeps its colour: vessels on the edge, points on the face."""
    return "edge_color" if layer.__class__.__name__ == "Vectors" else "face_color"


def _colour_attributes(layer) -> tuple[str, ...]:
    """Every attribute that has to be set for a layer to change colour.

    A Shapes layer needs both. A line has no face, so a box outline drawn as
    twelve line shapes kept napari's default white however its faces were
    coloured -- the only thing that took the role's colour was the handle
    rectangle's translucent fill.
    """
    if layer.__class__.__name__ == "Shapes":
        return ("face_color", "edge_color")
    return (_colour_attribute(layer),)


def _categorical_colours(layer, column: str, cycle) -> np.ndarray:
    """One RGBA row per item, looked up from *cycle* by the item's label.

    A value the cycle does not name -- a role from a newer config, a blank in a
    half-filled column -- takes the uncoloured grey rather than borrowing some
    other label's colour.
    """
    lookup = {label: colour for label, colour in cycle}
    values = layer.features[column]
    return np.array(
        [lookup.get(value, UNCOLOURED_RGBA) for value in values], dtype=float
    )


def _colour_layer(layer, column: str | None, kind: str = "continuous",
                  cycle=(), limits=None) -> None:
    """Colour a layer by one of its feature columns."""
    attributes = _colour_attributes(layer)
    if column is None:
        # No opinion: a stage that does not name a colouring means "leave what
        # is there", not "blank it". Most stages after build_network have
        # nothing to say about colour, and clearing on each would throw away
        # the previous stage's colouring every time.
        return
    if column in {"", "none"}:
        # An explicit "no colouring", which has to be a real branch: leaving the
        # layer as it was would make picking "none" a control that does nothing.
        for attribute in attributes:
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
    for attribute in attributes:
        setattr(layer, attribute, UNCOLOURED)
    if kind == "categorical" and cycle:
        # One colour per item, looked up by label, rather than handing napari
        # the cycle and the column and letting it pair them up. It pairs them
        # by the order the values are first *encountered*, not by the labels
        # they were declared against, so a layer holding only outlet nodes drew
        # them in the inlet colour -- the first colour in the cycle. Points
        # and Shapes disagree about that order too, so there is no ordering
        # that would be right for both.
        colours = _categorical_colours(layer, column, cycle)
        for attribute in attributes:
            setattr(layer, attribute, colours)
    else:
        try:
            values = np.asarray(layer.features[column], dtype=float)
        except (TypeError, ValueError):
            values = np.asarray([], dtype=float)
        if values.size == 0 or not np.any(np.isfinite(values)):
            # All-NaN or empty: stay flat grey. Naming the column in colormap
            # mode still makes napari map it; on some builds that aborts Qt
            # rather than raising, which `_apply_layers` cannot catch.
            _record_colour(layer, column)
            return
        for attribute in attributes:
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

    Points layers only get a feature dropdown from us; when a column is all-NaN
    we record the choice in metadata without entering napari's colormap mode,
    so the recorded name is the authoritative answer there.
    """
    tag = getattr(layer, "metadata", {}).get(OURS)
    recorded = tag.get("colour_by") if isinstance(tag, dict) else None
    if recorded and _colour_attribute(layer) == "face_color":
        return str(recorded)
    manager = getattr(layer, "_edge", None) or getattr(layer, "_face", None)
    properties = getattr(manager, "color_properties", None)
    name = getattr(properties, "name", None)
    mode = getattr(layer, f"{_colour_attribute(layer)}_mode", None)
    if name and mode != "direct":
        return str(name)
    if recorded:
        return str(recorded)
    return str(name) if name else None


def _image_options_for_napari(options: Mapping[str, Any]) -> dict[str, Any]:
    """Napari kwargs for an Image layer, expanding our ``mask_colour`` marker.

    Vessel-mask specs stay napari-free: they carry an RGBA tuple under
    ``mask_colour``. Here that becomes a two-stop colormap (transparent at 0,
    the role colour at 1) so binary volumes render as translucent overlays.
    """
    prepared = dict(options)
    colour = prepared.pop("mask_colour", None)
    if colour is None:
        return prepared
    from napari.utils.colormaps import Colormap

    rgba = tuple(float(c) for c in colour)
    prepared["colormap"] = Colormap(
        [[0.0, 0.0, 0.0, 0.0], list(rgba)],
        name="haemolynx_vessel_mask",
    )
    return prepared


def _add_or_update(viewer, spec) -> None:
    """Add *spec*, or update the layer of ours already carrying its name."""
    import pandas as pd  # noqa: F401  (napari builds features through pandas)

    existing = viewer.layers[spec.name] if spec.name in viewer.layers else None
    if existing is not None and not _is_ours(existing):
        # Someone else's layer happens to share the name. Never overwrite it.
        spec = replace(spec, name=f"{spec.name} (HaemoLynx)")
        existing = viewer.layers[spec.name] if spec.name in viewer.layers else None

    if existing is not None and existing.__class__.__name__.lower() == _CLASS_FOR[spec.kind]:
        # Shrinking Vectors/Points in place leaves stale segments on screen.
        if spec.kind in {"vectors", "points"}:
            new_count = len(np.asarray(spec.data))
            old_count = len(np.asarray(existing.data))
            if new_count < old_count:
                viewer.layers.remove(existing)
                existing = None
        if existing is not None:
            # A Shapes layer applies the types it already holds to whatever data
            # it is next given, so handing a box outline to a layer holding one
            # rectangle raises "Rectangle expects four corner vertices, 2
            # provided" -- after it has emptied itself, which loses the region.
            # The spec knows what each shape is, so the two go in together.
            shape_type = spec.options.get("shape_type") if spec.kind == "shapes" else None
            if shape_type is not None:
                existing.data = []
                existing.add(list(spec.data), shape_type=list(shape_type))
            else:
                existing.data = spec.data
            if spec.features:
                existing.features = dict(spec.features)
            existing.visible = spec.visible
            if spec.kind == "image" and "mask_colour" in spec.options:
                image_opts = _image_options_for_napari(spec.options)
                if "colormap" in image_opts:
                    existing.colormap = image_opts["colormap"]
                if spec.contrast_limits is not None:
                    existing.contrast_limits = spec.contrast_limits
                for key in ("blending", "opacity", "rendering"):
                    if key in image_opts:
                        setattr(existing, key, image_opts[key])
            if spec.kind == "vectors":
                for key in ("edge_width", "vector_style", "out_of_slice_display"):
                    if key in spec.options:
                        setattr(existing, key, spec.options[key])
                _apply_vectors_length(existing, spec)
            _colour_layer(existing, spec.colour_by, spec.colour_kind,
                          spec.colour_cycle, spec.contrast_limits)
            _store_sweep_metadata(existing, spec)
            _store_branch_hover_metadata(existing, spec)
            _maybe_store_z_filter_cache(existing, spec)
            return

    if existing is not None:
        viewer.layers.remove(existing)

    adder = getattr(viewer, f"add_{spec.kind}")
    options = dict(spec.options)
    if spec.kind == "image":
        options = _image_options_for_napari(options)
    for key in _BRANCH_HOVER_OPTION_KEYS:
        options.pop(key, None)
    if spec.features:
        options["features"] = dict(spec.features)
    add_kwargs = {
        "name": spec.name,
        "scale": spec.scale,
        "visible": spec.visible,
        "metadata": {OURS: {"kind": spec.kind}},
        **options,
    }
    if spec.kind == "image" and spec.contrast_limits is not None:
        add_kwargs.setdefault("contrast_limits", spec.contrast_limits)
    layer = adder(spec.data, **add_kwargs)
    _colour_layer(layer, spec.colour_by, spec.colour_kind,
                  spec.colour_cycle, spec.contrast_limits)
    _store_sweep_metadata(layer, spec)
    _store_branch_hover_metadata(layer, spec)
    _maybe_store_z_filter_cache(layer, spec)


def _store_sweep_metadata(layer, spec) -> None:
    """Keep sweep grid + indexing on the layer so sliders can refresh features."""
    tag = dict(getattr(layer, "metadata", {}).get(OURS) or {})
    tag["kind"] = getattr(spec, "kind", tag.get("kind"))
    if getattr(spec, "sweep", None) is not None:
        tag["sweep"] = spec.sweep
        tag["segment_owner"] = spec.segment_owner
        tag["sweep_edge_index"] = spec.sweep_edge_index
        tag["colour_by"] = spec.colour_by
        tag["contrast_limits"] = spec.contrast_limits
    else:
        tag.pop("sweep", None)
        tag.pop("segment_owner", None)
        tag.pop("sweep_edge_index", None)
    metadata = dict(getattr(layer, "metadata", {}) or {})
    metadata[OURS] = tag
    layer.metadata = metadata


def _store_branch_hover_metadata(layer, spec) -> None:
    """Remember which hover metrics a stage offered, for the controls panel."""
    available = spec.options.get("branch_hover_available")
    selected = spec.options.get("branch_hover_selected")
    if available is None and selected is None:
        return
    tag = dict(getattr(layer, "metadata", {}).get(OURS) or {})
    tag["kind"] = getattr(spec, "kind", tag.get("kind"))
    if available is not None:
        tag["branch_hover_available"] = tuple(available)
    if selected is not None:
        tag["branch_hover_selected"] = tuple(selected)
    metadata = dict(getattr(layer, "metadata", {}) or {})
    metadata[OURS] = tag
    layer.metadata = metadata


#: Axis name -> short slider label.
_SWEEP_AXIS_LABELS = {
    "dilation_percent": "Constriction/dilation %",
    "inlet_pressure_pa": "Inlet pressure (Pa)",
    "constriction_spacing_um": "Spacing (µm)",
    "constriction_length_um": "Length (µm)",
}


def _sweep_axis_label(name: str) -> str:
    return _SWEEP_AXIS_LABELS.get(str(name), str(name).replace("_", " "))


def _apply_sweep_index(layer, indices: tuple[int, ...]) -> None:
    """Swap flow feature columns for *indices* without rebuilding geometry."""
    import numpy as np

    tag = getattr(layer, "metadata", {}).get(OURS) or {}
    sweep = tag.get("sweep")
    owner = tag.get("segment_owner")
    edge_index = tag.get("sweep_edge_index")
    if sweep is None or owner is None or edge_index is None:
        return

    owner = np.asarray(owner, dtype=int)
    edge_index = np.asarray(edge_index, dtype=int)
    features = dict(layer.features)

    def _segment_column(edge_values) -> np.ndarray:
        return np.asarray(edge_values, dtype=float)[edge_index][owner]

    features["flow_abs"] = _segment_column(sweep.flow_abs_at(*indices))
    signed = sweep.flow_signed_at(*indices)
    if signed is not None and "flow_signed" in features:
        features["flow_signed"] = _segment_column(signed)
    drop = sweep.pressure_drop_at(*indices)
    if drop is not None and "pressure_drop" in features:
        features["pressure_drop"] = _segment_column(drop)
    layer.features = features

    colour_by = tag.get("colour_by") or "flow_abs"
    limits = tag.get("contrast_limits")
    if limits is None and colour_by == "flow_abs":
        limits = sweep.global_flow_abs_limits()
    _colour_layer(layer, colour_by, "continuous", (), limits)


def _attach_sweep_sliders(viewer, layer, spec) -> None:
    """One or two integer sliders for a sweep Vectors layer."""
    sweep = getattr(spec, "sweep", None)
    if sweep is None:
        return

    from magicgui.widgets import Container, Label, Slider

    dock_name = f"{spec.name} sweep"
    # Drop a previous dock for this layer on re-run.
    try:
        window = viewer.window
        for dock in list(getattr(window, "_dock_widgets", {}).values()):
            if getattr(dock, "objectName", lambda: "")() == dock_name or (
                hasattr(dock, "windowTitle") and dock.windowTitle() == dock_name
            ):
                window.remove_dock_widget(dock)
    except Exception:  # noqa: BLE001
        logger.debug("could not remove prior sweep dock %s", dock_name, exc_info=True)

    sliders: list = []
    value_labels: list = []
    for axis_name in sweep.axis_names:
        values = list(sweep.axis_values[axis_name])
        slider = Slider(
            value=0,
            min=0,
            max=max(len(values) - 1, 0),
            step=1,
            label=_sweep_axis_label(axis_name),
        )
        readout = Label(value=_format_sweep_value(axis_name, values[0] if values else 0))
        sliders.append((axis_name, slider, values, readout))
        value_labels.append(readout)

    def on_change(_event=None) -> None:
        indices = tuple(int(slider.value) for _name, slider, _vals, _lab in sliders)
        for (_name, slider, values, readout), index in zip(sliders, indices):
            if 0 <= index < len(values):
                readout.value = _format_sweep_value(_name, values[index])
        _apply_sweep_index(layer, indices)
        try:
            _refresh_layer_controls(viewer, layer)
            _attach_colour_scale(viewer, layer)
        except Exception:  # noqa: BLE001
            logger.debug("sweep slider refresh failed for %s", layer.name, exc_info=True)

    for _name, slider, _values, _readout in sliders:
        slider.changed.connect(on_change)

    rows = []
    for _name, slider, _values, readout in sliders:
        rows.append(slider)
        rows.append(readout)
    container = Container(widgets=rows, layout="vertical", labels=True)
    try:
        dock = viewer.window.add_dock_widget(
            container, name=dock_name, area="right", allowed_areas=["right", "left"]
        )
        try:
            dock.setObjectName(dock_name)
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        logger.debug("could not dock sweep sliders for %s", spec.name, exc_info=True)


def _format_sweep_value(axis_name: str, value) -> str:
    name = str(axis_name)
    if "percent" in name:
        return f"{value:g} %"
    if "pressure" in name:
        return f"{value:g} Pa"
    if name.endswith("_um"):
        return f"{value:g} µm"
    return f"{value:g}"


#: Spec kind -> the napari class name it becomes, for "is this the same sort of
#: layer I already have?".
_CLASS_FOR = {
    "image": "image", "labels": "labels", "points": "points",
    "vectors": "vectors", "shapes": "shapes",
}


#: Identifiers rather than quantities: colouring by one shows nothing.
NOT_WORTH_COLOURING_BY = frozenset(
    {"u", "v", "key", "edge_index", "node_id", "tooltip", "branch_id"}
)

#: Preferred order for flow-related columns in the colour-by dropdown.
_FLOW_COLOUR_COLUMN_ORDER = (
    "flow_abs",
    "flow_abs_log10",
    "flow_signed",
    "flow_dir_z",
    "flow_dir_y",
    "flow_dir_x",
    "pressure_drop",
    "pressure_u",
    "pressure_v",
)


def _colour_by_columns(layer) -> list[str]:
    """Feature columns worth offering for colouring, in a sensible order."""
    names = [
        name for name in getattr(layer, "features", {})
        if name not in NOT_WORTH_COLOURING_BY
    ]
    order = {name: index for index, name in enumerate(_FLOW_COLOUR_COLUMN_ORDER)}
    return sorted(names, key=lambda name: (order.get(name, len(order)), name))

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
    them, so choosing it tried to map "inlet" and "outlet" onto a colormap
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
        if getattr(self, "_following", False):
            # Re-entered from an edge/face_color event we triggered in
            # autoscale or _apply_contrast_limits: refresh only, no autoscale.
            layer = self._layer()
            column = _active_column(layer) if layer is not None else None
            self.refresh(column)
            return
        self._following = True
        try:
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
        finally:
            self._following = False

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
    """The dropdown napari gives Vectors once, and does not give Points.

    `QtVectorsControls` has an "edge feature:" box filled in its constructor
    and never updated when ``layer.features`` grows -- which is exactly why
    ``flow_abs`` written by the solve never appeared there. Worse, colouring
    in direct mode (RGB arrays for categorical columns) hides that box
    entirely. So HaemoLynx adds its own "Colour by" combo, rebuilt whenever the
    layer's columns change.

    Points layers get the same treatment because `QtPointsControls` has a
    colour swatch and no feature picker at all.
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
        columns = _colour_by_columns(layer)
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
    attribute = _colour_attribute(layer)
    if attribute in {"edge_color", "face_color"}:
        label = "Colour by:" if attribute == "edge_color" else "node feature:"
        chooser = _FeatureChooser(viewer, layer.name)
        layout.addRow(QLabel(label), chooser.native)
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
    for attribute in (
        "_haemolynx_feature",
        "_haemolynx_scale",
        "_haemolynx_branch_hover",
    ):
        widget = getattr(controls, attribute, None)
        if widget is None:
            continue
        if hasattr(widget, "follow_the_layer"):
            widget.follow_the_layer()
        else:
            widget.refresh()


def _is_branch_hover_layer(layer) -> bool:
    """Whether *layer* is our branch-hover Points layer."""
    if not _is_ours(layer):
        return False
    if layer.name == BRANCH_HOVER or layer.name.startswith(f"{BRANCH_HOVER} "):
        return True
    features = getattr(layer, "features", {}) or {}
    return "tooltip" in features and "branch_id" in features


def _branch_hover_available(layer) -> tuple[str, ...]:
    """Optional metrics this hover layer can offer right now."""
    from haemolynx.gui.branch_hover import available_metrics_from_features

    tag = getattr(layer, "metadata", {}).get(OURS) or {}
    stored = tag.get("branch_hover_available")
    if stored is not None:
        return tuple(stored)
    return available_metrics_from_features(getattr(layer, "features", {}) or {})


def _branch_hover_selected_for(layer) -> tuple[str, ...]:
    """Checkbox selection: session choice filtered by what is available."""
    from haemolynx.gui.branch_hover import (
        default_selected_metrics,
        filter_selected_metrics,
    )

    available = _branch_hover_available(layer)
    global _branch_hover_session_selected
    if _branch_hover_session_selected is None:
        return default_selected_metrics(available)
    return filter_selected_metrics(_branch_hover_session_selected, available)


def _apply_branch_hover_selection(layer, selected: Sequence[str]) -> None:
    """Rewrite the ``tooltip`` feature column for the current checkbox set."""
    from haemolynx.gui.branch_hover import (
        filter_selected_metrics,
        tooltips_from_feature_table,
    )

    available = _branch_hover_available(layer)
    chosen = filter_selected_metrics(selected, available)
    features = dict(layer.features)
    features["tooltip"] = tooltips_from_feature_table(features, chosen)
    layer.features = features
    tag = dict(getattr(layer, "metadata", {}).get(OURS) or {})
    tag["branch_hover_selected"] = chosen
    metadata = dict(getattr(layer, "metadata", {}) or {})
    metadata[OURS] = tag
    layer.metadata = metadata


def _branch_hover_mouse_move(layer, event) -> None:
    """Show the composed tooltip string when the cursor is over a midpoint."""
    from qtpy.QtGui import QCursor
    from qtpy.QtWidgets import QToolTip

    try:
        index = layer.get_value(
            event.position,
            view_direction=getattr(event, "view_direction", None),
            dims_displayed=list(getattr(event, "dims_displayed", ())),
            world=True,
        )
    except TypeError:
        index = layer.get_value(event.position, world=True)
    if index is None:
        QToolTip.hideText()
        return
    tips = getattr(layer, "features", {}).get("tooltip")
    if tips is None:
        return
    try:
        text = str(tips[int(index)])
    except (IndexError, TypeError, ValueError):
        return
    if text:
        QToolTip.showText(QCursor.pos(), text)


def _ensure_branch_hover_callback(layer) -> None:
    """Install the mouse-move tooltip callback once per layer instance."""
    callbacks = getattr(layer, "mouse_move_callbacks", None)
    if callbacks is None:
        return
    if _branch_hover_mouse_move in callbacks:
        return
    callbacks.append(_branch_hover_mouse_move)


class _BranchHoverPanel:
    """Checkboxes for optional branch-hover metrics, in the layer controls.

    Only metrics the current graph actually carries are offered. Selection is
    remembered for the napari session so a later stage that adds flow does not
    wipe a choice the user already made.
    """

    def __init__(self, viewer, layer_name: str) -> None:
        from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget

        self._viewer = viewer
        self._layer_name = layer_name
        self._boxes: dict[str, Any] = {}
        self.native = QWidget()
        self._layout = QVBoxLayout(self.native)
        self._layout.setContentsMargins(0, 2, 0, 2)
        self._layout.setSpacing(2)
        self._heading = QLabel("branch tooltip metrics")
        self._heading.setToolTip(
            "What to show when hovering a branch midpoint. "
            "branchID is always included."
        )
        self._layout.addWidget(self._heading)
        self._box_host = QWidget()
        self._box_layout = QVBoxLayout(self._box_host)
        self._box_layout.setContentsMargins(0, 0, 0, 0)
        self._box_layout.setSpacing(1)
        self._layout.addWidget(self._box_host)
        self.offered: tuple[str, ...] = ()
        self.selected: tuple[str, ...] = ()

    def _layer(self):
        layers = getattr(self._viewer, "layers", {}) if self._viewer else {}
        if self._layer_name not in layers:
            return None
        layer = layers[self._layer_name]
        return layer if _is_branch_hover_layer(layer) else None

    def refresh(self) -> None:
        """Rebuild checkboxes for the metrics this layer can currently offer."""
        from haemolynx.gui.branch_hover import panel_metric_options
        from qtpy.QtWidgets import QCheckBox

        layer = self._layer()
        if layer is None:
            self.native.setVisible(False)
            return
        available = _branch_hover_available(layer)
        selected = _branch_hover_selected_for(layer)
        options = panel_metric_options(available)
        current_keys = tuple(key for key, _label in options)
        if tuple(self._boxes) != current_keys:
            while self._box_layout.count():
                item = self._box_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            self._boxes = {}
            for key, label in options:
                box = QCheckBox(label)
                box.setObjectName(f"branch_hover_{key}")
                box.toggled.connect(self._toggled)
                self._box_layout.addWidget(box)
                self._boxes[key] = box
        for key, box in self._boxes.items():
            with _blocked(box):
                box.setChecked(key in selected)
        _apply_branch_hover_selection(layer, selected)
        _ensure_branch_hover_callback(layer)
        self.native.setVisible(True)
        # Recorded for tests: Qt reports children of an unshown window as
        # invisible, so callers check this rather than isVisible().
        self.offered = current_keys
        self.selected = selected

    def _toggled(self, *_args) -> None:
        global _branch_hover_session_selected
        layer = self._layer()
        if layer is None:
            return
        selected = tuple(
            key for key, box in self._boxes.items() if box.isChecked()
        )
        _branch_hover_session_selected = selected
        _apply_branch_hover_selection(layer, selected)
        self.selected = _branch_hover_selected_for(layer)


def _attach_branch_hover_controls(viewer, layer) -> bool:
    """Put the metrics checkboxes on the branch-hover layer's controls, once."""
    from qtpy.QtWidgets import QLabel

    if not _is_branch_hover_layer(layer):
        return False
    controls = _layer_controls(viewer, layer)
    if controls is None:
        _ensure_branch_hover_callback(layer)
        _apply_branch_hover_selection(layer, _branch_hover_selected_for(layer))
        return False
    if getattr(controls, "_haemolynx_branch_hover", None) is not None:
        controls._haemolynx_branch_hover.refresh()
        return True
    layout = controls.layout()
    if not hasattr(layout, "addRow"):
        _ensure_branch_hover_callback(layer)
        return False
    panel = _BranchHoverPanel(viewer, layer.name)
    layout.addRow(QLabel("hover info:"), panel.native)
    controls._haemolynx_branch_hover = panel
    panel.refresh()
    return True


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
    state=None, log=None, checkpoints=None, after_layers=None):
    """Run the pipeline off the GUI thread, reporting back as it goes.

    With *viewer* and *results*, each stage's output is turned into layers as it
    finishes and shown in the viewer the run was launched from.

    *state* is the panel's :class:`~haemolynx.gui.run_state.RunState`: how the
    panel knows it has a run going, and how that run is stopped on purpose.
    Without one the run is unstoppable but otherwise unchanged, which is all a
    test driving this function on its own needs.

    *log* is a :class:`~haemolynx.gui.log_view.LogView`, or anything with its
    handful of methods -- like *report* and *bars*, it is optional and duck
    typed, so a test hands over its own. With one, the library's logger is
    captured for exactly the length of the run and drained onto the widget on
    a timer.

    *checkpoints* is a :class:`~haemolynx.gui.stage_checkpoints.StageCheckpoints`
    the panel keeps across a run so a tab can revert to an earlier stage
    without rebuilding topology. Optional: without one, the run is unchanged.
    """
    from napari.qt.threading import thread_worker

    run_state = state if state is not None else RunState()
    bridge = _progress_bridge()
    show_layers = viewer is not None and results is not None
    #: The capture, while there is one. A namespace rather than a name, because
    #: it is set below `stopped`, which is what releases it.
    capture = SimpleNamespace(attachment=None)

    def progressed(event: ProgressEvent) -> None:
        # A cancelled run's last few events are already on their way here. The
        # bars have been put back to nothing; moving them again would show a
        # run that is no longer going.
        if run_state.cancelled:
            return
        if bars is not None:
            bars.show_event(event)
        if event.kind == STAGE_STARTED:
            report.value = f"Running {event.title} ({event.index + 1}/{event.total})..."

    def shown(group) -> None:
        """One stage's layers, in the viewer -- unless the run was stopped.

        A group emitted just before the cancel is still queued for this thread,
        and applying it would put back the layers just cleared.
        """
        if run_state.cancelled:
            return
        _apply_layers(viewer, group, report)
        if after_layers is not None:
            after_layers()

    bridge.event.connect(progressed)
    if show_layers:
        bridge.layers.connect(shown)

    def watched(event: ProgressEvent) -> None:
        """Pass the run's progress on, having first let it be stopped.

        `run_pipeline_stages` takes no cancel argument, so this and `produced`
        are where a cancellation acts.         Both are called between stages, or
        between graph building's eleven topology steps, so a run stops with
        nothing half-written -- and soon after being asked, rather than at the
        end of whatever stage it is in.

        The event becomes a log record *here*, on the run's own thread, rather
        than beside the progress bars on the GUI thread. That is what gives the
        log one producer and one ordered stream: a stage's banner and the
        counts that stage logs go through the same handler, in the order they
        happened. Emitted on the other side of the bridge instead, the banner
        would arrive whenever the GUI thread got round to it, and land above or
        below its own counts at random.
        """
        run_state.check()
        log_progress(event)
        bridge.event.emit(event)

    def produced(stage: str, output) -> None:
        """Build this stage's layers here, on the run's thread.

        Eagerly, because every stage after `build_network` writes onto the same
        graph: convert later and the viewer shows a later stage's numbers under
        this stage's name. And guarded, because a fault in drawing a run must
        never end it -- an eight-hour whole-brain run least of all.

        The cancellation check is outside that guard, deliberately: stopping is
        the one thing that must get past it.
        """
        run_state.check()
        try:
            group = results.stage_finished(stage, output)
        except Exception:  # noqa: BLE001 - reported, never raised at the run
            logger.exception("could not build layers for stage %s", stage)
            return
        if checkpoints is not None:
            try:
                checkpoints.record(stage, group, results, settings=settings)
            except Exception:  # noqa: BLE001 - a bad snapshot must not end the run
                logger.exception("could not record checkpoint for stage %s", stage)
        bridge.layers.emit(group)

    @thread_worker
    def run():
        # `bridge` is captured here, which is also what keeps it alive for as
        # long as the run that emits through it.
        return run_pipeline_stages(
            settings,
            schema,
            progress=watched,
            on_stage_output=produced if show_layers else None,
        )

    def finished(graph) -> None:
        run_state.stopped()
        button.enabled = True
        if run_state.cancelled:
            # It got to the end between being asked to stop and reaching the
            # next checkpoint. Saying "finished" beside an empty viewer, or
            # "cancelled" about a run that completed, would both be wrong.
            report.value = FINISHED_FIRST
            return
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
        run_state.stopped()
        button.enabled = True
        if isinstance(error, RunCancelled):
            # Not a failure: the bars were reset when the cancel was asked
            # for, and there is no stack trace worth logging.
            report.value = CANCELLED
            return
        if bars is not None:
            bars.fail(f"Failed: {type(error).__name__}")
        report.value = f"{type(error).__name__}: {error}"
        logger.exception("pipeline run failed", exc_info=error)
        # What superqt would have done for us, and the only reason it is spelled
        # out here: connecting `errored` below is what stops it doing it to a
        # cancellation as well.
        raise error

    def release_log() -> None:
        """Stop capturing the library's log, and show the last of what it said.

        The detach has to happen on the worker's `finished`, which is the only
        one of the three that fires however a run ends: superqt suppresses
        `returned` once `quit()` has been called, so a cancelled-but-completed
        run reaches neither `finished` nor `failed`. Released in either of
        those, the handler would stay on the logger after a cancel -- and then
        double every line of the next run.

        The final drain is here for the same reason: a run's last records are
        written in the milliseconds after the last timer tick, and would
        otherwise sit in the buffer until the next run started.
        """
        if capture.attachment is not None:
            capture.attachment.detach()
            capture.attachment = None
        if log is not None:
            log.stop()

    def stopped() -> None:
        """The worker has gone without `returned` or `errored` saying so.

        napari suppresses `returned` once `quit()` has been called -- it will
        not hand back a result it was told to abandon -- so a run that reached
        its end between the cancel and its next checkpoint announces nothing at
        all. Without this the guard would stay set and the Run button greyed
        out, which is the very state being fixed.
        """
        # Above the guard, deliberately: this is the one callback that runs on
        # every way out of a run, and the guard below returns early for two of
        # the three. See `release_log`.
        release_log()
        if not run_state.running:
            return
        run_state.stopped()
        button.enabled = True
        if run_state.cancelled:
            report.value = FINISHED_FIRST

    # `errored` is connected here rather than on the worker, because superqt
    # only leaves it alone once something else has claimed it: a worker created
    # without one gets a handler that re-raises whatever the run raised. A
    # cancellation goes out through `errored` like any other exception, so that
    # handler put `RunCancelled` and a stack trace in front of the user through
    # napari's error popup -- the report this replaces with a plain "Cancelled".
    # `_start_thread=False` because `_connect` otherwise starts the run here,
    # before the state below says there is one.
    worker = run(_connect={"errored": failed}, _start_thread=False)
    worker.returned.connect(finished)
    worker.finished.connect(stopped)
    run_state.start(worker=worker, results=results)
    button.enabled = False
    if bars is not None:
        bars.start()
    if log is not None:
        # Before `worker.start()`, or the first stage's records are gone by the
        # time anything is listening -- and the capture is what makes the
        # library's INFO reach a handler at all (`gui.run_log.attach`).
        log.start()
        capture.attachment = attach(log.run_log, level=log.level)
    report.value = "Running..."
    worker.start()
    return worker


#: What the About panel says. Written here rather than in the widget so it can
#: be checked without a display -- and so the two questions it answers stay
#: answered: where the colour controls are (napari's own layer controls, not
#: this plugin's panel) and what a config file is for.
ABOUT_TEXT = """\
HaemoLynx {version}

Turns 3D microvascular microscopy into a NetworkX graph that can be fed
into a haemodynamic solver to give haemodynamic edge weights, VTK exports
and network statistics.

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

Created by Finbar Argus, Harvey Davis, and the Animus Laboratory.
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


def _boundary_controls(viewer, rows, fields, schema, report):
    """The Boundaries tab's "point at it instead of typing it" controls.

    Two layers, both editable, both holding exactly what the settings hold --
    napari world coordinates are already the microns the settings store, so the
    layers are the settings rather than a view of them. Which direction wins is
    the only rule worth stating: *the layers are authoritative while the user
    is editing them; the rows are authoritative when a config is loaded or
    "Show" is pressed.* The two directions are therefore never both live, which
    is why there is no cycle to break -- the `applying` guard below is for the
    fact that `events.data` fires twice per edit, not for a feedback loop.

    Returns None without a viewer: the panel is buildable outside napari.
    """
    if viewer is None:
        return None

    import napari
    from magicgui.widgets import ComboBox, Container, FloatSlider, Label, PushButton

    from haemolynx.gui.boundary_picking import (
        AUTOMATED_OVERRIDES_MANUAL_NOTE,
        BC_COORDINATES,
        BC_LAYER_NAMES,
        BC_REGION_NAMES,
        HANDLE,
        band_boxes,
        ROLES,
        orderable_settings,
        outside_extent,
        role_manual_controls_enabled,
        role_settings,
        role_title,
        shared_settings,
        visible_settings,
        BoundaryPicks,
        coordinate_setting,
        group_for,
        method_setting,
        settings_for_method,
        settings_from_layers,
        regions_name,
        snap,
        terminal_axis_span,
        terminal_points,
        volume_setting,
        wanted_rows,
    )
    from haemolynx.gui.chrome_tooltips import (
        ACTION_TOOLTIPS,
        SHOW_BOUNDARIES_TOOLTIP,
        SNAP_BOUNDARIES_TOOLTIP,
    )

    #: Which role a new point or region belongs to. Not shown: the sub-tab bar
    #: built in `page` is what the user sees, and this follows it. Keeping the
    #: combo as the model means everything downstream still reads one value,
    #: and a role can still be chosen without a display.
    role = ComboBox(choices=list(ROLES), value=ROLES[0], label="Role")

    #: The two controls that are about the picture rather than about one role.
    show = PushButton(text="Show these boundary conditions")
    show.tooltip = SHOW_BOUNDARIES_TOOLTIP
    snap_button = PushButton(text="Snap selected to nearest terminal")
    snap_button.tooltip = SNAP_BOUNDARIES_TOOLTIP

    #: Everything else, once per role, so it sits on that role's own page next
    #: to the settings it fills in. A control that acts on "the chosen role"
    #: from a page that is not that role's is a control that can be pressed by
    #: mistake; one per page cannot be.
    actions = {
        name: SimpleNamespace(
            pick=PushButton(text="Pick coordinates in the viewer"),
            draw=PushButton(text="Draw a region"),
            depth=FloatSlider(value=0.0, min=0.0, max=1000.0,
                              label="Region depth (um)"),
            move=PushButton(text="Move or delete what you picked"),
            assign=PushButton(text="Assign selected to this role"),
            clear=PushButton(text="Clear this role's regions"),
        )
        for name in ROLES
    }
    for _action in actions.values():
        for _control, _tip in ACTION_TOOLTIPS.items():
            getattr(_action, _control).tooltip = _tip

    #: Which of a role's controls its chosen method has any use for.
    ACTIONS_FOR_METHOD = {
        "coordinates": ("pick", "move", "assign"),
        "volume": ("draw", "depth", "move", "assign", "clear"),
    }

    state = SimpleNamespace(applying=False, results=None, connected=set(),
                        visible=frozenset(), hidden=frozenset(), tabs=None,
                        actions={})

    #: Each role's page, and where each shared row currently sits. Filled in
    #: by `page`; empty until the panel has been laid out.
    holders: dict[str, Any] = {}
    shared_home: dict[str, Any] = {}

    def depth_slider():
        """The region depth on the page the user is looking at."""
        return actions[str(role.value)].depth

    def current_values() -> dict[str, Any]:
        return {name: fields[name].to_setting_value(widget.value)
                for name, widget in rows.items()}

    def write_rows(proposed: dict[str, Any]) -> None:
        """Put values into the form, touching only the rows that would change."""
        for name, value in wanted_rows(proposed, current_values()).items():
            if name in rows:
                rows[name].value = display_value_for(schema[name], value)

    def graph():
        """The graph a run built, if there is one yet."""
        results = state.results
        return getattr(results, "graph", None) if results is not None else None

    def layer(name):
        return viewer.layers[name] if name in viewer.layers else None

    def regions_layer(which: str | None = None):
        """The regions layer of *which* role, defaulting to the open page's."""
        return layer(regions_name(which if which is not None else str(role.value)))

    def region_layers():
        """Every region layer on screen, role by role."""
        return [(name, layer(regions_name(name)))
                for name in ROLES if layer(regions_name(name)) is not None]

    def our_layer_names():
        return (BC_COORDINATES, *BC_REGION_NAMES)

    def unused_warning(values) -> str:
        """Say when a role's picks will not be read, rather than silently fixing it."""
        notes = []
        for name in ROLES:
            method = values.get(method_setting(name))
            picked = len(values.get(coordinate_setting(name)) or ())
            boxed = len(values.get(volume_setting(name)) or ())
            if picked and method != "coordinates":
                notes.append(f"{picked} {name} coordinate(s) but "
                             f"{method_setting(name)} is {method!r}")
            if boxed and method != "volume":
                notes.append(f"{boxed} {name} region(s) but "
                             f"{method_setting(name)} is {method!r}")
        return "  Not used: " + "; ".join(notes) + "." if notes else ""

    def image_extent():
        """The world box the open images occupy, in microns."""
        extents = [layer.extent.world for layer in viewer.layers
                   if layer.name not in BC_LAYER_NAMES and layer.ndim >= 3]
        if not extents:
            return None
        return (np.min([e[0] for e in extents], axis=0),
                np.max([e[1] for e in extents], axis=0))

    def band_note(bands, measured: bool) -> str:
        """Say which span a band was drawn across, because the two differ."""
        if not bands:
            return ""
        if measured:
            return "  Bands drawn across the terminals, as a run measures them."
        return (
            "  Bands drawn across the image: a run measures them across the "
            "terminals instead, and a network rarely reaches its image's edge, "
            "so run '3. Graph' to see where they really fall."
        )

    def offscreen_warning(values) -> str:
        """Say when coordinates fall outside the image rather than drawing them there."""
        box = image_extent()
        if box is None:
            return ""
        stray = outside_extent(values, *box)
        if not stray:
            return ""
        return (
            "  Outside the image: " + "; ".join(stray) + ". These settings are "
            "microns, not voxel indices -- if they came from a viewer showing "
            f"indices, multiply by the voxel size. The image spans "
            f"{tuple(round(float(v), 1) for v in box[1])} um (z, y, x)."
        )

    def bands_now(values):
        """The slab each `edge_percent` role selects from, if it can be drawn.

        Returns the boxes and whether they are the real thing: a run measures
        the band across the terminals, so before a graph exists the image has
        to stand in and the report has to say so.
        """
        box = image_extent()
        if box is None:
            return {}, True
        axis = values.get("boundary_axis")
        span = terminal_axis_span(graph(), axis) if graph() is not None else None
        return band_boxes(values, *box, axis_span=span), span is not None

    def redraw() -> None:
        """Settings -> layers. One of the two authoritative directions.

        Guarded, because writing a layer fires its own `events.data`: without
        this the redraw's half-applied layer would be read straight back as if
        the user had edited it, and a role whose features had not been written
        yet would lose its boxes.
        """
        if state.applying:
            return
        state.applying = True
        try:
            values = current_values()
            bands, measured = bands_now(values)
            group = group_for(values, bands)
            _apply_layers(viewer, group, report)
            report.value = (f"Boundary conditions: {group.note}"
                            f"{band_note(bands, measured)}"
                            f"{unused_warning(values)}{offscreen_warning(values)}")
            for name in our_layer_names():
                listen(layer(name))
            drawn = {spec.name for spec in group.layers}
            for name in BC_REGION_NAMES:
                # `_apply_layers` only ever adds and updates, so a role that
                # has nothing left to draw would keep the layer it had --
                # unless a tool is pointed at it. Pressing Draw makes an empty
                # layer to draw into, and taking that away would leave the
                # next click with nowhere to land. Being merely selected is
                # not enough: napari selects whatever was added last.
                stale = layer(name)
                if (name not in drawn and stale is not None
                        and getattr(stale, "mode", "pan_zoom") == "pan_zoom"):
                    viewer.layers.remove(stale)
            set_depth_range()
        finally:
            state.applying = False

    def sync(*_args) -> None:
        """Layers -> settings. The other authoritative direction."""
        if state.applying:
            return
        state.applying = True
        try:
            points_layer = layer(BC_COORDINATES)
            proposed: dict[str, Any] = {}
            if points_layer is not None:
                proposed.update(settings_from_layers(
                    points=points_layer.data,
                    point_roles=roles_of(points_layer, len(points_layer.data)),
                ))
            drawn = region_layers()
            if drawn:
                # One call across every layer, not one per layer: each call
                # writes all four roles' lists, so writing per layer would
                # empty every role but the last one read.
                rectangles, rectangle_roles, depths = [], [], []
                for owner, target in drawn:
                    handles = handle_indices(target)
                    roles = roles_of(target, len(target.data), default=owner)
                    found = depths_of(target, len(target.data))
                    rectangles += [target.data[index] for index in handles]
                    rectangle_roles += [roles[index] for index in handles]
                    depths += [found[index] for index in handles]
                proposed.update(settings_from_layers(
                    rectangles=rectangles,
                    rectangle_roles=rectangle_roles,
                    depths=depths,
                ))
            write_rows(proposed)
            values = current_values()
            picks = BoundaryPicks.from_settings(values)
            report.value = (f"Boundary conditions: {picks.summary()}"
                            f"{unused_warning(values)}{offscreen_warning(values)}")
        finally:
            state.applying = False

    def handle_indices(target) -> list[int]:
        """Which shapes are the editable rectangles, not the box outlines.

        A region is drawn as a rectangle plus the twelve segments of the box it
        stands for. Only the rectangle is a region; reading the segments back
        would turn one box into thirteen.
        """
        parts = list(target.features.get("part", [])) if len(target.features) else []
        kinds = list(target.shape_type)
        return [
            index
            for index in range(len(target.data))
            # A rectangle the user has just drawn may have no `part` yet, and
            # an unfilled column reads back as NaN rather than as a string.
            if (parts[index] if index < len(parts)
                and isinstance(parts[index], str) else HANDLE) == HANDLE
            and kinds[index] != "line"
        ]

    def roles_of(target, count, default=None) -> list[str]:
        """Each item's role, filling anything unlabelled with the chosen one.

        A point added by napari's own tool arrives with whatever
        `feature_defaults` said, and a point added any other way arrives with
        nothing -- so the blank is filled here rather than relied upon there.
        """
        values = list(target.features.get("role", [])) if len(target.features) else []
        out = []
        for index in range(count):
            found = values[index] if index < len(values) else None
            out.append(str(found) if found in ROLES
                       else str(default if default is not None else role.value))
        return out

    def depths_of(target, count) -> list[float]:
        values = list(target.features.get("depth", [])) if len(target.features) else []
        out = []
        for index in range(count):
            found = values[index] if index < len(values) else None
            try:
                depth = float(found)
            except (TypeError, ValueError):
                depth = float("nan")
            # NaN is what an untyped feature column hands back for a shape
            # napari added itself, and a NaN depth makes a NaN box.
            out.append(depth if math.isfinite(depth) else float(depth_slider().value))
        return out

    def listen(target) -> None:
        """Follow a layer's edits, once per layer."""
        if target is None or id(target) in state.connected:
            return
        target.events.data.connect(sync)
        target.mouse_drag_callbacks.append(redraw_when_the_drag_ends)
        state.connected.add(id(target))

    def redraw_when_the_drag_ends(_layer, event):
        """Redraw once the mouse comes up, not on every step of the drag.

        A region drawn or moved by hand is one rectangle until it is drawn
        from the settings again, so without this the box a user just made has
        no depth on screen -- and redrawing on every `events.data` would be
        replacing the layer's contents underneath the drag that is producing
        them.
        """
        yield
        while event.type == "mouse_move":
            yield
        redraw()

    def set_defaults(target) -> None:
        """Tag whatever napari adds next as this role's, and as a handle.

        Every column the layer has, not just the ones this cares about: a
        default that names a subset is refused outright, and the failure is
        silent -- the next shape then arrives as the role that was chosen
        before, with no `part`, which stops it being read back at all.
        """
        known = {
            "role": str(role.value),
            "depth": float(depth_slider().value),
            # What the user draws by hand is the region itself; the outline
            # segments are only ever made from the settings.
            "part": HANDLE,
        }
        defaults = {name: value for name, value in known.items()
                    if name in target.features}
        try:
            target.feature_defaults = defaults
        except Exception:  # noqa: BLE001 - `roles_of` fills the blank anyway
            logger.debug("could not set feature defaults on %s", target.name,
                         exc_info=True)

    def set_depth_range() -> None:
        """Default the depth to the whole stack: a boundary band usually is."""
        extents = [l.extent.world for l in viewer.layers
                   if l.name not in BC_LAYER_NAMES and l.ndim >= 3]
        if not extents:
            return
        span = max(float(e[1][0] - e[0][0]) for e in extents)
        if span <= 0:
            return
        for slider in (action.depth for action in actions.values()):
            state.applying = True
            try:
                slider.max = max(span, float(slider.value))
            finally:
                state.applying = False
            if slider.value == 0.0:
                # Guarded: assigning the slider fires `on_depth_changed`, and
                # picking a default is not the user resizing anything. Clamped
                # to what the slider actually took: it rounds the maximum it
                # was given, and a value past that raises rather than clips.
                state.applying = True
                try:
                    slider.value = min(span, float(slider.max))
                finally:
                    state.applying = False

    def row_order(names: Sequence[str]) -> list[str]:
        """The Boundaries tab, grouped so a method is followed by what it reads.

        The schema lists all four methods, then all four coordinate lists, then
        all four volume lists -- a fine way to declare them and a poor way to
        read them. Here each role's method is followed immediately by the
        settings that method uses, so the row you have to fill in sits under
        the row that decided you have to fill it in.
        """
        remaining = list(names)
        ordered: list[str] = []
        for name in orderable_settings():
            if name in remaining:
                remaining.remove(name)
                ordered.append(name)
        return ordered + remaining

    def refresh_rows(*_args) -> None:
        """Show only the settings the chosen methods will actually read."""
        wanted = visible_settings(current_values())
        methods = {method_setting(role) for role in ROLES}
        # A shared row's visibility belongs to `place_shared`, which knows
        # whether it is on a page at all. Showing one that is on no page makes
        # a parentless Qt widget visible, and a visible widget with no parent
        # is a window: "Boundary last percent (percent)", floating on its own.
        owned = set(shared_settings())
        hidden = set()
        for name in orderable_settings():
            # A method row always stays: it is the row that decides which of
            # the others you need to fill in.
            if name in rows and name not in methods and name not in owned:
                rows[name].visible = name in wanted
                if name not in wanted:
                    hidden.add(name)
        refresh_actions()
        refresh_role_tabs()
        place_shared()
        state.visible = wanted
        state.hidden = frozenset(hidden | {name for name in shared_settings()
                                           if shared_home.get(name) is None})

    def place_shared() -> None:
        """Put a shared row on the page of whoever is reading it right now.

        One axis and one pair of bands describe the whole network, so there is
        one row each and Qt gives it one parent -- it cannot sit on all four
        pages at once. Moving it to the page being looked at is the next best
        thing, and better than a section underneath: the row is always beneath
        the method that asked for it, and there is never a second copy of a
        setting to disagree with the first.
        """
        if not holders:
            return
        current = str(role.value)
        reads = settings_for_method(
            current, current_values().get(method_setting(current))
        )
        wanted = [name for name in shared_settings()
                  if name in rows and name in reads]
        for name in shared_settings():
            home = shared_home.get(name)
            if home is None or (name in wanted and home == current):
                continue
            holders[home].remove(rows[name])
            # Removed from its page it has no parent, and a visible widget
            # with no parent is a window of its own.
            rows[name].visible = False
            shared_home[name] = None
        for offset, name in enumerate(wanted):
            if shared_home.get(name) is None:
                # Straight under the method row, which is the first on a page.
                holders[current].insert(1 + offset, rows[name])
                rows[name].visible = True
                shared_home[name] = current

    def refresh_role_tabs() -> None:
        """Grey out role sub-tabs when automated assignment owns that role.

        Large-vessel auto (``automated_vessel_assignment``) disables Inlet and
        Outlet; small-vessel auto disables Arteriole and Venule. Tabs stay
        visible -- greyed, not hidden -- unlike vessel-mask option rows.
        """
        tabs = getattr(state, "tabs", None)
        if tabs is None:
            return
        values = current_values()
        for index, name in enumerate(ROLES):
            enabled = role_manual_controls_enabled(name, values)
            tabs.setTabEnabled(index, enabled)
            if not enabled:
                if name in ("inlet", "outlet"):
                    tabs.setTabToolTip(index, AUTOMATED_OVERRIDES_MANUAL_NOTE)
                else:
                    tabs.setTabToolTip(
                        index,
                        "Small-vessel mask assignment overrides manual "
                        "arteriole/venule boundary selection.",
                    )
            else:
                tabs.setTabToolTip(index, fields[method_setting(name)].help)
        if not tabs.isTabEnabled(tabs.currentIndex()):
            for index in range(tabs.count()):
                if tabs.isTabEnabled(index):
                    tabs.setCurrentIndex(index)
                    break

    def refresh_actions() -> None:
        """Show a role's controls only where its method has a use for them."""
        values = current_values()
        drawable = viewer.dims.ndisplay == 2
        for name, action in actions.items():
            method = str(values.get(method_setting(name)))
            useful = set(ACTIONS_FOR_METHOD.get(method, ()))
            # Automated assignment overrides the matching manual role tabs.
            overridden = not role_manual_controls_enabled(name, values)
            for control in ("pick", "draw", "depth", "move", "assign", "clear"):
                widget = getattr(action, control)
                widget.visible = control in useful
                if overridden:
                    widget.enabled = False
                    if name in ("inlet", "outlet"):
                        widget.tooltip = AUTOMATED_OVERRIDES_MANUAL_NOTE
                    else:
                        widget.tooltip = (
                            "Small-vessel mask assignment overrides manual "
                            "arteriole/venule boundary selection."
                        )
                elif control == "draw" and not drawable:
                    widget.enabled = False
                    widget.tooltip = (
                        "napari cannot edit a Shapes layer in the "
                        "3D view. Switch to 2D to draw a region."
                    )
                else:
                    widget.enabled = True
                    widget.tooltip = ACTION_TOOLTIPS[control]
            state.actions[name] = frozenset(useful)

    def on_settings_changed(*_args) -> None:
        """Follow the form: the layers show what the settings currently say."""
        refresh_rows()
        if state.applying:
            return
        if any(name in viewer.layers for name in our_layer_names()):
            redraw()

    def on_show() -> None:
        redraw()

    def on_pick() -> None:
        redraw()
        target = layer(BC_COORDINATES)
        if target is None:
            return
        viewer.layers.selection.active = target
        set_defaults(target)
        target.mode = "add"
        report.value = (
            f"Click in the viewer to place {role.value} coordinates. "
            "Drag one to move it, select and press Delete to remove it."
        )

    def on_draw() -> None:
        if viewer.dims.ndisplay == 3:
            report.value = (
                "Regions can only be drawn in the 2D view -- napari does not "
                "allow editing a Shapes layer in 3D. Switch to 2D, draw the "
                "rectangle, then come back to 3D to see the box it makes."
            )
            return
        redraw()
        target = regions_layer()
        if target is None:
            target = viewer.add_shapes(
                name=regions_name(str(role.value)), ndim=3, scale=(1, 1, 1),
                # Typed, not `[]`: an empty list makes a float64 column, and
                # a role written into one comes back NaN -- which then reaches
                # a settings row as `nan`, and `literal_eval` cannot read that
                # row ever again.
                features={"role": np.empty(0, dtype=object),
                          "depth": np.empty(0, dtype=float)},
                metadata={OURS: {"kind": "shapes"}},
                edge_width=2.0, opacity=0.3,
            )
            listen(target)
        viewer.layers.selection.active = target
        set_defaults(target)
        target.mode = "add_rectangle"
        report.value = (
            f"Draw a rectangle for a {role.value} region. It takes the depth "
            f"below ({depth_slider().value:.0f} um), centred on the slice you draw it on."
        )

    def on_depth_changed(*_args) -> None:
        """How deep this role's regions are.

        The selected ones if any are selected, so several boxes can differ;
        otherwise every region of the role whose page the slider is on, which
        is what a slider labelled "Region depth" sitting under one role reads
        as. Either way it also sets the depth the next region will be drawn at.
        """
        target = regions_layer()
        if target is None or state.applying or not len(target.data):
            return
        set_defaults(target)
        handles = set(handle_indices(target))
        chosen = set(target.selected_data) & handles or handles
        if not chosen:
            return
        features = dict(target.features)
        if "depth" not in features:
            return
        column = list(features["depth"])
        for index in chosen:
            if index < len(column):
                column[index] = float(depth_slider().value)
        state.applying = True
        try:
            features["depth"] = np.asarray(column, dtype=float)
            target.features = features
        finally:
            state.applying = False
        sync()
        # The outline is drawn from the settings, so it only follows the
        # slider once the settings have been told.
        redraw()

    def on_move() -> None:
        """Hand over to napari's select tool, which is what moves a pick.

        Placing and moving are different modes of the same layer -- clicking in
        `add` mode makes another point rather than picking up the one under the
        cursor -- and nothing on screen says so.
        """
        method = str(current_values().get(method_setting(str(role.value))))
        regions = method == "volume"
        name = regions_name(str(role.value)) if regions else BC_COORDINATES
        target = layer(name)
        if target is None:
            report.value = "Nothing to move yet -- press Show, then pick or draw."
            return
        if regions and viewer.dims.ndisplay == 3:
            report.value = (
                "Regions can only be edited in the 2D view -- napari does not "
                "allow editing a Shapes layer in 3D. Coordinates can be moved "
                "in either view."
            )
            return
        viewer.layers.selection.active = target
        target.mode = "select"
        report.value = (
            f"Select mode on {name}. Click one to select it, drag to move it, "
            "press Delete to remove it, and drag a box to take several at once. "
            "Every move is written straight back to the settings."
        )

    def on_assign() -> None:
        """Give the selected items the chosen role."""
        changed = 0
        # Reassigning a region moves it to another layer: the role written
        # here reaches the settings through `sync`, and the redraw after it
        # rebuilds each layer from the role that now owns the box.
        for name in our_layer_names():
            target = layer(name)
            if target is None or not len(target.data):
                continue
            chosen = target.selected_data
            if not chosen:
                continue
            features = dict(target.features)
            column = [str(v) for v in features.get("role", [])]
            column += [str(role.value)] * (len(target.data) - len(column))
            for index in chosen:
                if index < len(column):
                    column[index] = str(role.value)
                    changed += 1
            state.applying = True
            try:
                features["role"] = np.asarray(column, dtype=object)
                target.features = features
            finally:
                state.applying = False
        if changed:
            sync()
            redraw()
        else:
            report.value = "Select the points or regions to reassign first."

    def on_snap() -> None:
        target = layer(BC_COORDINATES)
        candidates, ids = terminal_points(graph())
        if target is None or not len(target.data):
            report.value = "No picked coordinates to snap."
            return
        if not len(candidates):
            report.value = (
                "Nothing to snap to yet: snapping uses the graph's terminal "
                "nodes, so run at least '3. Graph' first. The coordinates you "
                "have are still correct -- a run snaps each one to its nearest "
                "terminal anyway."
            )
            return
        chosen = sorted(target.selected_data) or list(range(len(target.data)))
        data = np.asarray(target.data, dtype=float).copy()
        snapped, moved = snap(data[chosen], candidates)
        data[chosen] = snapped
        state.applying = True
        try:
            target.data = data
        finally:
            state.applying = False
        sync()
        report.value = (
            f"Snapped {len(chosen)} coordinate(s) onto terminal nodes; "
            f"the furthest moved {float(np.max(moved)):.1f} um. "
            "A large move means the click missed the vessel."
        )

    def on_clear() -> None:
        write_rows({volume_setting(str(role.value)): []})
        redraw()

    for _name in (
        *(name for _role in ROLES
          for name in (method_setting(_role), coordinate_setting(_role),
                       volume_setting(_role))),
        # The band settings draw a box too, so they move the picture as much
        # as a coordinate does.
        *shared_settings(),
        "automated_vessel_assignment",
        "use_small_vessel_masks_for_boundary_assignment",
    ):
        if _name in rows:
            rows[_name].changed.connect(on_settings_changed)
    refresh_rows()

    show.changed.connect(lambda *_: on_show())
    snap_button.changed.connect(lambda *_: on_snap())

    def wire(owner: str, control, handler) -> None:
        """A control on a role's page acts on that role, whatever is selected.

        The page it sits on is the answer to "which role", so pressing it says
        so rather than assuming the tab bar and the control agree.
        """
        def run(*_args) -> None:
            # The panel writes to these widgets itself -- defaulting every
            # role's depth slider, for one. That is not the user reaching for
            # a control, so it must not move the role onto that control's page.
            if state.applying:
                return
            if str(role.value) != owner:
                role.value = owner
            handler()

        control.changed.connect(run)

    for _name, _action in actions.items():
        wire(_name, _action.pick, on_pick)
        wire(_name, _action.draw, on_draw)
        wire(_name, _action.move, on_move)
        wire(_name, _action.assign, on_assign)
        wire(_name, _action.clear, on_clear)
        wire(_name, _action.depth, on_depth_changed)

    def on_ndisplay(*_args) -> None:
        # refresh_actions owns enabled for Draw (2D vs 3D and the automated
        # inlet/outlet override).
        refresh_actions()

    viewer.dims.events.ndisplay.connect(on_ndisplay)
    on_ndisplay()

    widget = Container(widgets=[show, snap_button], labels=True)

    def page(summary, names: Sequence[str]):
        """The Boundaries tab: one sub-tab per role, then what they share.

        Four roles times a method, a coordinate list, a region list and a node
        list is twenty rows on one page, and all four look alike -- picking a
        role at the top and reading its settings underneath is the only way the
        tab says which of the four you are configuring. The sub-tab is also the
        role a picked point takes, so there is one answer to "which role am I
        working on" rather than a tab and a dropdown that can disagree.
        """
        from qtpy.QtWidgets import QTabWidget, QVBoxLayout, QWidget

        role_tabs = QTabWidget()
        placed: set[str] = set()
        for name in ROLES:
            mine = [n for n in role_settings(name) if n in rows]
            placed.update(mine)
            action = actions[name]
            holder = Container(
                widgets=[
                    *(rows[n] for n in mine),
                    action.pick, action.draw, action.depth,
                    action.move, action.assign, action.clear,
                ],
                labels=True,
            )
            holders[name] = holder
            role_tabs.addTab(holder.native, role_title(name))
            role_tabs.setTabToolTip(role_tabs.count() - 1,
                                    fields[method_setting(name)].help)

        shared = [n for n in shared_settings() if n in rows]
        rest = [n for n in names if n not in placed and n not in shared]
        # Shared main/large/small ilastik knobs are reparented here when only
        # vessel-mask ilastik is on (declared under Input; same widgets).
        shared_ilastik_holder = Container(widgets=[], labels=True)

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(summary.native)
        if rest:
            layout.addWidget(Container(widgets=[rows[n] for n in rest],
                                       labels=True).native)
        layout.addWidget(shared_ilastik_holder.native)
        layout.addWidget(Label(value=AUTOMATED_OVERRIDES_MANUAL_NOTE).native)
        layout.addWidget(widget.native)
        layout.addWidget(role_tabs)
        layout.addStretch(1)
        for name in shared:
            shared_home[name] = None
        place_shared()

        def on_tab_changed(index: int) -> None:
            if 0 <= index < len(ROLES):
                role.value = ROLES[index]

        role_tabs.currentChanged.connect(on_tab_changed)
        state.tabs = role_tabs
        refresh_role_tabs()
        state.shared_ilastik_holder = shared_ilastik_holder
        return body

    def on_role_changed(*_args) -> None:
        """Follow the role: the tab bar, and what the next pick will be."""
        tabs = getattr(state, "tabs", None)
        if tabs is not None:
            index = list(ROLES).index(str(role.value))
            if tabs.currentIndex() != index:
                tabs.setCurrentIndex(index)
        place_shared()
        for name in our_layer_names():
            target = layer(name)
            if target is not None:
                set_defaults(target)

    role.changed.connect(on_role_changed)

    return SimpleNamespace(
        widget=widget, role=role, actions=actions, depth_slider=depth_slider,
        holders=holders, shared_home=shared_home,
        state=state, page=page,
        row_order=row_order, refresh_rows=refresh_rows,
        show=on_show, pick=on_pick, draw=on_draw, snap=on_snap, move=on_move,
        assign=on_assign, clear=on_clear, redraw=redraw, sync=sync,
        layer_names=(BC_COORDINATES, *BC_REGION_NAMES),
        shared_ilastik_holder=lambda: getattr(state, "shared_ilastik_holder", None),
    )


def _perturbation_controls(viewer, rows, fields, schema, report):
    """The Perturbations tab's list of "and what if", built a row at a time.

    One dropdown per perturbation, and choosing a type *reveals* that type's
    options rather than greying out the other three types' -- the same reason
    the Boundaries tab hides what a selection method does not read. A "+"
    below the last one adds another, so a run can ask five questions of one
    network.

    What each editor writes is the important part. `arteriole_diameter_change_percent`
    reaches the run only through a perturbation entry's overrides: an editor
    that wrote into `rows` would move the baseline every perturbation is
    measured against. These write into ``perturbations[i]["overrides"]``, which
    reaches the run through the one `perturbations` row and nowhere else.

    :mod:`haemolynx.gui.perturbation_editing` is the whole decision -- which
    editors a type reveals, what adding and removing do to the list -- and it
    needs no GUI. This is the Qt layer over it.

    Returns None without a viewer: the panel is buildable outside napari.
    """
    if viewer is None:
        return None

    from magicgui.widgets import ComboBox, Container, Label, LineEdit, PushButton

    from haemolynx.gui.perturbation_editing import (
        ADD_TOOLTIP,
        EDITOR_SETTINGS,
        NAME_TOOLTIP,
        PERTURBATION_TYPES,
        REMOVE_TOOLTIP,
        TYPE_TOOLTIP,
        add_entry,
        display_label_for_setting,
        editor_layout_order,
        from_settings,
        name_problems,
        perturbation_type_choices,
        remove_entry,
        rows_for_type,
        set_name,
        set_overrides,
        set_type,
        summary,
        to_settings,
        visible_tab_settings,
    )

    #: The one row the list actually travels in. Everything below edits this.
    LIST_SETTING = "perturbations"

    state = SimpleNamespace(applying=False, entries=[], editors=[])

    #: Where the per-entry editors are stacked, and the button that adds one.
    holder = Container(widgets=[], labels=False)
    add_button = PushButton(text="+  Add a perturbation")
    add_button.tooltip = ADD_TOOLTIP

    def read_row() -> list[dict[str, Any]]:
        widget = rows.get(LIST_SETTING)
        if widget is None:
            return []
        return from_settings(
            {LIST_SETTING: fields[LIST_SETTING].to_setting_value(widget.value)}
        )

    def write_row() -> None:
        """The entries -> the settings row. The only way they leave here."""
        widget = rows.get(LIST_SETTING)
        if widget is None:
            return
        state.applying = True
        try:
            widget.value = display_value_for(
                schema[LIST_SETTING], to_settings(state.entries)[LIST_SETTING]
            )
        finally:
            state.applying = False
        say()

    def say() -> None:
        problems = name_problems(state.entries)
        report.value = "Perturbations: " + summary(state.entries) + (
            "  Fix: " + "; ".join(problems) + "." if problems else ""
        )

    def overrides_from(editor) -> dict[str, Any]:
        """What this entry's *visible* editors say, and nothing else.

        The visible editors are the entry's overrides: an option belonging to
        a type the user has moved away from has stopped being set, rather than
        lingering as a value nothing applies.

        An empty one is left out rather than sent as None. An override replaces
        what the run configured, so a blank `pericyte_mask_path` editor would
        take away the mask the Diameters tab named -- and "unset" is not
        something a perturbation can meaningfully say.
        """
        shown = rows_for_type(editor.type.value)
        read = {
            name: fields[name].to_setting_value(editor.editors[name].value)
            for name in shown
            if name in editor.editors
        }
        return {name: value for name, value in read.items() if value is not None}

    def lay_out_editors(editor) -> None:
        """Name, type, this type's knobs in SETTINGS_FOR_TYPE order, Remove.

        Visibility alone is not enough: editors are built once for every type,
        so a naive append order buried ``arteriole_diameter_change_percent``
        under pericyte geometry rows on the combined type.
        """
        chosen = editor.type.value
        shown = set(rows_for_type(chosen))
        for name, widget in editor.editors.items():
            widget.visible = name in shown
        ordered = [
            editor.editors[name]
            for name in editor_layout_order(chosen)
            if name in editor.editors
        ]
        editor.container.clear()
        editor.container.append(editor.name)
        editor.container.append(editor.type)
        for widget in ordered:
            editor.container.append(widget)
        editor.container.append(editor.remove)
        editor.shown = frozenset(name for name in editor.editors if name in shown)
        editor.hidden = frozenset(
            name for name in editor.editors if name not in shown
        )
        editor.layout_order = tuple(
            name for name in editor_layout_order(chosen) if name in editor.editors
        )

    def show_the_chosen_type(editor) -> None:
        """Reveal the chosen type's options and hide every other type's.

        What was shown is recorded as well as applied: `.visible` on a row of
        a tab that is not on screen reads False whatever was set, so a test
        cannot ask the widget -- the same reason `_boundary_controls` keeps
        `state.visible`.
        """
        lay_out_editors(editor)

    def build_editor(index: int, entry: Mapping[str, Any]):
        """One perturbation's controls: name, type, and that type's options."""
        name_box = LineEdit(value=str(entry.get("name") or ""), label="Name")
        name_box.tooltip = NAME_TOOLTIP
        chosen = str(entry.get("type") or PERTURBATION_TYPES[0])
        type_box = ComboBox(
            choices=perturbation_type_choices(),
            value=chosen if chosen in PERTURBATION_TYPES else PERTURBATION_TYPES[0],
            label="Type",
        )
        type_box.tooltip = TYPE_TOOLTIP

        # Built once for every type, and hidden but for the chosen type's:
        # destroying and rebuilding widgets as the dropdown changes is how a
        # container ends up holding a row nothing can reach.
        configured = entry.get("overrides") or {}
        editors: dict[str, Any] = {}
        for name in EDITOR_SETTINGS:
            field = fields.get(name)
            if field is None:
                continue
            widget = _build_row(field)
            # Display labels (constriction/dilation wording) without renaming
            # schema keys: the cloned Field still carries the auto snake_case
            # label from form.label_for.
            setting = schema[name] if name in schema else None
            widget.label = display_label_for_setting(
                name, None if setting is None else setting.unit
            )
            if name in configured:
                widget.value = display_value_for(schema[name], configured[name])
            editors[name] = widget

        remove_button = PushButton(text="Remove")
        remove_button.tooltip = REMOVE_TOOLTIP
        editor = SimpleNamespace(
            index=index,
            name=name_box,
            type=type_box,
            editors=editors,
            shown=frozenset(),
            hidden=frozenset(),
            layout_order=(),
            remove=remove_button,
            # Empty shell; lay_out_editors fills it in SETTINGS_FOR_TYPE order.
            container=Container(widgets=[], labels=True),
        )

        def on_name(*_args) -> None:
            if state.applying:
                return
            state.entries = set_name(state.entries, index, str(name_box.value))
            write_row()

        def on_type(*_args) -> None:
            if state.applying:
                return
            state.entries = set_type(state.entries, index, str(type_box.value))
            show_the_chosen_type(editor)
            # After the reveal: the newly shown editors are now this entry's
            # overrides, and the ones just hidden are not.
            state.entries = set_overrides(state.entries, index, overrides_from(editor))
            write_row()

        def on_override(*_args) -> None:
            if state.applying:
                return
            state.entries = set_overrides(state.entries, index, overrides_from(editor))
            write_row()

        def on_remove(*_args) -> None:
            if state.applying:
                return
            remove_at(index)

        name_box.changed.connect(on_name)
        type_box.changed.connect(on_type)
        for widget in editors.values():
            widget.changed.connect(on_override)
        remove_button.changed.connect(on_remove)
        lay_out_editors(editor)
        return editor

    def rebuild() -> None:
        """Lay the editors out again, in the order the entries are in.

        All of them, because removing one moves every later entry's index and
        an editor knows which entry it edits.
        """
        state.applying = True
        try:
            holder.clear()
            state.editors = []
            for index, entry in enumerate(state.entries):
                editor = build_editor(index, entry)
                state.editors.append(editor)
                holder.append(editor.container)
        finally:
            state.applying = False

    def on_add(*_args) -> None:
        if state.applying:
            return
        state.entries = add_entry(state.entries)
        rebuild()
        write_row()

    def remove_at(index: int) -> None:
        """What pressing entry *index*'s "Remove" button does."""
        state.entries = remove_entry(state.entries, index)
        rebuild()
        write_row()

    def choose_type(index: int, perturbation_type: str) -> None:
        """What choosing a type in entry *index*'s dropdown does."""
        if 0 <= index < len(state.editors):
            state.editors[index].type.value = str(perturbation_type)

    def on_list_changed(*_args) -> None:
        """The row was edited by hand, or a config was loaded into it."""
        if state.applying:
            return
        state.entries = read_row()
        rebuild()
        say()

    add_button.changed.connect(on_add)
    if LIST_SETTING in rows:
        rows[LIST_SETTING].changed.connect(on_list_changed)
    state.entries = read_row()
    rebuild()

    def page(stage_summary, names: Sequence[str]):
        """The tab: the run's own settings, then the list of perturbations."""
        from qtpy.QtWidgets import QVBoxLayout, QWidget

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(stage_summary.native)
        flat = [name for name in visible_tab_settings(names) if name in rows]
        if flat:
            layout.addWidget(
                Container(widgets=[rows[name] for name in flat], labels=True).native
            )
        layout.addWidget(
            Label(
                value=(
                    "Each perturbation re-solves the network from the same "
                    "baseline. The values above are what one starts from; the "
                    "options below are what it changes."
                )
            ).native
        )
        layout.addWidget(holder.native)
        layout.addWidget(add_button.native)
        layout.addStretch(1)
        return body

    return SimpleNamespace(
        page=page,
        state=state,
        entries=lambda: list(state.entries),
        editors=lambda: list(state.editors),
        add=on_add,
        remove=remove_at,
        choose_type=choose_type,
        rebuild=rebuild,
        holder=holder,
        add_button=add_button,
        list_setting=LIST_SETTING,
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
    from magicgui.widgets import CheckBox, ComboBox, Container, Label, PushButton, TextEdit
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (
        QHBoxLayout,
        QScrollArea,
        QStackedWidget,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )

    viewer = napari_viewer if napari_viewer is not None else napari.current_viewer()

    schema = default_schema()
    tabs = tabs_for(schema)

    rows: dict[str, Any] = {}
    fields: dict[str, Field] = {}
    tab_widget = QTabWidget()

    report = TextEdit(value="Ready.")
    report.read_only = True

    # Two passes. Every row has to exist before anything that reads them can be
    # built, and the boundary controls read them -- so rows first, pages second.
    for tab in tabs:
        for field in tab.fields:
            rows[field.name] = _build_row(field)
            fields[field.name] = field

    # Shared ilastik knobs stay out of the initial tab containers until
    # place_shared_ilastik hosts them. Keep them hidden while unparented: a
    # visible widget with no parent is a floating top-level window.
    for name in SHARED_ILASTIK_SETTINGS:
        if name in rows:
            rows[name].visible = False

    # Perturbations claims legacy flags and typed-entry options so Field
    # objects exist, but only ALWAYS_VISIBLE_TAB_SETTINGS are parented as flat
    # rows. Editors clone their own widgets. Hide the rest and tuck them under
    # a hidden holder: Diameters-section hide_when_unmet would otherwise set
    # them visible under run_haemodynamics=True, and a visible widget with no
    # parent is a floating top-level window (``147a545`` + ``30c605e``).
    from haemolynx.gui.perturbation_editing import orphaned_tab_settings

    orphaned_perturbation_rows: frozenset[str] = frozenset()
    for tab in tabs:
        if tab.stage.call == "run_perturbations":
            orphaned_perturbation_rows = frozenset(
                orphaned_tab_settings(field.name for field in tab.fields)
            )
            break
    orphan_holder = Container(widgets=[], labels=False)
    orphan_holder.visible = False
    for name in orphaned_perturbation_rows:
        if name in rows:
            rows[name].visible = False
            orphan_holder.append(rows[name])

    #: Stages that lay their own page out, keyed by the stage function they
    #: belong to rather than by the tab's title, so renaming a tab cannot
    #: silently drop them. Any future stage-specific page has a home here.
    pages: dict[str, Any] = {}
    boundaries = _boundary_controls(viewer, rows, fields, schema, report)
    if boundaries is not None:
        pages["assign_boundaries"] = boundaries.page
    perturbations = _perturbation_controls(viewer, rows, fields, schema, report)
    if perturbations is not None:
        pages["run_perturbations"] = perturbations.page

    #: Snapshots from the last run that showed layers: what "Revert to previous
    #: stage" on a tab reloads. Cleared when the layers are, and replaced when
    #: a new run that shows layers starts.
    checkpoints = StageCheckpoints()
    revert_buttons: dict[str, Any] = {}
    from haemolynx.gui.chrome_tooltips import REVERT_STAGE_TOOLTIP

    #: Input-tab container that can receive the shared ilastik rows when main
    #: segmentation uses ilastik. Boundaries gets a holder of its own.
    input_settings: Any = None
    #: Which tab currently parents the shared ilastik rows.
    shared_ilastik_placement: dict[str, str | None] = {"host": None}

    for tab in tabs:
        summary = Label(value=tab.stage.summary)
        build = pages.get(tab.stage.call or "")
        # Shared ilastik knobs start unparented; place_shared_ilastik hosts them.
        names = [
            field.name
            for field in tab.fields
            if field.name not in SHARED_ILASTIK_SETTING_SET
        ]
        if build is not None:
            native = build(summary, names)
        elif tab.stage.call == "segment":
            input_settings = Container(
                widgets=[summary, *(rows[name] for name in names)],
                labels=True,
            )
            native = input_settings.native
        else:
            native = Container(
                widgets=[summary, *(rows[name] for name in names)],
                labels=True,
            ).native
        # A plain QScrollArea rather than `Container(scrollable=True)`: the
        # magicgui one reports the full height of its contents, so a tab with
        # 39 rows stretches the whole napari window instead of scrolling.
        # QScrollArea's own size hint ignores the widget inside it, which is
        # exactly what keeps the panel a sensible size. Its scrollbars default
        # to appearing only when needed, vertically and horizontally.
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(native)
        # Revert lives in shared chrome below "Show each topology step", not
        # inside the tab page: one button per tab that has a predecessor, shown
        # for the active tab and centered on the panel. After a full run it
        # reloads that previous tab's end-of-stage checkpoint so later settings
        # can be tweaked without rebuilding the network.
        if previous_tab(tab.stage.title) is not None:
            revert = PushButton(text="Revert to previous stage")
            revert.enabled = False
            revert.tooltip = REVERT_STAGE_TOOLTIP
            revert.native.setObjectName("haemolynx_revert")
            revert_buttons[tab.stage.title] = revert
        page_layout.addStretch(1)
        scroller = QScrollArea()
        scroller.setWidgetResizable(True)
        scroller.setWidget(page)
        tab_widget.addTab(scroller, tab.stage.title)
        if tab.stage.call:
            index = tab_widget.count() - 1
            tab_widget.setTabToolTip(index, f"{tab.stage.call}(settings, ...)")

    # Per-tab Revert pages, stacked in tab order: empty for the first tab,
    # centered button for every later one. The stack tracks the tab widget so
    # the chrome below show-steps always shows the active tab's Revert.
    revert_stack = QStackedWidget()
    revert_stack.setObjectName("haemolynx_revert_stack")
    for tab in tabs:
        slot = QWidget()
        slot.setObjectName("haemolynx_revert_slot")
        row = QHBoxLayout(slot)
        row.setContentsMargins(0, 0, 0, 0)
        row.addStretch(1)
        button = revert_buttons.get(tab.stage.title)
        if button is not None:
            row.addWidget(button.native, 0, Qt.AlignHCenter)
        row.addStretch(1)
        revert_stack.addWidget(slot)

    def sync_revert_stack(index: int) -> None:
        if 0 <= index < revert_stack.count():
            revert_stack.setCurrentIndex(index)

    tab_widget.currentChanged.connect(sync_revert_stack)
    sync_revert_stack(tab_widget.currentIndex())
    #: What a loaded config said each path setting was, before its FileEdit
    #: made it absolute. Empty until a config is opened.
    loaded_paths: dict[str, Any] = {}

    #: What pointing the run at an open layer put into the form, and which
    #: layer it was. `settings` is None until a layer has been adopted.
    adopted = SimpleNamespace(settings=None, name=None)

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
        # Kept so that opening a config afterwards does not silently point the
        # run back at whatever image that file was written for.
        adopted.settings = dict(chosen.settings)
        adopted.name = getattr(layer, "name", None)
        apply_prerequisites()
        note = chosen.note
        applied = _scale_layer_from_its_file(layer, chosen.settings.get("input_path"))
        if applied is not None:
            note += (
                f" Scaled the layer to {tuple(round(v, 4) for v in applied)} (z, y, x) "
                "microns, from the file, so it sits where the results will."
            )
        report.value = note

    def place_shared_ilastik() -> None:
        """Host shared ilastik rows on Input or Boundaries (same widgets).

        These rows are deliberately left out of the initial tab containers and
        moved later. A magicgui row with no Qt parent that is set visible
        becomes a top-level window beside napari — the same failure
        ``place_shared`` already guards against for boundary-method rows.
        Hide before detach, show only after a successful append, and never
        record a host that did not actually receive the widgets.
        """
        values = current_values()
        host = shared_ilastik_host(values)
        boundaries_holder = None
        if boundaries is not None:
            getter = getattr(boundaries, "shared_ilastik_holder", None)
            boundaries_holder = getter() if callable(getter) else getter

        if host == shared_ilastik_placement["host"]:
            # Already placed (or correctly unhosted). Do not poke ``visible``
            # here: setting True on an unparented row opens a floating window.
            return

        def _detach(container) -> None:
            if container is None:
                return
            for name in SHARED_ILASTIK_SETTINGS:
                row = rows.get(name)
                if row is None:
                    continue
                # Hide before remove: a visible widget with no parent is a
                # window of its own (see Boundaries ``place_shared``).
                row.visible = False
                try:
                    container.remove(row)
                except Exception:
                    pass

        _detach(input_settings)
        _detach(boundaries_holder)

        attached: str | None = None
        if host == "input" and input_settings is not None:
            for name in SHARED_ILASTIK_SETTINGS:
                if name not in rows:
                    continue
                input_settings.append(rows[name])
                rows[name].visible = True
            attached = "input"
        elif host == "boundaries" and boundaries_holder is not None:
            for name in SHARED_ILASTIK_SETTINGS:
                if name not in rows:
                    continue
                boundaries_holder.append(rows[name])
                rows[name].visible = True
            attached = "boundaries"
        else:
            for name in SHARED_ILASTIK_SETTINGS:
                if name in rows:
                    rows[name].visible = False

        shared_ilastik_placement["host"] = attached

    def apply_prerequisites(*_args) -> None:
        """Apply schema prerequisites: hide nested rows, grey others."""
        values = current_values()
        place_shared_ilastik()
        for name, widget in rows.items():
            if name in SHARED_ILASTIK_SETTING_SET:
                # Visibility and parent belong to place_shared_ilastik.
                widget.enabled = True
                widget.tooltip = fields[name].help
                continue
            if name in orphaned_perturbation_rows:
                # Never parented as flat tab rows; typed editors clone their
                # own widgets. Do not reveal — visible + no parent is a window.
                widget.visible = False
                widget.enabled = True
                widget.tooltip = fields[name].help
                continue
            field = fields[name]
            enabled = field.is_enabled(values)
            if field.hide_when_unmet:
                # Input / Diameters / FWHM / Boundaries vessel / Graph
                # centreline options: only relevant nested knobs appear.
                widget.visible = field.is_visible(values)
                widget.enabled = True
                widget.tooltip = field.help
            else:
                widget.enabled = enabled
                widget.tooltip = field.help if enabled else field.why_disabled(values)

    for name, value in DISPLAY_SETTINGS_OFF_IN_NAPARI.items():
        if name in rows:
            rows[name].value = value

    for widget in rows.values():
        widget.changed.connect(apply_prerequisites)

    #: User-facing values of ``do_skeletonize`` / ``do_graph_building`` before
    #: Revert turns them off for resume. Restored by "Clear layers and state".
    skip_toggle_snapshot: dict[str, bool] = {}
    revert_setting_skips = False

    def snapshot_skip_toggles(*_args) -> None:
        if revert_setting_skips:
            return
        for name in SKIP_FOR_RESUME:
            if name in rows:
                skip_toggle_snapshot[name] = bool(rows[name].value)

    apply_prerequisites()
    snapshot_skip_toggles()
    for name in SKIP_FOR_RESUME:
        if name in rows:
            rows[name].changed.connect(snapshot_skip_toggles)

    bars = ProgressBars()
    run_state = RunState(bars=bars)

    # The run's own narration, in a dock of its own rather than in the panel:
    # its lines are wide and there are thousands of them, and the panel is a
    # narrow column of settings. One instance, owned here, because it is fed by
    # the run this panel starts -- which is why it is not a widget contribution
    # in `napari.yaml`, where a second, menu-launched one would sit empty.
    log_view = LogView()
    log_dock = None
    if viewer is not None:
        log_dock = viewer.window.add_dock_widget(
            log_view.native, name=LOG_DOCK_NAME, area="bottom"
        )

    def show_log() -> None:
        """Put the log window back in front, in case the user closed it."""
        if log_dock is None:
            return
        try:
            log_dock.setVisible(True)
            log_dock.raise_()
        except Exception:  # noqa: BLE001 - a dock already gone is survivable
            logger.debug("could not show the log window", exc_info=True)

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
        from haemolynx.gui.chrome_tooltips import USE_LAYER_TOOLTIP

        use_button.tooltip = USE_LAYER_TOOLTIP
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
    clear_button = PushButton(text="Clear layers and state")

    from haemolynx.gui.chrome_tooltips import (
        CLEAR_LAYERS_TOOLTIP,
        LOAD_CONFIG_TOOLTIP,
        RUN_CHECKS_TOOLTIP,
        RUN_PIPELINE_TOOLTIP,
        SAVE_CONFIG_TOOLTIP,
        SHOW_RESULTS_TOOLTIP,
        SHOW_STEPS_TOOLTIP,
    )

    load_button.tooltip = LOAD_CONFIG_TOOLTIP
    save_button.tooltip = SAVE_CONFIG_TOOLTIP
    check_button.tooltip = RUN_CHECKS_TOOLTIP
    run_button.tooltip = RUN_PIPELINE_TOOLTIP
    clear_button.tooltip = CLEAR_LAYERS_TOOLTIP

    # What a run puts in the viewer, and how it is coloured. These are panel
    # controls rather than settings: a config file is read by CLI runs too,
    # where "show it in napari" means nothing.
    show_results = CheckBox(value=True, text="Show each stage in the viewer")
    show_steps = CheckBox(value=False, text="Show each topology step")
    show_results.tooltip = SHOW_RESULTS_TOOLTIP
    show_steps.tooltip = SHOW_STEPS_TOOLTIP
    show_results.native.setObjectName("haemolynx_show_results")
    show_steps.native.setObjectName("haemolynx_show_steps")
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QFormLayout, QLabel, QWidget
    from superqt import QDoubleRangeSlider, QDoubleSlider

    z_depth_label = QLabel("Z depth filter (µm)")
    z_depth_slider = QDoubleRangeSlider(Qt.Orientation.Horizontal)
    z_depth_slider.setObjectName("haemolynx_z_depth_slider")
    z_depth_slider.setEnabled(False)
    z_depth_slider.setVisible(False)
    z_depth_host = QWidget()
    z_depth_host.setObjectName("haemolynx_z_depth_host")
    z_depth_form = QFormLayout(z_depth_host)
    z_depth_form.addRow(z_depth_label, z_depth_slider)
    z_depth_parking = QWidget()
    z_depth_parking.setVisible(False)
    z_depth_host.setParent(z_depth_parking)
    z_depth_mount: dict[str, Any] = {"controls": None}

    arrow_length_label = QLabel("Arrow size")
    arrow_length_slider = QDoubleSlider(Qt.Orientation.Horizontal)
    arrow_length_slider.setObjectName("haemolynx_arrow_length_slider")
    arrow_length_slider.setRange(_ARROW_LENGTH_MIN, _ARROW_LENGTH_MAX)
    arrow_length_slider.setSingleStep(0.1)
    arrow_length_slider.setEnabled(False)
    arrow_length_slider.setVisible(False)
    arrow_length_host = QWidget()
    arrow_length_host.setObjectName("haemolynx_arrow_length_host")
    arrow_length_form = QFormLayout(arrow_length_host)
    arrow_length_form.addRow(arrow_length_label, arrow_length_slider)
    arrow_length_parking = QWidget()
    arrow_length_parking.setVisible(False)
    arrow_length_host.setParent(arrow_length_parking)
    arrow_length_mount: dict[str, Any] = {"controls": None}

    view = SimpleNamespace(results=None)

    def reparent_z_depth_slider() -> None:
        if viewer is None:
            return
        _reparent_z_depth_slider(
            viewer,
            slider=z_depth_slider,
            host=z_depth_host,
            parking=z_depth_parking,
            controls_holder=z_depth_mount,
            results=view.results,
        )

    def reparent_arrow_length_slider() -> None:
        if viewer is None:
            return
        _reparent_arrow_length_slider(
            viewer,
            slider=arrow_length_slider,
            host=arrow_length_host,
            parking=arrow_length_parking,
            controls_holder=arrow_length_mount,
            z_depth_host=z_depth_host,
        )

    def reparent_layer_control_sliders() -> None:
        reparent_z_depth_slider()
        reparent_arrow_length_slider()

    if viewer is not None:
        viewer._haemolynx_reparent_z_depth_slider = reparent_layer_control_sliders
        viewer._haemolynx_reparent_arrow_length_slider = reparent_arrow_length_slider
        viewer.layers.selection.events.active.connect(
            lambda *_args: reparent_layer_control_sliders()
        )

    def _after_layers_applied() -> None:
        """Refresh Z depth slider range and apply the current filter."""
        if viewer is None:
            return
        reparent_layer_control_sliders()
        if view.results is None or not z_depth_slider.isEnabled():
            return
        z_lo, z_hi = z_depth_slider.value()
        _apply_z_filter(
            viewer,
            z_lo,
            z_hi,
            z_extent=view.results.image_z_extent_um(),
        )

    if viewer is not None:
        viewer._haemolynx_after_layers_applied = _after_layers_applied

    def on_z_depth_changed(*_args) -> None:
        if viewer is None or view.results is None:
            return
        z_lo, z_hi = z_depth_slider.value()
        _apply_z_filter(
            viewer,
            z_lo,
            z_hi,
            z_extent=view.results.image_z_extent_um(),
        )

    def on_arrow_length_changed(value: float) -> None:
        if viewer is None:
            return
        active = viewer.layers.selection.active
        if not _vectors_layer_uses_arrow_length(active):
            return
        length = max(_ARROW_LENGTH_MIN, min(_ARROW_LENGTH_MAX, float(value)))
        active.length = length
        _remember_arrow_length(active, length)

    z_depth_slider.valueChanged.connect(on_z_depth_changed)
    arrow_length_slider.valueChanged.connect(on_arrow_length_changed)

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
        try:
            loaded = load_config(Path(path), schema)
        except Exception as error:
            # Surface the failure in the panel rather than through Qt's
            # uncaught-signal traceback -- a bad file (wrong schema, corrupt
            # YAML from an older save) is a user-facing message, not a crash.
            report.value = f"Could not load {path}:\n{error}"
            return
        loaded_paths.clear()
        for name, value in loaded.items():
            if name in rows:
                if schema[name].kind == "path" and value is not None:
                    loaded_paths[name] = value
                rows[name].value = display_value_for(schema[name], value)

        # A config names the image it was written for, which is rarely the one
        # on screen -- the shipped one names an image that is not even in the
        # repository. With a layer open, that layer is the input and the file's
        # `input_path` is not read; everything else in the file still applies.
        kept = ""
        if adopted.settings:
            for name, value in adopted.settings.items():
                if name in rows:
                    rows[name].value = value
                    loaded_paths.pop(name, None)
            kept = f", keeping {adopted.name or 'the open layer'} as the input"

        apply_prerequisites()
        snapshot_skip_toggles()
        report.value = f"Loaded {path}{kept}"
        if boundaries is not None and any(
            name in viewer.layers for name in boundaries.layer_names
        ):
            # Only if they are already on screen: opening a config should not
            # add layers nobody asked for.
            boundaries.redraw()

    def on_load() -> None:
        from qtpy.QtWidgets import QFileDialog

        path, _filter = QFileDialog.getOpenFileName(
            None, "Open a HaemoLynx config", "", "YAML (*.yaml *.yml)"
        )
        if not path:
            return
        load_config_file(path)

    def save_config_file(path: Path | str) -> bool:
        """Write the panel's current settings to *path*.

        Returns True when the file was written. Failures (schema validation,
        I/O, still-broken YAML edge cases) are reported in the panel rather
        than raised through the Qt/psygnal signal — same contract as load.
        """
        try:
            dump_config(Path(path), schema, values=current_values())
        except Exception as error:
            report.value = f"Could not save {path}:\n{error}"
            return False
        report.value = f"Wrote {path}"
        return True

    def on_save() -> None:
        from qtpy.QtWidgets import QFileDialog

        path, _filter = QFileDialog.getSaveFileName(
            None, "Save these settings", "config.yaml", "YAML (*.yaml *.yml)"
        )
        if not path:
            return
        save_config_file(path)

    def on_check() -> None:
        result = preflight(_settings(), schema)
        lines = [f"FAILED: {message}" for message in result.errors]
        lines += [f"warning: {message}" for message in result.warnings]
        report.value = "\n".join(lines) if lines else "All checks passed."

    def refresh_revert_buttons() -> None:
        """Enable each tab's revert button only when its previous stage is saved."""
        for title, button in revert_buttons.items():
            button.enabled = (
                not run_state.running and can_revert_from(title, checkpoints)
            )

    def on_run() -> None:
        if run_state.running:
            # The button is disabled while a run is going, so this is only
            # reached from a script or a keyboard -- but it is also the one
            # place that says how to get out of a run, so it says it.
            report.value = ALREADY_RUNNING
            return
        settings = _settings()
        if not preflight(settings, schema).ok:
            report.value = "Checks failed; nothing was run. Press 'Run checks' for detail."
            return
        results = None
        if show_results.value and viewer is not None:
            results = ResultLayers(
                show_steps=bool(show_steps.value),
                settings=settings,
            )
            view.results = results
            if boundaries is not None:
                boundaries.state.results = results
            # A new run that will show layers replaces the previous run's
            # checkpoints; a run with layers off leaves them alone so a revert
            # from an earlier shown run is still possible.
            checkpoints.clear()
        # How much of the run to show follows the setting that already means
        # "tell me everything"; the window itself is a panel control, like
        # `show_results`, because a config file is read by CLI runs too.
        log_view.set_level(
            VERBOSE_LEVEL if settings.get("verbose_logging") else DEFAULT_LEVEL
        )
        show_log()
        refresh_revert_buttons()
        worker = _run_in_background(
            settings, schema, report, run_button, bars,
            viewer=viewer if show_results.value else None,
            results=results,
            state=run_state,
            log=log_view,
            checkpoints=checkpoints if results is not None else None,
            after_layers=_after_layers_applied if show_results.value else None,
        )
        # Enable revert once the run has actually stopped (success, failure, or
        # the quiet finished-after-quit path). Connecting here rather than
        # inside `_run_in_background` keeps that helper free of panel widgets.
        if worker is not None:
            worker.returned.connect(lambda *_: refresh_revert_buttons())
            worker.errored.connect(lambda *_: refresh_revert_buttons())
            worker.finished.connect(lambda *_: refresh_revert_buttons())

    def on_clear() -> None:
        """Take our layers out of the viewer, stop the run, and forget state.

        Clearing mid-run used to leave the run going against layers that were
        no longer there, and the panel with a permanently greyed-out Run
        button. Both halves of that are here: the run is asked to stop, and
        everything it left behind is put back -- so the next run can start as
        soon as this one has. Cached resume/checkpoint pickles and Revert's
        skip toggles are also reset so Revert → Clear → Run does not reload
        an old graph.
        """
        if viewer is None:
            return
        removed = _clear_our_layers(viewer)
        stopping = run_state.cancel()
        if stopping:
            # The log is deliberately not among the things a cancel puts back.
            # A half-filled bar and a remembered graph are lies once a run has
            # been stopped, which is why `RunState` resets them; what the run
            # said before it was stopped is still true, and is usually why the
            # user stopped it. So it is marked, not cleared.
            log_view.cancelled()
        view.results = None
        reparent_layer_control_sliders()
        z_depth_slider.setEnabled(False)
        z_depth_slider.setVisible(False)
        arrow_length_slider.setEnabled(False)
        arrow_length_slider.setVisible(False)
        if boundaries is not None:
            boundaries.state.results = None
        discarded_artefacts = False
        settings = _settings()
        removed_paths = discard_cached_artefacts_for_settings(settings)
        discarded_artefacts = bool(removed_paths)
        restored_skips = False
        disconnected: list[str] = []
        for name in SKIP_FOR_RESUME:
            if name in rows:
                try:
                    rows[name].changed.disconnect(snapshot_skip_toggles)
                    disconnected.append(name)
                except (TypeError, RuntimeError):
                    pass
        try:
            for name in SKIP_FOR_RESUME:
                if name in rows and name in skip_toggle_snapshot:
                    if rows[name].value != skip_toggle_snapshot[name]:
                        rows[name].value = skip_toggle_snapshot[name]
                        restored_skips = True
        finally:
            for name in disconnected:
                rows[name].changed.connect(snapshot_skip_toggles)
        if restored_skips:
            apply_prerequisites()
        checkpoints.clear()
        refresh_revert_buttons()
        report.value = clear_message(
            removed,
            stopping,
            discarded_artefacts=discarded_artefacts,
            restored_skips=restored_skips,
        )

    def on_revert(tab_title: str) -> None:
        """Reload the previous tab's end-of-stage state for *tab_title*."""
        if run_state.running:
            report.value = ALREADY_RUNNING
            return
        if viewer is None:
            report.value = "No viewer to restore layers into."
            return
        plan = checkpoints.plan_restore(tab_title, settings=_settings())
        if plan is None:
            report.value = (
                "Nothing to restore: run the pipeline with 'Show each stage "
                "in the viewer' first."
            )
            refresh_revert_buttons()
            return
        results = view.results
        if results is None:
            results = ResultLayers()
            view.results = results
            if boundaries is not None:
                boundaries.state.results = results
        checkpoints.apply_to_results(results, plan.checkpoint)
        _clear_our_layers(viewer)
        for group in plan.groups:
            _apply_layers(viewer, group)
        _after_layers_applied()
        saved_skip_snapshot = dict(skip_toggle_snapshot)
        disconnected: list[str] = []
        for name in SKIP_FOR_RESUME:
            if name in rows:
                try:
                    rows[name].changed.disconnect(snapshot_skip_toggles)
                    disconnected.append(name)
                except (TypeError, RuntimeError):
                    pass
        nonlocal revert_setting_skips
        revert_setting_skips = True
        try:
            for name in plan.skip_settings:
                if name in rows:
                    rows[name].value = False
        finally:
            revert_setting_skips = False
            skip_toggle_snapshot.update(saved_skip_snapshot)
            for name in disconnected:
                rows[name].changed.connect(snapshot_skip_toggles)
        apply_prerequisites()

        def select_restored_tab() -> None:
            """Show the restored stage's tab (M), not the tab that owned Revert (K).

            Selecting immediately covers programmatic / test calls. Scheduling
            again on the next event-loop tick covers a Qt click quirk: finishing
            a button press on tab K can restore that tab's page after we have
            already moved to M, which looked like "revert bounced back".
            """
            titles = [tab_widget.tabText(i) for i in range(tab_widget.count())]
            if plan.tab_title in titles:
                tab_widget.setCurrentIndex(titles.index(plan.tab_title))

        select_restored_tab()
        from qtpy.QtCore import QTimer

        QTimer.singleShot(0, select_restored_tab)
        refresh_revert_buttons()
        report.value = restore_message(plan)

    for title, button in revert_buttons.items():
        button.changed.connect(
            lambda *_args, tab=title: on_revert(tab)
        )

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
    view_controls.native.setObjectName("haemolynx_view_controls")

    panel = QWidget()
    z_depth_parking.setParent(panel)
    # What the panel would send to a run, and what a run would report back,
    # for a test that cannot press buttons and wait.
    panel._haemolynx_values = current_values
    panel._haemolynx_progress = bars
    panel._haemolynx_log = log_view
    panel._haemolynx_log_dock = log_dock
    panel._haemolynx_run = on_run
    panel._haemolynx_clear = on_clear
    panel._haemolynx_revert = on_revert
    panel._haemolynx_checkpoints = checkpoints
    panel._haemolynx_revert_buttons = revert_buttons
    panel._haemolynx_revert_stack = revert_stack
    panel._haemolynx_refresh_revert = refresh_revert_buttons
    panel._haemolynx_run_state = run_state
    panel._haemolynx_run_button = run_button
    panel._haemolynx_view = view
    panel._haemolynx_show_results = show_results
    panel._haemolynx_show_steps = show_steps
    panel._haemolynx_view_controls = view_controls
    panel._haemolynx_z_depth_slider = z_depth_slider
    panel._haemolynx_z_depth_host = z_depth_host
    panel._haemolynx_z_depth_parking = z_depth_parking
    panel._haemolynx_reparent_z_depth = reparent_layer_control_sliders
    panel._haemolynx_arrow_length_slider = arrow_length_slider
    panel._haemolynx_arrow_length_host = arrow_length_host
    panel._haemolynx_reparent_arrow_length = reparent_arrow_length_slider
    panel._haemolynx_apply_z_filter = _apply_z_filter
    panel._haemolynx_after_layers_applied = _after_layers_applied
    panel._haemolynx_load_config = load_config_file
    panel._haemolynx_save_config = save_config_file
    panel._haemolynx_report = lambda: report.value
    panel._haemolynx_rows = lambda: rows
    panel._haemolynx_boundaries = boundaries
    panel._haemolynx_perturbations = perturbations
    panel._haemolynx_tabs = tab_widget
    layout = QVBoxLayout(panel)
    if layer_row is not None:
        layout.addWidget(layer_row.native)
    layout.addWidget(tab_widget)
    # Hidden host for Perturbations-claimed rows that are not flat tab chrome
    # (legacy flags + typed-entry Field shells). Keeps them off the screen and
    # out of the top-level window list while config round-trip still reads them.
    if orphaned_perturbation_rows:
        orphan_holder.visible = False
        layout.addWidget(orphan_holder.native)
    # Show-results / show-topology-steps, then Revert centered under them, then
    # the run chrome. Revert is intentionally outside the tab pages so it sits
    # in one place for every stage that can restore a predecessor.
    layout.addWidget(view_controls.native)
    layout.addWidget(revert_stack)
    layout.addWidget(buttons.native)
    layout.addWidget(bars.native)
    layout.addWidget(report.native)
    refresh_revert_buttons()
    return panel
