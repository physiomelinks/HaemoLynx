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
import weakref
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
import functools
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from haemolynx.gui.form import (
    CHANNEL_SETTINGS,
    Field,
    SHARED_ILASTIK_SETTINGS,
    SHARED_ILASTIK_SETTING_SET,
    display_value_for,
    channel_choices,
    prerequisite_chain,
    shared_ilastik_host,
)
from haemolynx.gui.diameter_source import (
    BOTH_ON_NOTE,
    DIAMETER_SOURCES,
    DIAMETER_SOURCE_LABEL,
    DIAMETER_SOURCE_SETTINGS,
    both_on,
    settings_for,
    source_from,
)
from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE
from haemolynx.gui.layers import input_for_layer, voxel_size_xyz_from_scale
from haemolynx.gui.layout import (
    Disclosure,
    TabLayout,
    changed_settings,
    entry_disclosure,
    layout_for,
    role_nodes_disclosure,
    shared_ilastik_disclosure,
)
from haemolynx.gui.log_view import LogView, VERBOSE_LEVEL
from haemolynx.gui.run_log import DEFAULT_LEVEL, attach
from haemolynx.gui.results import (
    BRANCH_HOVER,
    BRANCH_ORDER_SIGNED,
    EDIT_DRAFT,
    FLOW_DIR_COLUMNS,
    FLOW_DIR_RGB_COLUMN,
    FLOW_HEADING_COLUMN,
    FLOW_SOLUTION,
    IMAGE,
    NODES,
    SKELETON,
    VESSEL_TUBES,
    VESSELS,
    ResultLayers,
    copy_graph,
    _flow_dir_contrast_limits,
    _flow_heading_contrast_limits,
    clip_volume_to_z,
    colour_cycle_for,
    filter_points_by_z,
    filter_vectors_by_z,
    is_z_depth_filtered_layer,
    is_z_depth_windowed_volume_layer,
    mask_unsolved_flow_columns,
    perturbation_layer_names,
    z_window_is_full,
)
from haemolynx.gui.optimise_progress import OptimisationProgressDisplay
from haemolynx.gui.optimise_review import proposed_changes, review_text
from haemolynx.gui.progress import ProgressDisplay
from haemolynx.gui.run_state import (
    ALREADY_RUNNING,
    CANCELLED,
    FINISHED_FIRST,
    POST_PROCESS,
    RunCancelled,
    RunState,
    clear_message,
    mid_run_stop_after,
    paused_bar_text,
    paused_message,
    post_processing_tab_title,
    regenerate_stop_after,
)
from haemolynx.gui.stage_checkpoints import (
    SKIP_FOR_RESUME,
    StageCheckpoints,
    can_revert_from,
    discard_cached_artefacts_for_settings,
    output_dir_from_prefix,
    previous_tab,
    restore_message,
    stages_before,
)
from haemolynx.gui.run_snapshot import (
    DEFAULT_FILENAME as RUN_SNAPSHOT_FILENAME,
    SAVE_FILTER as RUN_SNAPSHOT_FILTER,
    RunSnapshotError,
    apply_snapshot_to_checkpoints,
    apply_snapshot_to_results,
    capture_run,
    default_run_path,
    ensure_run_suffix,
    read_run_snapshot,
    replay_groups,
    write_resume_artefacts,
    write_run_snapshot,
)
from haemolynx.gui.tabs import tabs_for
from haemolynx.gui.vessel_tubes import (
    DEFAULT_TUBE_QUALITY,
    DEFAULT_VESSEL_DRAW,
    TUBE_QUALITY_SIDES,
    TUBE_SHADING,
    VESSEL_DRAW_LINES,
    VESSEL_DRAW_TUBES,
    clamp_tube_quality,
    colors_for_tube_vertices,
    vessel_tube_mesh,
    vessel_tubes_layer_name,
)
from haemolynx.io import resolve_voxel_size_xyz
from haemolynx.io.load import _to_binary_volume_for_skeletonization
from haemolynx.optimisation.scorecard import worse_rows
from haemolynx.optimisation import (
    FWHM_GROUP_NAMES,
    FWHM_SETTING_NAMES,
    OPTIMISE_SETTING_NAMES,
    OptimisationEvent,
    build_report_text,
    config_filename,
    optimise_fwhm_settings,
    optimise_skeleton_and_graph_settings,
)
from haemolynx.parsers import (
    Schema,
    dump_config,
    ensure_yaml_suffix,
    load_config,
    parameters_of,
    prefixed_arguments,
    prerequisite_name,
)
from haemolynx.parsers.schema import is_prerequisite_met
from haemolynx.pipeline import (
    default_schema,
    load_volume_for_skeletonise,
    preflight,
    resolve_settings,
    run_pipeline_stages,
    segment,
)
from haemolynx.pipeline.progress import STAGE_STARTED, ProgressEvent, log_progress

logger = logging.getLogger(__name__)

#: What a layer looks like when the user asks for no colouring at all.
UNCOLOURED = "#cccccc"

#: The same grey, as RGBA, for the places that build a colour array by hand.
UNCOLOURED_RGBA = (0.8, 0.8, 0.8, 1.0)

#: Settings the panel starts with switched off, because they put a run's
#: results somewhere other than napari. ``visualize_results`` (Produce IDE
#: plots) writes matplotlib PNG and Plotly HTML, including
#: ``final_graph_3d.html``; the nested Show / mode / hold rows stay at their
#: schema defaults so an untouched panel does not earn an
#: IneffectiveSettingWarning. Tick Produce to write files, then Show plots
#: in IDE if a browser window is wanted -- the viewer is already open.
DISPLAY_SETTINGS_OFF_IN_NAPARI = {
    "visualize_results": False,
}

#: Settings that are not a per-run decision, so they get no row of their
#: own: flow_direction_colouring/flow_arrow_scale are cosmetics for a layer
#: that is always added anyway, and the two FWHM label values are the
#: numbers the rasterised branch label volume marks background and junction
#: voxels with -- internal bookkeeping, held at the schema's defaults. A
#: value is either a constant or a function of the other settings. Never
#: parented as flat tab rows, forced on every call to ``apply_prerequisites``
#: so a loaded config cannot silently change one with no visible control to
#: notice it by.
FORCED_HIDDEN_SETTINGS: dict[str, Any] = {
    "flow_direction_colouring": True,
    "flow_arrow_scale": 1.0,
    "fwhm_background_label": 0,
    "fwhm_junction_label": -1,
}


def forced_hidden_value(name: str, values: Mapping[str, Any]) -> Any:
    """The value a FORCED_HIDDEN_SETTINGS entry takes given ``values``."""
    forced = FORCED_HIDDEN_SETTINGS[name]
    return forced(values) if callable(forced) else forced

#: What napari calls the log window's dock.
LOG_DOCK_NAME = "HaemoLynx run log"

#: View chrome (Z-depth, scale bar, snapshot). Floated over the
#: canvas so a left strip cannot steal height from the bottom run log.
VIEW_DOCK_NAME = "HaemoLynx view"

#: Cosmetic TIFF of the current view; never a pipeline setting or input.
SNAPSHOT_STEM = "haemolynx_snapshot"
SNAPSHOT_NO_OUTPUT_FOLDER = (
    "No output folder is set; set the VTK output prefix before saving a snapshot."
)
SNAPSHOT_NO_VIEWER = "No viewer to snapshot."

#: Metadata key for the unclipped volume on a layer that is not tagged
#: as ours (a user-dropped image). Ours layers keep the same array on the
#: ``OURS`` tag as ``z_window_full``.
Z_WINDOW_CACHE_KEY = "haemolynx_z_window_full"

#: Napari layer unit for a physically meaningful scale bar.
SCALE_BAR_UNIT = "micrometer"

#: Canvas corner for napari's scale-bar overlay (``CanvasPosition`` value).
SCALE_BAR_POSITION = "bottom_right"


def _give_bottom_docks_the_corners(viewer) -> None:
    """Let a bottom dock (the run log) keep the bottom strip of the window.

    QMainWindow's default corners give a left/right dock the bottom-left or
    bottom-right, so a tall side panel shrinks the log to nothing as a run
    fills the viewer.
    """
    from qtpy.QtCore import Qt

    window = getattr(viewer.window, "_qt_window", None)
    if window is None or not hasattr(window, "setCorner"):
        return
    try:
        window.setCorner(Qt.BottomLeftCorner, Qt.BottomDockWidgetArea)
        window.setCorner(Qt.BottomRightCorner, Qt.BottomDockWidgetArea)
    except Exception:  # noqa: BLE001
        logger.debug("could not prefer bottom dock corners", exc_info=True)


def _float_dock_over_canvas(viewer, dock) -> None:
    """Undock *dock* and sit it on the canvas instead of in a side strip."""
    from qtpy.QtCore import QTimer

    setter = getattr(dock, "setFloating", None)
    if setter is None:
        setter = getattr(getattr(dock, "native", None), "setFloating", None)
    if setter is None:
        return
    try:
        setter(True)
    except Exception:  # noqa: BLE001
        logger.debug("could not float dock", exc_info=True)
        return

    inner = dock.widget() if callable(getattr(dock, "widget", None)) else None
    hint = inner.sizeHint() if inner is not None else dock.sizeHint()
    try:
        # A fresh widget's sizeHint() tends to under-report before Qt has
        # laid it out for real, which used to clip "Save snapshot" at the
        # bottom of the panel -- pad past the hint rather than trust it exactly.
        dock.resize(max(int(hint.width()), 280), max(int(hint.height()) + 24, 220))
    except Exception:  # noqa: BLE001
        logger.debug("could not size floating dock", exc_info=True)

    def place() -> None:
        try:
            qt_viewer = getattr(viewer.window, "_qt_viewer", None)
            canvas = getattr(qt_viewer, "canvas", None)
            native = getattr(canvas, "native", canvas)
            if native is None:
                return
            top_left = native.mapToGlobal(native.rect().topLeft())
            dock.move(int(top_left.x()) + 16, int(top_left.y()) + 16)
        except Exception:  # noqa: BLE001
            logger.debug("could not place dock over the canvas", exc_info=True)

    QTimer.singleShot(0, place)


#: Gap between the view-snap buttons and the canvas's bottom-left corner (px).
VIEW_SNAP_MARGIN = 8


def snap_view_to_plane(viewer, plane: str) -> bool:
    """Centre the data and look straight at *plane* ("XY", "XZ" or "YZ").

    A 3D view keeps napari's own axis order and turns the camera; a 2D view
    reorders the dims so the plane's two axes are the ones on screen. Either
    way ``reset_view`` recentres and refits first, so the snap also brings
    back data panned or zoomed out of sight. False (and nothing changes)
    when there is no 3D data to look at a plane of.
    """
    from haemolynx.gui.view_snap import (
        camera_directions_for_plane,
        dims_order_for_plane,
    )

    ndim = int(viewer.dims.ndim)
    if ndim < 3:
        return False
    if int(viewer.dims.ndisplay) == 3:
        viewer.dims.order = tuple(range(ndim))
        viewer.reset_view()
        view_direction, up_direction = camera_directions_for_plane(plane)
        viewer.camera.set_view_direction(view_direction, up_direction)
    else:
        viewer.dims.order = dims_order_for_plane(plane, ndim)
        viewer.reset_view()
    return True


def _install_view_snap_buttons(viewer):
    """XY / XZ / YZ buttons pinned to the canvas's bottom-left corner.

    Parented to the main window: not the vispy canvas, since a child of an
    OpenGL widget does not reliably paint on every platform, and not the Qt
    viewer, which is a QSplitter that adopts every child as a pane and so
    squeezed the canvas to the buttons' width. Moved back to the corner
    whenever the canvas or viewer is resized or moved. One set per viewer: a
    second panel on the same viewer reuses it. None without a Qt window.
    """
    existing = getattr(viewer, "_haemolynx_view_snap_buttons", None)
    if existing is not None:
        return existing
    from qtpy.QtCore import QEvent, QObject, Qt
    from qtpy.QtWidgets import QHBoxLayout, QLayout, QPushButton, QWidget

    from haemolynx.gui.chrome_tooltips import VIEW_SNAP_TOOLTIPS
    from haemolynx.gui.view_snap import VIEW_PLANES

    window = getattr(viewer, "window", None)
    qt_viewer = getattr(window, "_qt_viewer", None) if window is not None else None
    canvas = getattr(qt_viewer, "canvas", None)
    native = getattr(canvas, "native", canvas)
    main_window = getattr(window, "_qt_window", None) if window is not None else None
    if qt_viewer is None or native is None or main_window is None:
        return None

    bar = QWidget(main_window)
    bar.setObjectName("haemolynx_view_snap")
    # Only the buttons show: napari's stylesheet otherwise paints the bar as
    # an opaque strip over the canvas.
    bar.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    bar.setStyleSheet("#haemolynx_view_snap { background: transparent; }")
    layout = QHBoxLayout(bar)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    # The bar is exactly as wide as its buttons, never stretched across the
    # canvas.
    layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
    bar.buttons = {}
    # Weak references only: Qt owns the bar through the viewer, a link the
    # garbage collector cannot see, so a strong one back to the viewer or
    # canvas would keep the closed viewer alive for good.
    viewer_ref = weakref.ref(viewer)
    native_ref = weakref.ref(native)

    def snap(plane: str) -> None:
        live = viewer_ref()
        if live is not None:
            snap_view_to_plane(live, plane)

    for plane in VIEW_PLANES:
        button = QPushButton(plane, bar)
        button.setObjectName(f"haemolynx_view_snap_{plane.lower()}")
        button.setToolTip(VIEW_SNAP_TOOLTIPS[plane])
        button.setFixedWidth(40)
        button.clicked.connect(lambda _checked=False, plane=plane: snap(plane))
        layout.addWidget(button)
        bar.buttons[plane] = button
    bar.adjustSize()
    bar_ref = weakref.ref(bar)

    def place() -> None:
        bar = bar_ref()
        native = native_ref()
        if bar is None or native is None:
            return
        try:
            corner = native.mapTo(bar.parentWidget(), native.rect().bottomLeft())
            bar.move(
                int(corner.x()) + VIEW_SNAP_MARGIN,
                int(corner.y()) - bar.height() - VIEW_SNAP_MARGIN + 1,
            )
            bar.raise_()
        except RuntimeError:
            # The canvas or bar was deleted with the viewer.
            logger.debug("view-snap buttons outlived the canvas", exc_info=True)

    class _FollowCanvas(QObject):
        def eventFilter(self, _watched, event):  # noqa: N802 - Qt's name
            if event.type() in (QEvent.Resize, QEvent.Move, QEvent.Show):
                place()
            return False

    follower = _FollowCanvas(bar)
    native.installEventFilter(follower)
    # Resizing a dock moves the canvas within the window without sending the
    # canvas itself a Move.
    qt_viewer.installEventFilter(follower)
    bar._haemolynx_follow_canvas = follower
    bar.place = place
    place()
    bar.show()
    viewer._haemolynx_view_snap_buttons = bar
    return bar


def _create_widget(**kwargs):
    """magicgui's `create_widget`, imported only when a panel is built."""
    from magicgui.widgets import create_widget

    return create_widget(**kwargs)


def _safe_widget_value(widget) -> Any:
    """A row's raw value, treating unparsable literal text as an empty box.

    `LiteralEvalLineEdit` (the widget behind the ``int_list``/``float_list``/
    ``mapping``/``any`` kinds) evaluates its text as a Python literal on
    every read of ``.value``. Text that is not a complete literal -- most
    often a box a user has cleared by hand, but equally every keystroke of
    typing one out (``[1, 2,`` is a `SyntaxError` mid-edit) -- raises there
    instead of returning. `current_values()` reads every row's `.value` in
    one dict comprehension whenever *any* row changes, so one such widget
    breaks prerequisite handling for the whole panel until its text becomes
    valid again. Treat that the same way `Field.to_setting_value` already
    treats an empty string: unset, not a crash.
    """
    try:
        return widget.value
    except (SyntaxError, ValueError):
        return ""


def _export_dir(values: dict[str, Any]) -> Path:
    """Where a layer with no file behind it gets written: beside the outputs."""
    prefix = values.get("vtk_output_prefix")
    return Path(prefix).parent if prefix else Path.cwd()


def output_folder_from_settings(values: Mapping[str, Any]) -> Path | None:
    """Pipeline output directory (parent of ``vtk_output_prefix``), or None.

    Same folder VTK and other run artifacts use. A snapshot is a cosmetic
    export into this directory; it is never read back as pipeline input.

    A cleared FileEdit arrives as ``"."`` or the working directory rather
    than None. A bare filename (no directory) is treated the same way, so
    snapshots and pickle discards do not land in the process cwd.
    """
    return output_dir_from_prefix(values.get("vtk_output_prefix"))


def _confirm_discard_hand_edits(parent, tab_title: str) -> bool:
    """Ask before a run from *tab_title* drops the Post processing tab's edits."""
    from qtpy.QtWidgets import QMessageBox

    answer = QMessageBox.question(
        parent,
        "Discard the hand edits?",
        f"Running from {tab_title} starts again from a network without the "
        f"edits made on {POST_PROCESSING_TAB}; they will be lost. Run anyway?",
    )
    return answer == QMessageBox.Yes


def _dialog_parent(viewer) -> Any:
    """The napari main window, so file dialogues stay in front of the viewer."""
    if viewer is None:
        return None
    window = getattr(viewer, "window", None)
    if window is None:
        return None
    return getattr(window, "_qt_window", None) or getattr(window, "qt_viewer", None)


def unique_snapshot_path(
    output_dir: Path, *, when: datetime | None = None
) -> Path:
    """``haemolynx_snapshot_YYYYMMDD_HHMMSS.tif``; ``_2``, ``_3``, … if taken."""
    stamp = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
    candidate = Path(output_dir) / f"{SNAPSHOT_STEM}_{stamp}.tif"
    if not candidate.exists():
        return candidate
    n = 2
    while True:
        candidate = Path(output_dir) / f"{SNAPSHOT_STEM}_{stamp}_{n}.tif"
        if not candidate.exists():
            return candidate
        n += 1


#: How far in each nesting level of an Advanced button is drawn, in pixels,
#: and how far in the rows it reveals sit from the button itself.
ADVANCED_INDENT_PX = 14

#: The panel's starting value where it is not the schema default -- what an
#: Advanced button's "changed" count measures against, so Produce IDE plots
#: being off in napari does not read as a change nobody made.
ADVANCED_BASELINE: dict[str, Any] = dict(DISPLAY_SETTINGS_OFF_IN_NAPARI)


def advanced_button_text(title: str, expanded: bool, changed: int = 0) -> str:
    """What an Advanced button says: open or closed, and how much is changed."""
    arrow = "▾" if expanded else "▸"
    note = f" ({changed} changed)" if changed else ""
    return f"{arrow} {title}{note}"


def _hide_row_label(widget) -> None:
    """Drop the empty label a labelled Container gives a nested block.

    magicgui wraps every widget that is not a button in a row with a label,
    pinned to the widest label's width, so a nested block -- an Advanced
    button, the shared ilastik holder -- would otherwise start half-way across
    the tab, squeezed into the value column.
    """
    labeled = widget._labeled_widget()
    if labeled is not None:
        labeled._label_widget.visible = False


def _flush(container) -> None:
    """No margins: a nested block lines up with the rows around it."""
    container.margins = (0, 0, 0, 0)


def _line_up_labels(container) -> None:
    """Give every label in *container* the widest one's width, so values align.

    magicgui does this whenever a widget is added to a container, but not for
    the widgets it is built with -- so a box built whole would keep ragged
    value columns, and no label in it would reach the width
    :func:`_wrap_row_labels` caps and wraps at.
    """
    container._unify_label_widths()


class _AdvancedDisclosure:
    """One "Advanced" button and the rows it reveals.

    :mod:`haemolynx.gui.layout` decides which rows, under which row, and at
    what depth; this draws that. It starts closed. The rows inside keep their
    own prerequisites -- hidden or greyed exactly as they would be outside --
    and the button itself hides when none of them would show, so it never
    opens onto nothing.
    """

    def __init__(self, spec: Disclosure, rows: Mapping[str, Any], *, always_visible: bool = False) -> None:
        from magicgui.widgets import Container, PushButton

        from haemolynx.gui.chrome_tooltips import ADVANCED_SETTINGS_TOOLTIP

        self.spec = spec
        self.names: tuple[str, ...] = tuple(spec.names)
        self.expanded = False
        self.changed = 0
        #: For a block that something else shows and hides (the shared
        #: ilastik rows, which move between two tabs).
        self.always_visible = always_visible
        #: Panel controls placed inside that are not settings: they always show.
        self.extras: list[Any] = []
        self._extras_holder = None
        self.headings: list[tuple[Any, tuple[str, ...]]] = []

        self.button = PushButton(text=advanced_button_text(spec.title, False))
        self.button.tooltip = ADVANCED_SETTINGS_TOOLTIP
        native = self.button.native
        native.setObjectName("haemolynx_advanced")
        native.setAccessibleName(spec.key)
        native.setFlat(True)
        native.setStyleSheet("text-align: left; padding: 2px 4px;")

        self.body = Container(widgets=[], labels=False)
        self.body.margins = (ADVANCED_INDENT_PX, 0, 0, 4)
        self._fill(spec.groups, rows)
        self.container = Container(widgets=[self.button, self.body], labels=False)
        self.container.margins = (ADVANCED_INDENT_PX * spec.level, 0, 0, 0)
        self.body.visible = False
        self.button.changed.connect(lambda *_args: self.toggle())

    def _fill(self, groups, rows: Mapping[str, Any]) -> None:
        from magicgui.widgets import Container, Label

        for heading, names in groups:
            if heading:
                label = Label(value=heading)
                label.native.setStyleSheet("font-weight: bold;")
                self.body.append(label)
                self.headings.append((label, tuple(names)))
            chunk = Container(widgets=[rows[n] for n in names if n in rows], labels=True)
            _flush(chunk)
            _line_up_labels(chunk)
            self.body.append(chunk)

    def set_rows(self, names: Sequence[str], rows: Mapping[str, Any]) -> None:
        """Put these rows, and only these, behind the button (one run, no headings)."""
        self.body.clear()
        self.headings = []
        self._extras_holder = None
        self.names = tuple(names)
        self._fill(((None, tuple(names)),) if names else (), rows)
        for widget in self.extras:
            self._extras_holder_container().append(widget)

    def _extras_holder_container(self):
        from magicgui.widgets import Container

        if self._extras_holder is None:
            self._extras_holder = Container(widgets=[], labels=True)
            _flush(self._extras_holder)
            self.body.append(self._extras_holder)
        return self._extras_holder

    def add(self, widget) -> None:
        """Put a panel control that is not a setting behind the button too."""
        from magicgui.widgets import Container

        self._extras_holder_container().append(widget)
        if isinstance(widget, Container):
            _hide_row_label(widget)
        self.extras.append(widget)

    def toggle(self, expanded: bool | None = None) -> None:
        """Open or close; with no argument, the other way from now."""
        self.expanded = (not self.expanded) if expanded is None else bool(expanded)
        self.body.visible = self.expanded
        self._retitle()

    def _retitle(self) -> None:
        self.button.text = advanced_button_text(self.spec.title, self.expanded, self.changed)

    def refresh(self, shows, values: Mapping[str, Any], schema, baseline=None) -> None:
        """Follow the other settings: whether to show at all, and what is changed.

        *shows* says whether one row would show, given everything above it.
        """
        names = [n for n in self.names if n in schema]
        shown = self.always_visible or bool(self.extras) or any(shows(n) for n in names)
        self.container.visible = shown
        for label, members in self.headings:
            label.visible = any(shows(n) for n in members if n in schema)
        self.changed = len(changed_settings(names, schema, values, baseline))
        self._retitle()


def _box_containers(
    layout: TabLayout, rows: Mapping[str, Any], disclosures: dict, lead=(), replacements=None
):
    """One labelled Container per box of *layout*, with its Advanced buttons in place.

    *lead* (a tab's summary line) heads the first box when that box is
    untitled. Every button made is recorded in *disclosures* by its key.
    *replacements* draws a panel control in a row's place, or nothing for a
    row mapped to None: the row still anchors its Advanced buttons.
    """
    from magicgui.widgets import Container

    replacements = replacements or {}
    made: list[tuple[Any, Any]] = []
    lead = list(lead)
    for index, box in enumerate(layout.boxes):
        widgets = lead if (index == 0 and box.title is None) else []
        buttons = []
        for item in box.items:
            if isinstance(item, str) and item in replacements:
                if replacements[item] is not None:
                    widgets.append(replacements[item])
            elif isinstance(item, str):
                widgets.append(rows[item])
            else:
                disclosure = _AdvancedDisclosure(item, rows)
                buttons.append(disclosure)
                widgets.append(disclosure.container)
        container = Container(widgets=list(widgets), labels=True)
        _line_up_labels(container)
        for disclosure in buttons:
            _hide_row_label(disclosure.container)
            disclosures[disclosure.spec.key] = disclosure
        made.append((box, container))
        if index == 0 and box.title is None:
            lead = []
    if lead:
        # The first box has a title, so the summary stands above it on its own.
        made.insert(0, (None, Container(widgets=lead, labels=True)))
    return made


def _lay_out_boxes(
    layout: TabLayout,
    rows: Mapping[str, Any],
    *,
    disclosures: dict,
    group_boxes: dict,
    lead=(),
    replacements=None,
):
    """Draw *layout*: untitled boxes as plain runs of rows, titled ones as group boxes.

    Returns the drawn page and each box's Container by key. A titled box is
    recorded in *group_boxes* with every setting in it, so the panel can hide
    the whole box -- rather than leave an empty titled frame -- once none of
    them would show.
    """
    from qtpy.QtWidgets import QGroupBox, QVBoxLayout, QWidget

    page = QWidget()
    page_layout = QVBoxLayout(page)
    page_layout.setContentsMargins(0, 0, 0, 0)
    containers: dict[str, Any] = {}
    for box, container in _box_containers(
        layout, rows, disclosures, lead=lead, replacements=replacements
    ):
        if box is None or box.title is None:
            page_layout.addWidget(container.native)
        else:
            group = QGroupBox(box.title)
            group.setObjectName(f"haemolynx_section_{box.key}")
            QVBoxLayout(group).addWidget(container.native)
            page_layout.addWidget(group)
            group_boxes[box.key] = (group, list(box.names))
        if box is not None:
            containers[box.key] = container
    return page, containers


_CHANNEL_COMBO_BOX = None


def _channel_combo_box_class():
    """A drop-down of an image's channels (built on first use, like every
    magicgui import here)."""
    global _CHANNEL_COMBO_BOX
    if _CHANNEL_COMBO_BOX is None:
        from magicgui.widgets import ComboBox

        class ChannelComboBox(ComboBox):
            """Takes any channel it is given, listing it if it must: a value
            written in -- from a config file, before the image path that says
            which channels exist -- is kept, not refused or swapped for
            another channel. :meth:`show_channels` then lists the file's."""

            def _on_value_change(self, value=None) -> None:
                # "No channel" is None, a real choice here: magicgui's base
                # class would swallow the change to it (None is its "unset").
                self.changed.emit(value)

            def _entries(self) -> list[tuple[str, Any]]:
                return [(self.native.itemText(i), data) for i, data in enumerate(self.choices)]

            @ComboBox.value.setter
            def value(self, value):
                if value is not None and value not in self.choices:
                    self.choices = self._entries() + channel_choices([], int(value))[-1:]
                ComboBox.value.fset(self, value)

            def show_channels(self, channels) -> None:
                """List *channels* (from :func:`haemolynx.io.tiff_channels`), keeping the current one."""
                keep = self.value
                self.choices = channel_choices(channels, keep)
                ComboBox.value.fset(self, keep)

        _CHANNEL_COMBO_BOX = ChannelComboBox
    return _CHANNEL_COMBO_BOX


def _follow_image_channels(channel_row, path_row) -> None:
    """Keep *channel_row*'s list to the channels of the file *path_row* names."""
    from haemolynx.io import tiff_channels

    def refresh(*_args) -> None:
        value = path_row.value
        path = None if value in (None, "", ".") else Path(value)
        channel_row.show_channels(
            tiff_channels(path) if path is not None and path.is_file() else []
        )

    path_row.changed.connect(refresh)
    refresh()


def _build_row(field: Field):
    """One magicgui widget for one form row."""
    if field.name in CHANNEL_SETTINGS:
        widget = _channel_combo_box_class()(
            choices=field.options["choices"], value=field.value, name=field.name,
            label=field.label,
        )
        widget.tooltip = field.help
        return widget
    widget = _create_widget(
        value=field.value,
        name=field.name,
        label=field.label,
        widget_type=field.widget_type,
        options=dict(field.options),
    )
    widget.tooltip = field.help
    if field.placeholder:
        # magicgui's create_widget has no placeholder kwarg; the backend
        # (Qt) does. An empty box means None, so this is a LineEdit -- see
        # widget_type_for -- or, for a path, a FileEdit, whose own native is
        # a container around the line edit that actually holds the text.
        target = getattr(widget, "line_edit", widget)
        set_placeholder = getattr(target.native, "setPlaceholderText", None)
        if callable(set_placeholder):
            set_placeholder(field.placeholder)
    return widget


def _scale_layer_from_its_file(
    layer, path, axis_order: str = "zyx"
) -> tuple[float, float, float] | None:
    """Give *layer* the voxel size its own file describes, if it has none.

    napari's readers do not apply a TIFF's resolution tags, so a stack opened
    by dragging it in sits at one unit per voxel while everything this plugin
    draws is in microns -- node `pos` and edge `voxels` are physical already.
    On the nerve stack that is 2.029 um of z drawn as 1, so the image ends up
    at 58% of its depth and the vessels do not lie on the vessels.

    Only ever fills a gap: a layer whose scale someone has already set is left
    alone, and so is a file whose tags say nothing. Returns the scale applied,
    or None if it left the layer as it was. The layer's axes are the file's,
    which *axis_order* (``image_axis_order``) names.
    """
    from haemolynx.io import file_axis_spacing_from_xyz, read_voxel_size_xyz

    if path is None or voxel_size_xyz_from_scale(getattr(layer, "scale", None)):
        return None
    found = read_voxel_size_xyz(path, axis_order)
    if found is None:
        return None
    scale = file_axis_spacing_from_xyz(found[0], axis_order)
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

    def pause(self, message: str) -> None:
        self.display.pause(message)
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


class _OptimiseReview:
    """One optimiser's proposal, shown under its button until Apply or Discard.

    An optimisation run hands :meth:`propose` its result and a function that
    writes it into the panel; this shows what would change against the
    panel's current values (see :mod:`haemolynx.gui.optimise_review`) and
    keeps the function until the user decides. Nothing reaches the panel's
    rows before Apply. Built and used on the GUI thread only.
    """

    def __init__(self, current_values, report, *, what: str) -> None:
        from magicgui.widgets import Container, PushButton, TextEdit

        from haemolynx.gui.chrome_tooltips import OPTIMISE_TOOLTIPS

        self._current_values = current_values
        self._report = report
        self._what = what
        self._apply = None
        self.text = TextEdit(value="")
        self.text.read_only = True
        self.text.tooltip = OPTIMISE_TOOLTIPS["review"]
        self.apply_button = PushButton(text="Apply")
        self.apply_button.tooltip = OPTIMISE_TOOLTIPS["apply"]
        self.discard_button = PushButton(text="Discard")
        self.discard_button.tooltip = OPTIMISE_TOOLTIPS["discard"]
        buttons = Container(
            widgets=[self.apply_button, self.discard_button], layout="horizontal", labels=False
        )
        self.container = Container(widgets=[self.text, buttons], labels=False)
        self.container.visible = False
        self.apply_button.changed.connect(lambda *_args: self.apply())
        self.discard_button.changed.connect(lambda *_args: self.discard())

    @property
    def pending(self) -> bool:
        return self._apply is not None

    def propose(self, result, apply) -> str:
        """Show *result*'s proposal and keep *apply* for the Apply button; the
        status line to report."""
        changes = proposed_changes(result.settings, self._current_values())
        self.text.value = review_text(changes, result.scorecard)
        self._apply = apply
        self.container.visible = True
        count = len(changes)
        worse = worse_rows(result.scorecard)
        warning = (
            f" WARNING: worse than your settings on {len(worse)} measure"
            f"{'s' if len(worse) != 1 else ''}."
            if worse
            else ""
        )
        return (
            f"{self._what[0].upper()}{self._what[1:]} ready to review below: {count} "
            f"change{'s' if count != 1 else ''}.{warning} Apply or Discard."
        )

    def apply(self) -> None:
        if self._apply is None:
            return
        apply, self._apply = self._apply, None
        apply()
        self.container.visible = False
        self._report.value = f"Applied the {self._what}."

    def discard(self) -> None:
        if self._apply is None:
            return
        self._apply = None
        self.container.visible = False
        self._report.value = f"Discarded the {self._what}; the panel is unchanged."

    def clear(self) -> None:
        """Forget a proposal without a word: the state it belonged to is gone."""
        self._apply = None
        self.container.visible = False


class OptimiseProgressBars:
    """The "Optimise settings" run's own two progress bars.

    Structurally identical to :class:`ProgressBars`, but reading
    :class:`~haemolynx.gui.optimise_progress.OptimisationProgressDisplay`
    instead -- kept separate rather than generalising `ProgressBars`, so the
    pipeline run's well-tested bars are untouched. Every method here touches
    Qt, so all of them must be called on the GUI thread.
    """

    def __init__(self) -> None:
        from qtpy.QtWidgets import QProgressBar, QVBoxLayout, QWidget

        self.display = OptimisationProgressDisplay()
        self.group_bar = QProgressBar()
        self.candidate_bar = QProgressBar()
        self.native = QWidget()
        layout = QVBoxLayout(self.native)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.group_bar)
        layout.addWidget(self.candidate_bar)
        self._refresh()

    def start(self) -> None:
        self.display.start()
        self._refresh()

    def show_event(self, event: OptimisationEvent) -> None:
        self.display.update(event)
        self._refresh()

    def finish(self, message: str = "Optimised") -> None:
        self.display.finish(message)
        self._refresh()

    def fail(self, message: str = "Failed") -> None:
        self.display.fail(message)
        self._refresh()

    def reset(self) -> None:
        self.display.reset()
        self._refresh()

    def _refresh(self) -> None:
        for bar, state in (
            (self.group_bar, self.display.groups),
            (self.candidate_bar, self.display.candidates),
        ):
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

#: View-only vessels drawing: tubes (default) or napari Vectors line ribbons.
#: Survives layer rebuilds within a napari session; not a pipeline setting.
_vessel_draw_mode: str = DEFAULT_VESSEL_DRAW
#: The tubes' render quality (see ``TUBE_QUALITY_SIDES``), shared by every
#: tubes layer in the session and set from the slider on their controls.
_tube_quality: int = DEFAULT_TUBE_QUALITY

#: Keys stashed on a branch-hover LayerSpec that must not reach napari.
_BRANCH_HOVER_OPTION_KEYS = frozenset(
    {"branch_hover_available", "branch_hover_selected"}
)

#: Keys stashed on the skeleton LayerSpec that must not reach napari.
_THICK_THIN_OPTION_KEYS = frozenset({"thick_vessel_mask"})

#: Keys stashed on the IMAGE LayerSpec that must not reach napari.
_SEGMENTATION_CLEANUP_OPTION_KEYS = frozenset({"raw_segmented_image"})

#: Debug colours for the skeleton layer's thick/thin toggle -- distinct from
#: every mask/vessel-type colour already in use elsewhere in the viewer.
THIN_SKELETON_COLOUR: tuple[float, float, float, float] = (0.0, 0.85, 0.85, 1.0)
THICK_SKELETON_COLOUR: tuple[float, float, float, float] = (1.0, 0.55, 0.0, 1.0)

#: Debug colours for the segmentation-cleanup toggle: unchanged voxels,
#: voxels cleanup added (reconnect/smooth), voxels cleanup removed
#: (smooth/remove-small) -- three colours, not one "changed" colour, so a
#: user can tell an over-eager reconnect bridge (green) apart from a
#: cleanup step that deleted real vessel (red) at a glance.
SEGMENTATION_CLEANUP_UNCHANGED_COLOUR: tuple[float, float, float, float] = (0.78, 0.78, 0.82, 0.35)
SEGMENTATION_CLEANUP_ADDED_COLOUR: tuple[float, float, float, float] = (0.20, 0.85, 0.25, 1.0)
SEGMENTATION_CLEANUP_REMOVED_COLOUR: tuple[float, float, float, float] = (0.90, 0.20, 0.20, 1.0)


def _thick_thin_skeleton_labels(skeleton: np.ndarray, thick_vessel_mask: np.ndarray) -> np.ndarray:
    """Skeleton voxels as three labels: 0 background, 1 thin, 2 thick.

    A voxel is "thick" when it is both skeleton and inside the fat catchment
    ``skeletonize_thickness_gated`` built the tree from; every other skeleton
    voxel is "thin". Pure and shape-agnostic -- the two inputs just need to
    broadcast together.
    """
    def labels_for(skel, thick):
        skel = np.asarray(skel, dtype=bool)
        thick = np.asarray(thick, dtype=bool)
        labels = np.zeros(skel.shape, dtype=np.uint8)
        labels[skel] = 1
        labels[skel & thick] = 2
        return labels

    if _on_disk(skeleton) and np.shape(thick_vessel_mask) == np.shape(skeleton):
        return _labels_by_slab(labels_for, skeleton, thick_vessel_mask)
    return labels_for(skeleton, thick_vessel_mask)


def _on_disk(*volumes) -> bool:
    """A disk-backed (low-RAM option) volume: its display labels are built a
    slab at a time into a new disk-backed array, with the same values."""
    return any(isinstance(v, np.memmap) for v in volumes)


def _labels_by_slab(labels_for, *volumes) -> np.ndarray:
    from haemolynx.preprocessing.memmap_support import (
        LOW_MEMORY_BLOCK_VOXELS,
        delete_when_unmapped,
        new_memmap_array,
    )

    shape = volumes[0].shape
    # Made only to be displayed: the file goes when the layer drops it.
    out = delete_when_unmapped(new_memmap_array(shape, np.uint8))
    step = max(1, LOW_MEMORY_BLOCK_VOXELS // max(1, int(np.prod(shape[1:], dtype=np.int64))))
    for start in range(0, shape[0], step):
        out[start:start + step] = labels_for(*(v[start:start + step] for v in volumes))
    return out


def _as_uint8_layer_data(volume: np.ndarray) -> np.ndarray:
    """``volume.astype(np.uint8)`` -- an independent copy a Labels layer can be
    painted on -- written a slab at a time to disk for a disk-backed volume."""
    if _on_disk(volume):
        return _labels_by_slab(lambda slab: np.asarray(slab).astype(np.uint8), volume)
    return volume.astype(np.uint8)


def _is_ours(layer) -> bool:
    return bool(getattr(layer, "metadata", {}).get(OURS))


def _is_vessel_vectors_layer(layer) -> bool:
    """Whether *layer* is the HaemoLynx vessels Vectors source of truth."""
    if not _is_ours(layer):
        return False
    if layer.__class__.__name__ != "Vectors":
        return False
    name = str(getattr(layer, "name", ""))
    return name == VESSELS or name.startswith(f"{VESSELS} ")


def _is_vessel_tubes_layer(layer) -> bool:
    """Whether *layer* is the HaemoLynx vessel-tube Surface."""
    if not _is_ours(layer):
        return False
    name = str(getattr(layer, "name", ""))
    if name == VESSEL_TUBES or name.startswith(f"{VESSEL_TUBES} "):
        return True
    tag = getattr(layer, "metadata", {}).get(OURS) or {}
    return tag.get("role") == "vessel_tubes"


def _vector_edge_rgba(layer) -> np.ndarray:
    """Per-segment RGBA currently drawn on a Vectors layer."""
    n = len(np.asarray(getattr(layer, "data", ()), dtype=float))
    colours = np.asarray(getattr(layer, "edge_color", ()), dtype=float)
    if colours.ndim == 2 and len(colours) == n and colours.shape[1] >= 3:
        if colours.shape[1] == 3:
            return np.concatenate(
                [colours, np.ones((n, 1), dtype=float)], axis=1
            )
        return np.asarray(colours[:, :4], dtype=float)
    out = np.ones((n, 4), dtype=float)
    out[:, :3] = UNCOLOURED_RGBA[:3]
    return out


def _retint_vessel_tubes(viewer, vessels) -> None:
    """Copy Vectors edge colours onto the matching tube Surface vertices."""
    if viewer is None or _vessel_draw_mode != VESSEL_DRAW_TUBES:
        return
    name = vessel_tubes_layer_name(vessels.name)
    if name not in getattr(viewer, "layers", {}):
        return
    tubes = viewer.layers[name]
    if not _is_vessel_tubes_layer(tubes):
        return
    tag = getattr(tubes, "metadata", {}).get(OURS) or {}
    segment_index = tag.get("segment_index")
    if segment_index is None:
        return
    try:
        tubes.vertex_colors = colors_for_tube_vertices(
            np.asarray(segment_index), _vector_edge_rgba(vessels)
        )
    except Exception:  # noqa: BLE001 - a missed retint must not break a run
        logger.debug(
            "could not retint vessel tubes for %s", vessels.name, exc_info=True
        )


def _maybe_retint_vessel_tubes(layer) -> None:
    """Retint tubes after a Vectors colouring, even if napari events are quiet."""
    if not _is_vessel_vectors_layer(layer):
        return
    viewer = getattr(layer, "_haemolynx_viewer", None)
    if viewer is None:
        viewer = getattr(layer, "viewer", None)
    if viewer is not None:
        _retint_vessel_tubes(viewer, layer)


def _ensure_tube_colour_follow(viewer, vessels) -> None:
    """Retint tubes whenever the Vectors edge colours change."""
    vessels._haemolynx_viewer = viewer
    if getattr(vessels, "_haemolynx_follow_tubes", False):
        return
    events = getattr(vessels, "events", None)
    signal = getattr(events, "edge_color", None) if events else None
    if signal is None:
        return

    def _follow(*_args) -> None:
        _retint_vessel_tubes(viewer, vessels)

    signal.connect(_follow)
    vessels._haemolynx_follow_tubes = True


def _set_tube_mesh(layer, vertices, faces, colours) -> None:
    """Write vertices, faces and colours without a half-updated vispy mesh.

    Napari refreshes the Surface visual on each assignment. Expanding or
    shrinking the mesh while ``vertex_colors`` still has the old length
    raises in vispy (``incorrect number of colors``). Block refresh until
    both agree, then draw once.
    """
    colours = np.asarray(colours, dtype=float)
    if colours.ndim == 1:
        colours = np.empty((0, 4), dtype=float)
    blocker = getattr(layer, "_block_refresh", None)
    if callable(blocker):
        with blocker():
            layer.data = (vertices, faces)
            layer.vertex_colors = colours
    else:
        # No refresh-block guard on this napari build: the two assignments
        # below can each trigger a vispy redraw before the other lands, and a
        # mismatched vertex/vertex_colors length raises "incorrect number of
        # colors" from that redraw -- outside this function's call stack, so
        # nothing here can catch it. Surface the gap instead of risking a
        # silent, hard-to-reproduce crash from an unguarded build.
        logger.warning(
            "Surface layer %r has no _block_refresh; tube mesh updates are "
            "not atomic on this napari build and may crash on resize.",
            getattr(layer, "name", layer),
        )
        layer.data = (vertices, faces)
        layer.vertex_colors = colours
    if getattr(layer, "visible", False):
        refresh = getattr(layer, "refresh", None)
        if callable(refresh):
            refresh()


def _hide_tube_surface(layer) -> None:
    """Stop drawing a tube Surface, including vispy leftovers while hidden."""
    if layer is None:
        return
    try:
        _set_tube_mesh(
            layer,
            np.empty((0, 3), dtype=float),
            np.empty((0, 3), dtype=np.intp),
            np.empty((0, 4), dtype=float),
        )
    except Exception:  # noqa: BLE001 - still hide even if the mesh cannot clear
        logger.debug("could not clear vessel tube mesh", exc_info=True)
    layer.visible = False


def _request_canvas_redraw(viewer) -> None:
    """Ask the vispy canvas to repaint after a vessels drawing-mode change."""
    window = getattr(viewer, "window", None)
    qt_viewer = getattr(window, "_qt_viewer", None) if window else None
    canvas = getattr(qt_viewer, "canvas", None)
    native = getattr(canvas, "native", None)
    update = getattr(native, "update", None)
    if callable(update):
        update()


def _make_surfaces_follow_the_dims_order() -> None:
    """Draw a 3D Surface in the displayed axis order, as napari 0.8 did.

    napari 0.9's Surface slicing hands back the stored vertices unchanged when
    every axis is on screen, ignoring ``dims.displayed``, while Image and
    Vectors layers permute theirs. So once the dims order is anything but
    ``(0, 1, 2)`` in 3D -- a 2D XZ/YZ snap, napari's roll or transpose button,
    all of which outlive a cleared viewer -- the tubes are drawn with their
    axes swapped against the image and the vessel lines they are built from.
    Patched once, and only when that very unpermuted array comes back, so a
    napari that fixes this upstream is left alone.
    """
    try:
        from napari.layers.surface._slice import _SurfaceSliceRequest
    except ImportError:  # napari < 0.9 slices a Surface in displayed order itself
        return
    original = _SurfaceSliceRequest.__call__
    if getattr(original, "_haemolynx_follows_dims_order", False):
        return

    def __call__(self):
        response = original(self)
        displayed = list(self.slice_input.displayed)
        vertices = response.vertices
        if (
            len(self.slice_input.not_displayed)
            or displayed == sorted(displayed)
            or vertices is not self.data[0]
            or np.ndim(vertices) != 2
            or np.shape(vertices)[1] != len(displayed)
        ):
            return response
        return replace(response, vertices=np.asarray(vertices)[:, displayed])

    __call__._haemolynx_follows_dims_order = True
    _SurfaceSliceRequest.__call__ = __call__


def _sync_one_vessel_tubes(viewer, vessels, tubes_on: bool) -> str:
    """Show tubes or line ribbons for one vessels Vectors layer; the name of
    its tubes layer, whether or not it has one yet."""
    _make_surfaces_follow_the_dims_order()
    name = vessel_tubes_layer_name(vessels.name)
    existing = viewer.layers[name] if name in viewer.layers else None
    if existing is not None and not _is_ours(existing):
        name = f"{name} (HaemoLynx)"
        existing = viewer.layers[name] if name in viewer.layers else None

    if not tubes_on:
        vessels.visible = True
        _hide_tube_surface(existing)
        return name

    vertices, faces, segment_index = vessel_tube_mesh(
        getattr(vessels, "data", ()),
        getattr(vessels, "features", None),
        quality=_tube_quality,
        edge_width=getattr(vessels, "edge_width", None),
    )
    if len(vertices) == 0:
        vessels.visible = False
        _hide_tube_surface(existing)
        return name
    colours = colors_for_tube_vertices(segment_index, _vector_edge_rgba(vessels))
    ours = {
        "kind": "surface",
        "role": "vessel_tubes",
        "segment_index": np.asarray(segment_index),
    }
    scale = tuple(float(v) for v in getattr(vessels, "scale", (1.0, 1.0, 1.0)))
    if existing is not None and existing.__class__.__name__.lower() == "surface":
        existing.scale = scale
        extra = dict(getattr(existing, "metadata", {}) or {})
        tag = dict(extra.get(OURS) or {})
        tag.update(ours)
        extra[OURS] = tag
        existing.metadata = extra
        _set_tube_mesh(existing, vertices, faces, colours)
        existing.visible = True
        tubes_layer = existing
    else:
        if existing is not None:
            _process_pending_qt_events()
            viewer.layers.remove(existing)
            _process_pending_qt_events()
        tubes_layer = viewer.add_surface(
            (vertices, faces),
            name=name,
            scale=scale,
            vertex_colors=colours,
            shading=TUBE_SHADING,
            blending="translucent",
            metadata={OURS: ours},
        )
    vessels.visible = False
    _ensure_tube_colour_follow(viewer, vessels)
    try:
        _attach_tube_quality_slider(viewer, tubes_layer)
    except Exception:  # noqa: BLE001 - a missing slider must not stop drawing
        logger.debug("could not attach the tube quality slider", exc_info=True)
    return name


def _tube_quality_sliders(viewer) -> list:
    """Every tube-quality slider on the tubes layers' controls."""
    sliders = []
    for layer in list(getattr(viewer, "layers", ())):
        if not _is_vessel_tubes_layer(layer):
            continue
        controls = _layer_controls(viewer, layer)
        slider = getattr(controls, "_haemolynx_tube_quality", None) if controls else None
        if slider is not None:
            sliders.append(slider)
    return sliders


def set_tube_quality(viewer, quality: int) -> None:
    """Redraw every tube at *quality* (see ``TUBE_QUALITY_SIDES``).

    Every tubes layer's slider is moved to match, since the level is the
    session's, not one layer's. The shading is left as the user has it.
    """
    global _tube_quality
    _tube_quality = clamp_tube_quality(quality)
    for slider in _tube_quality_sliders(viewer):
        if slider.value() != _tube_quality:
            slider.blockSignals(True)
            slider.setValue(_tube_quality)
            slider.blockSignals(False)
    _sync_vessel_tubes(viewer)


def _attach_tube_quality_slider(viewer, layer) -> bool:
    """Put the "Render quality" slider on a tubes layer's controls, once.

    It lives on the layer's own controls, so napari shows it only while a
    tubes layer is selected. Each step right draws the same tubes with more
    sides round them, rounder and costlier to draw. It acts on release, not
    while dragging: a big network takes a moment to rebuild.
    """
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QLabel, QSlider

    from haemolynx.gui.chrome_tooltips import TUBE_QUALITY_TOOLTIP

    controls = _layer_controls(viewer, layer)
    if controls is None:
        return False
    slider = getattr(controls, "_haemolynx_tube_quality", None)
    if slider is not None:
        if slider.value() != _tube_quality:
            slider.blockSignals(True)
            slider.setValue(_tube_quality)
            slider.blockSignals(False)
        return True
    layout = controls.layout()
    if not hasattr(layout, "addRow"):
        return False
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setObjectName("haemolynx_tube_quality")
    slider.setRange(0, len(TUBE_QUALITY_SIDES) - 1)
    slider.setSingleStep(1)
    slider.setPageStep(1)
    slider.setTickPosition(QSlider.TickPosition.TicksBelow)
    slider.setTickInterval(1)
    slider.setTracking(False)
    slider.setValue(_tube_quality)
    slider.setToolTip(TUBE_QUALITY_TOOLTIP)
    label = QLabel("render quality:")
    label.setToolTip(TUBE_QUALITY_TOOLTIP)
    slider.valueChanged.connect(lambda value: set_tube_quality(viewer, value))
    layout.addRow(label, slider)
    controls._haemolynx_tube_quality = slider
    return True


def _sync_vessel_tubes(viewer) -> None:
    """Rebuild or hide tube Surfaces to match the session drawing mode.

    Only for the network the view panel is showing (see
    :func:`show_layer_set`): the baseline's vessels normally, a
    perturbation's once it is chosen -- and then every other network's tubes
    go, and its vessels are left as the swap left them, hidden.
    """
    if viewer is None:
        return
    tubes_on = _vessel_draw_mode == VESSEL_DRAW_TUBES
    shown = _layer_set_shown(viewer)
    seen_tube_names: set[str] = set()
    try:
        for layer in list(viewer.layers):
            if shown is None:
                if not _is_vessel_vectors_layer(layer):
                    continue
            elif not (_is_ours(layer) and layer.__class__.__name__ == "Vectors"
                      and _layer_set_of(layer) == shown
                      and layer.name == perturbation_layer_names(shown)[0]):
                continue
            seen_tube_names.add(_sync_one_vessel_tubes(viewer, layer, tubes_on))
        if not tubes_on or shown is not None or _perturbation_sets_in(viewer):
            for layer in list(viewer.layers):
                if _is_vessel_tubes_layer(layer) and layer.name not in seen_tube_names:
                    _hide_tube_surface(layer)
        _request_canvas_redraw(viewer)
    except Exception:  # noqa: BLE001 - drawing must not stop a run
        logger.exception("could not sync vessel tubes")


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


def _uniform_colour(colours: Any) -> np.ndarray | None:
    """The one colour every row shares, or None when rows differ (or none)."""
    if colours is None:
        return None
    rows = np.asarray(colours)
    if rows.ndim == 1:
        return rows
    if rows.ndim != 2 or len(rows) == 0:
        return None
    return rows[0] if np.all(rows == rows[0]) else None


def _set_z_filtered_layer_data(
    viewer,
    layer,
    kind: str,
    data: Any,
    features: Mapping[str, np.ndarray],
    segment_owner: Any,
):
    """Write filtered geometry; recreate when shrinking avoids stale Vectors draw.

    Returns the layer the data ended up on (a new one when recreated).

    The recreate path below removes the old layer and immediately adds its
    replacement -- the same ``viewer.layers.remove`` that
    ``_clear_our_layers`` found could crash the process outright (a native
    access violation, not a catchable Python exception): vispy tears down
    that layer's GL resources synchronously, and queued-but-unflushed GL
    work from recent interaction racing with that teardown is what
    crashed. This is the Z-depth slider's own version of the same
    operation -- shrinking the visible vessel count recreates the Vectors/
    Points layer on essentially every drag -- so it gets the same
    before-and-between event-queue drain.
    """
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
            for key in ("size", "shown", "symbol"):
                if hasattr(layer, key):
                    add_kwargs[key] = _uniform_points_property(
                        getattr(layer, key), new_count
                    )
            if hasattr(layer, "out_of_slice_display"):
                add_kwargs["out_of_slice_display"] = layer.out_of_slice_display
            colour_attr, colour = "face_color", getattr(layer, "face_color", None)
        # Only a single flat colour carries over as-is. Per-row colours belong
        # to the old rows -- the caller re-applies the colouring rule instead
        # (see _colouring_rule); pasting them onto a shorter layer gave its
        # vessels other vessels' colours.
        colour = _uniform_colour(colour)
        _process_pending_qt_events()
        viewer.layers.remove(layer)
        _process_pending_qt_events()
        new_layer = adder(data, **add_kwargs)
        if colour is not None:
            setattr(new_layer, colour_attr, colour)
        return new_layer
    layer.data = data
    if features:
        layer.features = features
    if kind == "points":
        _sync_points_per_point_properties(layer, new_count)
    if segment_owner is not None:
        metadata = dict(getattr(layer, "metadata", {}) or {})
        ours = dict(metadata.get(OURS) or {})
        ours["segment_owner"] = np.asarray(segment_owner)
        metadata[OURS] = ours
        layer.metadata = metadata
    return layer


def _apply_z_filter(
    viewer,
    z_min: float,
    z_max: float,
    *,
    z_extent: float | None = None,
) -> None:
    """Redraw graph Vectors/Points layers filtered to a physical Z band."""
    full_range = z_window_is_full(z_min, z_max, z_extent)
    # A snapshot, not the live list: recreating a layer on shrink removes it
    # and appends the replacement, which reindexes viewer.layers mid-loop and
    # silently skips whichever layer landed on the vacated index.
    for layer in list(viewer.layers):
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
        rule = _colouring_rule(layer)
        layer = _set_z_filtered_layer_data(
            viewer, layer, kind, data, features, segment_owner
        )
        # Controls first: a recreated layer's new colour-range control takes
        # its column as newly chosen and fits the range to the rows left in
        # the window. The rule, applied after, puts back the full layer's.
        try:
            _attach_colour_scale(viewer, layer)
        except Exception:  # noqa: BLE001 - a missing colour bar is survivable
            logger.debug(
                "could not reattach colour controls to %s",
                getattr(layer, "name", "?"), exc_info=True,
            )
        _reapply_colouring(layer, rule, cache["features"])
        # _set_z_filtered_layer_data recreates the layer (a new object) when
        # the window shrinks -- the branch-hover mouse-move callback and the
        # "branch tooltip metrics" checkbox panel were attached to the old
        # one and do not carry over, so hovering the recreated layer would
        # silently stop showing a tooltip until the next real pipeline run.
        # _apply_layers attaches these the same way for a freshly-built
        # layer; do it here too so a recreated layer keeps working, and a
        # merely-updated one is a cheap no-op (_attach_branch_hover_controls
        # only rebuilds the panel when it is missing). The same goes for the
        # Colour by / colour map / colour range controls (attached above):
        # napari builds fresh controls for the new layer, and without them a
        # narrowed window left the vessels with no way to change their
        # colouring at all.
        try:
            _refresh_layer_controls(viewer, layer)
        except Exception:  # noqa: BLE001 - a stale readout is survivable
            logger.debug(
                "could not refresh colour controls on %s",
                getattr(layer, "name", "?"), exc_info=True,
            )
        try:
            _attach_branch_hover_controls(viewer, layer)
        except Exception:  # noqa: BLE001 - missing hover panel is survivable
            logger.debug(
                "could not reattach branch-hover controls to %s",
                getattr(layer, "name", "?"), exc_info=True,
            )
    _sync_vessel_tubes(viewer)


def _colouring_rule(layer) -> dict[str, Any] | None:
    """How *layer* is coloured, as a rule another set of its rows can take.

    Not the colours themselves: those are one RGBA row per item, computed for
    the rows the layer held, and meaningless once the Z-depth filter changes
    which rows it holds -- copying them across drew every remaining vessel in
    some other vessel's flow colour.
    """
    column = _active_column(layer)
    if not column:
        return None
    rule: dict[str, Any] = {"column": column}
    if getattr(layer, f"{_colour_attribute(layer)}_mode", None) == "colormap":
        rule["colormap"] = getattr(layer, _colormap_attribute(layer), None)
        try:
            low, high = getattr(layer, _contrast_limits_attribute(layer))
            rule["limits"] = (float(low), float(high))
        except (TypeError, ValueError, AttributeError):
            pass
    return rule


def _reapply_colouring(layer, rule, full_features: Mapping[str, Any] | None) -> None:
    """Colour *layer*'s current rows by *rule*, exactly as the full layer was.

    The colour scale comes from the full layer, never from the rows left in
    the Z window: a continuous column keeps its contrast limits and colormap,
    and a text column its full set of labels, so a vessel is the same colour
    whatever window it is seen through.
    """
    if rule is None:
        return
    column = rule["column"]
    if column == FLOW_DIR_RGB_COLUMN:
        _colour_layer(layer, column, "direct")
        return
    if _is_text_column(layer, column):
        values = (full_features or {}).get(column)
        if values is None:
            values = layer.features[column]
        _colour_layer(layer, column, "categorical", colour_cycle_for(values))
        return
    limits = rule.get("limits")
    if limits is None and full_features is not None and column in full_features:
        try:
            full = np.asarray(full_features[column], dtype=float)
        except (TypeError, ValueError):
            full = np.asarray([], dtype=float)
        finite = full[np.isfinite(full)]
        if finite.size and float(finite.max()) > float(finite.min()):
            limits = (float(finite.min()), float(finite.max()))
    _colour_layer(layer, column, "continuous", (), limits)
    colormap = rule.get("colormap")
    if colormap is not None and getattr(layer, f"{_colour_attribute(layer)}_mode", None) == "colormap":
        setattr(layer, _colormap_attribute(layer), colormap)
        if limits is not None:
            _apply_contrast_limits(layer, *limits)


def _z_window_cache(layer) -> np.ndarray | None:
    """The unclipped volume stored for the view-only Z depth filter, if any."""
    metadata = getattr(layer, "metadata", None) or {}
    tag = metadata.get(OURS) or {}
    cached = tag.get("z_window_full")
    if cached is None:
        cached = metadata.get(Z_WINDOW_CACHE_KEY)
    if cached is None:
        return None
    # As stored: np.asarray would turn a disk-backed (low-RAM option) volume
    # into a plain ndarray view, and clip_volume_to_z tells the two apart by
    # type to decide whether its clipped copy goes to disk or to RAM.
    return cached if isinstance(cached, np.ndarray) else np.asarray(cached)


def _store_z_window_cache(layer, data: Any = None) -> None:
    """Remember the full volume so the Z depth filter can restore it.

    A reference, not a copy: the Z filter never writes into it (a clipped
    view is always a new array), and a copy doubled every volume's memory --
    and, for a disk-backed volume, read the whole of it into RAM, which is
    what the low-RAM option exists to avoid.
    """
    kind = layer.__class__.__name__.lower()
    if not is_z_depth_windowed_volume_layer(kind):
        return
    if data is None:
        data = layer.data
    stored = data if isinstance(data, np.ndarray) else np.asarray(data)
    metadata = dict(getattr(layer, "metadata", {}) or {})
    if _is_ours(layer):
        tag = dict(metadata.get(OURS) or {})
        tag["z_window_full"] = stored
        metadata[OURS] = tag
    else:
        metadata[Z_WINDOW_CACHE_KEY] = stored
    layer.metadata = metadata


def data_for_pipeline(layer) -> Any:
    """The full volume a run should read, ignoring the view-only Z-depth filter."""
    cached = _z_window_cache(layer)
    return cached if cached is not None else getattr(layer, "data", None)


def _canvas_screenshot(viewer) -> np.ndarray | None:
    """RGB(A) array of the current napari canvas, or None if it cannot be taken."""
    window = getattr(viewer, "window", None)
    take = getattr(window, "screenshot", None) if window is not None else None
    if take is None:
        take = getattr(viewer, "screenshot", None)
    if take is None:
        return None
    try:
        rgb = take(canvas_only=True, flash=False)
    except TypeError:
        try:
            rgb = take(canvas_only=True)
        except Exception:  # noqa: BLE001
            logger.debug("canvas screenshot failed", exc_info=True)
            return None
    except Exception:  # noqa: BLE001
        logger.debug("canvas screenshot failed", exc_info=True)
        return None
    arr = np.asarray(rgb)
    return arr if arr.size else None


def write_viewer_snapshot(
    viewer, output_dir: Path, *, when: datetime | None = None
) -> Path:
    """Write one 2D TIFF of the current napari canvas into *output_dir*.

    This is a screenshot of what the user sees — visible layers, the Z-depth
    filter, colours, scale bar — not a multi-page dump of layer arrays and
    not the hidden ``z_window_full`` cache. Cosmetic only; never pipeline input.

    Filename: ``haemolynx_snapshot_YYYYMMDD_HHMMSS.tif``, with ``_2``,
    ``_3``, … when that timestamp is already taken.
    """
    import tifffile

    rgb = _canvas_screenshot(viewer)
    if rgb is None:
        raise ValueError("Could not capture the current view.")
    arr = np.ascontiguousarray(rgb)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = unique_snapshot_path(folder, when=when)
    extras: dict[str, Any] = {}
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        extras["photometric"] = "rgb"
    tifffile.imwrite(str(path), arr, **extras)
    return path


def _layer_voxel_size_z(layer) -> float:
    scale = getattr(layer, "scale", None)
    if scale is None:
        return 1.0
    try:
        return float(scale[0]) if len(scale) > 0 else 1.0
    except TypeError:
        return 1.0


def _viewer_z_extent_um(viewer) -> float | None:
    """Physical Z span from visible layers, preferring cached full volumes."""
    max_extent = 0.0
    for layer in viewer.layers:
        kind = layer.__class__.__name__.lower()
        if is_z_depth_windowed_volume_layer(kind):
            full = _z_window_cache(layer)
            data = full if full is not None else getattr(layer, "data", None)
            if data is None:
                continue
            arr = np.asarray(data)
            if arr.ndim < 3:
                continue
            max_extent = max(max_extent, _layer_voxel_size_z(layer) * int(arr.shape[0]))
            continue
        if kind not in ("points", "vectors"):
            # Anything else -- surface (tube mesh), shapes, tracks -- has no
            # single per-row array this function knows how to read, and a
            # tube mesh's ``(vertices, faces, values)`` tuple is genuinely
            # ragged: np.asarray on it raises. Its own vessels Vectors
            # layer already contributes the same extent below.
            continue
        data = getattr(layer, "data", None)
        if data is None:
            continue
        arr = np.asarray(data)
        if kind == "points" and arr.ndim == 2 and arr.shape[-1] >= 1 and len(arr):
            max_extent = max(max_extent, float(np.max(arr[:, 0])))
        elif kind == "vectors" and arr.ndim == 3 and arr.shape[1] == 2 and len(arr):
            max_extent = max(max_extent, float(np.max(arr[:, 0, 0])))
    return max_extent if max_extent > 0.0 else None


def _apply_volume_z_display(
    viewer,
    z_min: float,
    z_max: float,
    *,
    z_extent: float | None = None,
) -> None:
    """Clip image/labels layers to a physical Z window.

    Reads the unclipped ``z_window_full`` cache. Full-range restores the
    cached stack. Does not crop pipeline inputs.
    """
    identity = z_window_is_full(z_min, z_max, z_extent)
    for layer in viewer.layers:
        kind = layer.__class__.__name__.lower()
        if not is_z_depth_windowed_volume_layer(kind):
            continue
        cached = _z_window_cache(layer)
        if cached is None:
            _store_z_window_cache(layer)
            cached = _z_window_cache(layer)
        if cached is None:
            continue
        if identity:
            # Already showing the full volume (every stage's layer refresh
            # lands here): reassigning would re-upload it to the GPU for
            # nothing.
            if layer.data is not cached:
                layer.data = cached
            continue
        dz = _layer_voxel_size_z(layer)
        layer.data = clip_volume_to_z(cached, dz, z_min, z_max, z_extent=z_extent)


def _current_z_depth_window(viewer) -> tuple[float, float, float | None] | None:
    """The Z-depth filter's current ``(z_min, z_max, z_extent)``, or ``None``.

    ``None`` when nothing has run ``apply_view_z`` yet, or the window covers
    the full range (nothing for a caller to compose with). Set by
    ``apply_view_z`` on ``viewer._haemolynx_z_depth_window`` so the thick/thin
    skeleton toggle -- a separate module-level mechanism with no other link
    to that closure -- can clip its own redraws to the same window instead of
    silently discarding it (see ``_apply_thick_thin_skeleton_display``).
    """
    window = getattr(viewer, "_haemolynx_z_depth_window", None) if viewer is not None else None
    if window is None:
        return None
    z_min, z_max, z_extent = window
    if z_window_is_full(z_min, z_max, z_extent):
        return None
    return window


def _scale_bar_overlay(viewer):
    """Napari 0.9 ``canvas.overlays.scale_bar``, else deprecated ``viewer.scale_bar``."""
    overlays = getattr(getattr(viewer, "canvas", None), "overlays", None)
    if overlays is not None:
        getter = getattr(overlays, "get", None)
        overlay = getter("scale_bar") if callable(getter) else None
        if overlay is None:
            overlay = getattr(overlays, "scale_bar", None)
        if overlay is not None:
            return overlay
    return getattr(viewer, "scale_bar", None)


def _vispy_canvas(viewer):
    """The vispy canvas that lazily creates overlay visuals, if the window exists."""
    window = getattr(viewer, "window", None)
    qt_viewer = getattr(window, "_qt_viewer", None) if window is not None else None
    return getattr(qt_viewer, "canvas", None) if qt_viewer is not None else None


def _set_scale_bar_units(viewer) -> None:
    """Label world coordinates as µm so the overlay is physically meaningful."""
    for layer in viewer.layers:
        if not hasattr(layer, "units"):
            continue
        try:
            ndim = int(getattr(layer, "ndim", 0) or 0)
            layer.units = (SCALE_BAR_UNIT,) * ndim if ndim else SCALE_BAR_UNIT
        except Exception:  # noqa: BLE001 - a layer that rejects units is fine
            logger.debug(
                "could not set scale-bar units on %s",
                getattr(layer, "name", layer),
                exc_info=True,
            )


def _ensure_scale_bar_drawn(viewer, overlay, visible: bool) -> None:
    """Create/hide the vispy overlay and flush a redraw.

    Napari 0.9 builds the canvas visual only when the overlay becomes visible,
    and defers unit updates until the next draw. Setting the model flag without
    that visual (or without a paint) leaves the checkbox as a no-op on screen.
    """
    vispy = _vispy_canvas(viewer)
    if vispy is None:
        return
    if visible:
        update = getattr(vispy, "_update_viewer_overlays", None)
        if callable(update):
            try:
                update()
            except Exception:  # noqa: BLE001 - private napari; model flag still set
                logger.debug("could not create the scale-bar visual", exc_info=True)
        units_update = getattr(vispy, "_update_world_units", None)
        if callable(units_update):
            try:
                units_update()
            except Exception:  # noqa: BLE001
                logger.debug("could not refresh scale-bar units", exc_info=True)
    mapping = getattr(vispy, "_viewer_overlay_to_visual", None)
    visuals = mapping.get(overlay, []) if mapping is not None else []
    for visual in visuals:
        node = getattr(visual, "node", None)
        if node is not None:
            node.visible = bool(visible)
    native = getattr(vispy, "native", None)
    if native is not None and hasattr(native, "update"):
        native.update()


def _apply_scale_bar(viewer, visible: bool) -> None:
    """Show or hide napari's canvas scale bar in the bottom-right corner."""
    overlay = _scale_bar_overlay(viewer)
    if overlay is None:
        return
    show = bool(visible)
    if hasattr(overlay, "position"):
        try:
            overlay.position = SCALE_BAR_POSITION
        except Exception:  # noqa: BLE001 - older napari used a different enum
            logger.debug("could not set scale-bar position", exc_info=True)
    if show:
        _set_scale_bar_units(viewer)
    overlay.visible = show
    _ensure_scale_bar_drawn(viewer, overlay, show)


#: 3D rendering an Image layer gets once it becomes the active selection, so
#: its structure is easier to compare against the derived skeleton/graph
#: while it is the thing being inspected. Plain MIP -- napari's own default --
#: has no depth cue at all: it takes the brightest voxel along each ray
#: regardless of how far away it is, which washes a segmented volume's own
#: foreground into whatever noise sits behind it. Attenuated MIP weights
#: nearer voxels more heavily instead, and nearest-neighbour 3D interpolation
#: keeps a binary-ish mask's edges crisp rather than blurred.
IMAGE_FOCUS_RENDER_OPTIONS: dict[str, Any] = {
    "rendering": "attenuated_mip",
    "attenuation": 0.2,
    "interpolation3d": "nearest",
    "gamma": 0.7,
}

#: What every other Image layer's 3D rendering falls back to -- napari's own
#: defaults -- so only the layer actually being inspected stands out.
IMAGE_DEFAULT_RENDER_OPTIONS: dict[str, Any] = {
    "rendering": "mip",
    "attenuation": 0.05,
    "interpolation3d": "linear",
    "gamma": 1.0,
}

#: Opacity the segmented-image layer gets when it is binary-ish (see
#: results.binary_value_range) and is the active selection, vs. every other
#: time -- rendering/interpolation stay whatever results.py laid out
#: (BINARY_IMAGE_VOLUME_OPTIONS's own "translucent" mode, which must not be
#: overwritten back to mip here), so opacity is the one knob adjusted to
#: make the focused layer stand out.
IMAGE_FOCUS_OPACITY_BINARY = 0.85
IMAGE_DEFAULT_OPACITY_BINARY = 0.45


def _focus_image_layer_rendering(viewer) -> None:
    """Give the pipeline's own input-image layer a clearer 3D render while
    it is the active selection, and napari's plain default otherwise.

    Scoped to the one layer this feature is for (HaemoLynx's own ``IMAGE``
    layer, the segmented input drawn for comparison against the derived
    skeleton/graph) rather than every napari Image-kind layer in the viewer:
    a user's own reference image dropped into the viewer is not ours to
    touch, and HaemoLynx's own vessel-mask volumes (also Image-kind) have
    their own deliberately-chosen rendering (``MASK_VOLUME_OPTIONS``, crisp
    nearest-neighbour edges) that this must not overwrite either. Display-only,
    like the scale bar and Z-depth filter: never written into settings or
    stage inputs.

    A binary-ish segmented image (see ``results.binary_value_range``) was
    already given its own translucent rendering and transparent-background
    colour at layer-creation time -- napari's MIP-family rendering shows
    only whichever single sample is brightest along a ray, which for a
    dense, space-filling vessel volume is foreground almost everywhere,
    turning it solid regardless of colour. Overwriting that back to
    ``mip``/``attenuated_mip`` here would undo the fix, so only its opacity
    is adjusted for focus; everything else is left as results.py laid it out.
    """
    if viewer is None:
        return
    layer = None
    for candidate in viewer.layers:
        if _is_ours(candidate) and getattr(candidate, "name", None) == IMAGE:
            layer = candidate
            break
    if layer is None:
        return
    focused = layer is viewer.layers.selection.active
    tag = getattr(layer, "metadata", {}).get(OURS) or {}
    if tag.get("binary_render"):
        try:
            layer.opacity = (
                IMAGE_FOCUS_OPACITY_BINARY if focused else IMAGE_DEFAULT_OPACITY_BINARY
            )
        except Exception:  # noqa: BLE001 - a layer that rejects opacity is left alone
            logger.debug("could not set opacity on %s", getattr(layer, "name", layer), exc_info=True)
        return
    options = IMAGE_FOCUS_RENDER_OPTIONS if focused else IMAGE_DEFAULT_RENDER_OPTIONS
    for key, value in options.items():
        try:
            setattr(layer, key, value)
        except Exception:  # noqa: BLE001 - a layer that rejects an option is left alone
            logger.debug(
                "could not set %s on %s", key, getattr(layer, "name", layer),
                exc_info=True,
            )


def _uniform_points_property(value: Any, n: int) -> Any:
    """Return a scalar or length-*n* value napari Points accepts for size/shown/symbol.

    Napari 0.9 broadcasts per-point properties only when ``len(value) == len(data)``.
    After a Z-filter or graph shrink the stale array must become a scalar (when uniform)
    or be dropped to the first entry, not passed through at the old length.
    """
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim == 0 or arr.size == 1:
        val = arr if arr.ndim == 0 else arr.flat[0]
        return val.item() if isinstance(val, np.generic) else val
    flat = arr.ravel()
    if len(flat) == n:
        return value
    if len(flat) > 0:
        try:
            if np.all(flat == flat[0]):
                first = flat[0]
                return first.item() if isinstance(first, np.generic) else first
        except (TypeError, ValueError):
            pass
        first = flat[0]
        return first.item() if isinstance(first, np.generic) else first
    return value


def _sync_points_per_point_properties(
    layer,
    n: int,
    options: Mapping[str, Any] | None = None,
) -> None:
    """Keep ``size`` / ``shown`` / ``symbol`` aligned with ``len(layer.data)``."""
    opts = options or {}
    for key in ("size", "shown", "symbol"):
        if key in opts:
            value = opts[key]
        elif hasattr(layer, key):
            value = getattr(layer, key)
        else:
            continue
        fixed = _uniform_points_property(value, n)
        if fixed is not None:
            setattr(layer, key, fixed)


def _double_range_pair(object_name: str):
    """Min/max ``QDoubleSlider`` pair with a range-slider value API.

    superqt's ``QDoubleRangeSlider`` is not reliably draggable in napari's
    left dock: a factory ``(0, 1)`` µm window on a real stack parks both
    handles on top of each other at the left edge, and handles sitting at
    the groove ends (full range) often miss hit-testing. Two
    ``QDoubleSlider`` s match the working Arrow size control.

    ``valueChanged`` fires on every handle move (so labels can update);
    ``sliderReleased`` fires when a drag ends. Display rebuilds should
    wait for release, not each tick.
    """
    from qtpy.QtCore import Qt, Signal
    from qtpy.QtWidgets import (
        QAbstractSlider,
        QAbstractSpinBox,
        QFormLayout,
        QLabel,
        QLineEdit,
        QSizePolicy,
        QWidget,
    )
    from superqt import QDoubleSlider

    class DoubleRangePair(QWidget):
        valueChanged = Signal(object)
        sliderReleased = Signal()

        def __init__(self):
            super().__init__()
            self._lo = QDoubleSlider(Qt.Orientation.Horizontal)
            self._hi = QDoubleSlider(Qt.Orientation.Horizontal)
            self._lo.setObjectName(f"{object_name}_min")
            self._hi.setObjectName(f"{object_name}_max")
            for slider in (self._lo, self._hi):
                slider.setMinimumHeight(18)
                slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                slider.setRange(0.0, 1.0)
            self._lo.setValue(0.0)
            self._hi.setValue(0.0)
            self._lo_label = QLabel("min")
            self._hi_label = QLabel("max")
            form = QFormLayout(self)
            form.setContentsMargins(0, 0, 0, 0)
            form.setSpacing(2)
            form.addRow(self._lo_label, self._lo)
            form.addRow(self._hi_label, self._hi)
            self._lo.valueChanged.connect(self._on_lo)
            self._hi.valueChanged.connect(self._on_hi)
            self._lo.sliderPressed.connect(self._on_pressed)
            self._hi.sliderPressed.connect(self._on_pressed)
            self._lo.sliderReleased.connect(self._on_released)
            self._hi.sliderReleased.connect(self._on_released)
            self._updating = False
            self._dragging = False
            self._haemolynx_extent_ready = False

        def setEnabled(self, enabled: bool) -> None:  # noqa: N802 - Qt API
            """Grey min/max sliders, companion fields, and an optional outer row."""
            on = bool(enabled)
            row = getattr(self, "_haemolynx_row", None)
            if on and row is not None:
                row.setEnabled(True)
            super().setEnabled(on)
            self._lo.setEnabled(on)
            self._hi.setEnabled(on)
            self._lo_label.setEnabled(on)
            self._hi_label.setEnabled(on)
            for child in self.findChildren(
                (QAbstractSlider, QAbstractSpinBox, QLineEdit, QLabel)
            ):
                child.setEnabled(on)
            outer = getattr(self, "_haemolynx_label", None)
            if outer is not None:
                outer.setEnabled(on)
            if not on and row is not None:
                row.setEnabled(False)

        def isSliderDown(self) -> bool:  # noqa: N802 - Qt API
            """True while either handle is being dragged."""
            return bool(
                self._dragging
                or self._lo.isSliderDown()
                or self._hi.isSliderDown()
            )

        def _on_pressed(self) -> None:
            self._dragging = True

        def _on_released(self) -> None:
            self._dragging = False
            self.sliderReleased.emit()

        def _on_lo(self, value: float) -> None:
            if self._updating:
                return
            if value > self._hi.value():
                self._updating = True
                try:
                    self._hi.setValue(value)
                finally:
                    self._updating = False
            self.valueChanged.emit(self.value())

        def _on_hi(self, value: float) -> None:
            if self._updating:
                return
            if value < self._lo.value():
                self._updating = True
                try:
                    self._lo.setValue(value)
                finally:
                    self._updating = False
            self.valueChanged.emit(self.value())

        def value(self) -> tuple[float, float]:
            lo, hi = float(self._lo.value()), float(self._hi.value())
            return (min(lo, hi), max(lo, hi))

        def setValue(self, pair) -> None:
            lo, hi = float(pair[0]), float(pair[1])
            if lo > hi:
                lo, hi = hi, lo
            self._updating = True
            try:
                self._lo.setValue(lo)
                self._hi.setValue(hi)
            finally:
                self._updating = False
            if not self.signalsBlocked():
                self.valueChanged.emit(self.value())

        def setRange(self, minimum: float, maximum: float) -> None:
            self._lo.setRange(minimum, maximum)
            self._hi.setRange(minimum, maximum)

        def setSingleStep(self, step: float) -> None:
            self._lo.setSingleStep(step)
            self._hi.setSingleStep(step)

        def minimum(self) -> float:
            return float(self._lo.minimum())

        def maximum(self) -> float:
            return float(self._lo.maximum())

        def blockSignals(self, block: bool) -> bool:  # noqa: N802 - Qt API
            self._lo.blockSignals(block)
            self._hi.blockSignals(block)
            return super().blockSignals(block)

    widget = DoubleRangePair()
    widget.setObjectName(object_name)
    widget.setMinimumHeight(44)
    widget.setSizePolicy(
        QSizePolicy.Expanding, QSizePolicy.Fixed
    )
    return widget


def _z_extent_and_step(results: ResultLayers | None, viewer=None) -> tuple[float | None, float]:
    """Physical Z span (µm) and slider step from results or open layers."""
    extent = results.image_z_extent_um() if results is not None else None
    step = None
    if results is not None:
        try:
            step = float(results._voxel_size_zyx[0])  # noqa: SLF001
        except (TypeError, IndexError):
            step = None
    if (extent is None or extent <= 0.0) and viewer is not None:
        extent = _viewer_z_extent_um(viewer)
        if viewer is not None:
            for layer in viewer.layers:
                kind = layer.__class__.__name__.lower()
                if is_z_depth_windowed_volume_layer(kind):
                    step = _layer_voxel_size_z(layer)
                    break
    if step is None or step <= 0.0:
        step = 1.0
    return extent, step


def _sync_z_depth_slider(slider, results: ResultLayers | None, viewer=None) -> None:
    """Set the Z-depth range from the image/skeleton; stay visible on the left.

    Guarded the way ``_park_layer_control_host`` already is: this slider
    lives in a floating dock, and a stage-completion callback firing once
    per stage of a real run has, on real production-scale data, found its
    underlying Qt widget already deleted (``RuntimeError: wrapped C/C++
    object ... has been deleted``) -- reproducible only under that load,
    not in a synthetic test. Left unguarded, the exception is raised
    inside a Qt-queued slot, where PyQt/PySide print it and move on rather
    than propagating it, silently skipping ``slider.setEnabled(True)``
    below and leaving every later stage's sync attempt to fail the same
    way. Catching it here means one destroyed widget degrades to an inert
    Z-depth filter instead of a repeating, swallowed exception on every
    subsequent stage of the run.
    """
    try:
        extent, step = _z_extent_and_step(results, viewer)
        if extent is None or extent <= 0.0:
            slider.setEnabled(False)
            return
        slider.blockSignals(True)
        try:
            slider.setRange(0.0, extent)
            if hasattr(slider, "setSingleStep"):
                # Finer than a voxel so setValue(µm) sticks and handles can move.
                slider.setSingleStep(max(min(step, 0.1), 1e-6))
            lo, hi = slider.value()
            uninitialized = not getattr(slider, "_haemolynx_extent_ready", False)
            if uninitialized or hi <= lo or hi > extent or lo < 0.0:
                lo, hi = 0.0, extent
            else:
                lo = max(0.0, min(lo, extent))
                hi = max(lo, min(hi, extent))
            slider.setValue((lo, hi))
            slider._haemolynx_extent_ready = True
        finally:
            slider.blockSignals(False)
        slider.setEnabled(True)
    except RuntimeError:
        logger.debug("Z-depth slider is gone; skipping this sync", exc_info=True)


def _park_layer_control_host(host, parking) -> None:
    """Take *host* out of napari's layer controls without destroying it."""
    try:
        host.setParent(parking)
    except RuntimeError:
        return


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


def _reparent_arrow_length_slider(
    viewer,
    *,
    slider,
    host,
    parking,
    controls_holder: dict[str, Any],
) -> None:
    """Mount the shared arrow-size row at the top of flow-direction controls."""
    _park_layer_control_host(host, parking)
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

    layout.insertRow(0, host)
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
    if is_z_depth_windowed_volume_layer(spec.kind):
        _store_z_window_cache(layer, spec.data)


def _drop_legacy_branch_hover_layer(viewer, group) -> None:
    """Remove leftover midpoint-circle hover Points from earlier GUI builds."""
    emitted = {spec.name for spec in getattr(group, "layers", ())}
    if BRANCH_HOVER in emitted:
        return
    layers = getattr(viewer, "layers", None)
    if layers is None or BRANCH_HOVER not in layers:
        return
    layer = layers[BRANCH_HOVER]
    if _is_ours(layer):
        layers.remove(layer)


def _apply_layers(viewer, group, report=None) -> None:
    """Put one stage's layers in the viewer. Runs on the GUI thread."""
    _drop_legacy_branch_hover_layer(viewer, group)
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
            try:
                _attach_thick_thin_skeleton_toggle(viewer, viewer.layers[spec.name])
            except Exception:  # noqa: BLE001 - missing toggle is survivable
                logger.debug("could not attach thick/thin skeleton toggle to %s",
                             spec.name, exc_info=True)
            try:
                _attach_segmentation_cleanup_toggle(viewer, viewer.layers[spec.name])
            except Exception:  # noqa: BLE001 - missing toggle is survivable
                logger.debug("could not attach segmentation-cleanup toggle to %s",
                             spec.name, exc_info=True)
    for name, column in group.recolour:
        layer = viewer.layers[name] if name in viewer.layers else None
        if layer is not None and _is_ours(layer):
            _colour_layer(layer, column)
    if group.ndisplay is not None:
        viewer.dims.ndisplay = group.ndisplay
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
    _sync_vessel_tubes(viewer)
    if report is not None and group.note:
        report.value = f"{group.title}: {group.note}"


def _apply_layer_groups(viewer, groups, report=None) -> None:
    """Put back what several stage groups leave, drawing each layer once.

    Loading a saved run and "Run from this stage" both replay every stage
    up to some point. Applied one after another, the vessels and nodes
    layers were redrawn once per stage, and where a stage's layer is smaller
    than the one before it :func:`_add_or_update` removes and re-adds it:
    the GL teardown that crashed the process with an access violation in the
    driver (reproduced on a 3,191-vessel network, the second run from the
    Perturbations tab). Pacing the groups a Qt event-loop cycle apart made
    that rarer, not safe. Each layer's final state is all the viewer needs
    (see :func:`~haemolynx.gui.results.merge_stage_layers`).
    """
    from haemolynx.gui.results import merge_stage_layers

    merged = merge_stage_layers(list(groups))
    if merged is not None:
        _apply_layers(viewer, merged, report)


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


def _ensure_flow_dir_features(layer) -> None:
    """Fill missing or NaN ``flow_dir_*`` / ``flow_heading_deg`` from arrows."""
    if layer.__class__.__name__ != "Vectors":
        return
    data = np.asarray(layer.data, dtype=float)
    if data.ndim != 3 or data.shape[1] != 2 or data.shape[0] == 0:
        return
    from haemolynx.visualization.flow_direction import (
        flow_direction_components,
        flow_heading_deg,
    )

    features = dict(getattr(layer, "features", {}))
    displacements = data[:, 1, :]
    components = np.asarray(
        [flow_direction_components(vector) for vector in displacements],
        dtype=float,
    )
    for index, name in enumerate(("flow_dir_z", "flow_dir_y", "flow_dir_x")):
        column = np.asarray(features.get(name, []), dtype=float)
        if column.shape[0] != len(data) or not np.any(np.isfinite(column)):
            features[name] = components[:, index]
            continue
        missing = ~np.isfinite(column)
        if np.any(missing):
            filled = column.copy()
            filled[missing] = components[missing, index]
            features[name] = filled
    heading = np.asarray(features.get(FLOW_HEADING_COLUMN, []), dtype=float)
    if heading.shape[0] != len(data) or not np.any(np.isfinite(heading)):
        features[FLOW_HEADING_COLUMN] = np.asarray(
            [flow_heading_deg(vector) for vector in displacements],
            dtype=float,
        )
    else:
        missing = ~np.isfinite(heading)
        if np.any(missing):
            filled = heading.copy()
            filled[missing] = [
                flow_heading_deg(vector) for vector in displacements[missing]
            ]
            features[FLOW_HEADING_COLUMN] = filled
    rgb_dummy = np.asarray(features.get(FLOW_DIR_RGB_COLUMN, []), dtype=float)
    if rgb_dummy.shape[0] != len(data):
        features[FLOW_DIR_RGB_COLUMN] = np.zeros(len(data), dtype=float)
    layer.features = features


def _flow_dir_rgba(layer) -> np.ndarray:
    """Per-vector RGBA for the 3D direction map, from current ``flow_dir_*``."""
    from haemolynx.visualization.flow_direction import flow_direction_rgba

    _ensure_flow_dir_features(layer)
    n = len(np.asarray(getattr(layer, "data", ()), dtype=float))
    features = getattr(layer, "features", {})
    if n == 0 or "flow_dir_z" not in features:
        return np.zeros((n, 4), dtype=float)
    rgba = flow_direction_rgba(
        features["flow_dir_z"],
        features["flow_dir_y"],
        features["flow_dir_x"],
    )
    if FLOW_SOLUTION in features:
        from haemolynx.gui.branch_hover import FLOW_SOLUTION_TEXT

        # The same grey every other flow-based colouring draws them in.
        unsolved = np.asarray(features[FLOW_SOLUTION], dtype=object) == FLOW_SOLUTION_TEXT[False]
        rgba[unsolved] = UNCOLOURED_RGBA
    return rgba


def _flow_heading_colormap() -> str:
    """Cyclic map for azimuth; fall back when ``hsv`` is unavailable."""
    from napari.utils.colormaps import ensure_colormap

    for name in ("hsv", "twilight_shifted"):
        try:
            ensure_colormap(name)
            return name
        except (KeyError, ValueError, TypeError):
            continue
    return "viridis"


def _default_colormap_for(column: str | None) -> str:
    """The map a colour-by column starts with, before the user overrides it."""
    if column == FLOW_HEADING_COLUMN:
        return _flow_heading_colormap()
    if column in FLOW_DIR_COLUMNS or column == BRANCH_ORDER_SIGNED:
        return "coolwarm"
    return "viridis"


#: Ends the name of the copy :func:`_grey_nan_colormap` makes of a map.
GREY_NAN_SUFFIX = " (grey NaN)"


def _grey_nan_colormap(name: str):
    """Colormap *name*, drawing NaN in the uncoloured grey.

    NaN is a value a row has not got -- the flow of a vessel the solve left
    unsolved, above all (``results.SOLVED_FLOW_COLUMNS``) -- and napari's own
    maps draw it transparent, so those vessels vanished instead of showing
    light grey. *name* itself for anything napari cannot read as a map.
    """
    from napari.utils.colormaps import Colormap, ensure_colormap

    try:
        base = ensure_colormap(_base_colormap_name(name))
    except (KeyError, ValueError, TypeError):
        return name
    return Colormap(
        colors=base.colors,
        controls=base.controls,
        interpolation=base.interpolation,
        name=f"{base.name}{GREY_NAN_SUFFIX}",
        nan_color=UNCOLOURED_RGBA,
        low_color=base.low_color,
        high_color=base.high_color,
    )


def _base_colormap_name(name: str) -> str:
    """The map a :func:`_grey_nan_colormap` copy was made from."""
    return name[: -len(GREY_NAN_SUFFIX)] if name.endswith(GREY_NAN_SUFFIX) else name


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
        _maybe_retint_vessel_tubes(layer)
        return
    if column in FLOW_DIR_COLUMNS or column in {FLOW_HEADING_COLUMN, FLOW_DIR_RGB_COLUMN}:
        _ensure_flow_dir_features(layer)
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
    if column == FLOW_DIR_RGB_COLUMN or kind == "direct":
        # Napari Vectors colour from a 1D feature + LUT cannot represent a
        # sphere; assign per-arrow RGB with R=x, G=y, B=z.
        colours = _flow_dir_rgba(layer)
        for attribute in attributes:
            setattr(layer, attribute, colours)
        _record_colour(layer, column)
        _maybe_retint_vessel_tubes(layer)
        return
    if kind == "categorical" and cycle and len(layer.features[column]):
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
            _maybe_retint_vessel_tubes(layer)
            return
        colormap = _grey_nan_colormap(_default_colormap_for(column))
        for attribute in attributes:
            cmap_attr = f"{attribute.replace('_color', '')}_colormap"
            if hasattr(layer, cmap_attr):
                setattr(layer, cmap_attr, colormap)
            setattr(layer, attribute, column)
        # After the column, and through the same path the Fit buttons use: the
        # range has to be applied *and* the colours re-mapped against it. Set
        # before, and the assignment above maps with the old range; set with a
        # plain setattr, and nothing re-maps at all -- which is how `flow_abs`
        # came to be the selected colouring and not the one on screen.
        if limits is None:
            if column == FLOW_HEADING_COLUMN:
                limits = _flow_heading_contrast_limits(values)
            elif column in FLOW_DIR_COLUMNS:
                limits = _flow_dir_contrast_limits(values)
            else:
                limits = _data_range(layer, column)
        if limits is not None:
            _apply_contrast_limits(layer, *limits)
    _record_colour(layer, column)
    _maybe_retint_vessel_tubes(layer)


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


def _keep_layer_interaction(layer, spec, *, visible, mode) -> None:
    """Restore what the user set on *layer* after a redraw.

    A spec may hide a layer (skeleton after the graph exists) but must not
    un-hide one the user turned off, and assigning ``data`` resets napari's
    edit mode to pan-zoom, which would drop a Draw that was still in progress.
    """
    layer.visible = bool(visible) if spec.visible else False
    if mode is not None and mode != getattr(layer, "mode", None):
        try:
            layer.mode = mode
        except Exception:  # noqa: BLE001 - not every kind has every mode
            logger.debug("could not restore layer mode %s on %s", mode, spec.name)


def _add_or_update(viewer, spec) -> None:
    """Add *spec*, or update the layer of ours already carrying its name."""
    import pandas as pd  # noqa: F401  (napari builds features through pandas)

    existing = viewer.layers[spec.name] if spec.name in viewer.layers else None
    if existing is not None and not _is_ours(existing):
        # Someone else's layer happens to share the name. Never overwrite it.
        spec = replace(spec, name=f"{spec.name} (HaemoLynx)")
        existing = viewer.layers[spec.name] if spec.name in viewer.layers else None

    saved_visible = bool(existing.visible) if existing is not None else spec.visible
    saved_mode = getattr(existing, "mode", None) if existing is not None else None

    if existing is not None and existing.__class__.__name__.lower() == _CLASS_FOR[spec.kind]:
        if spec.kind == "surface":
            # A new mesh and its colours land together, or vispy redraws one
            # against the other's length (see _set_tube_mesh).
            vertices, faces = spec.data[:2]
            _set_tube_mesh(existing, vertices, faces, spec.options.get("vertex_colors", ()))
            _keep_layer_interaction(existing, spec, visible=saved_visible, mode=saved_mode)
            return
        # Shrinking Vectors/Points in place leaves stale segments on screen.
        if spec.kind in {"vectors", "points"}:
            new_count = len(np.asarray(spec.data))
            old_count = len(np.asarray(existing.data))
            if new_count < old_count:
                # A synchronous GL teardown: let vispy's queued work flush
                # first, as _clear_our_layers does, or the driver can fault.
                _process_pending_qt_events()
                viewer.layers.remove(existing)
                _process_pending_qt_events()
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
            if spec.kind == "points":
                _sync_points_per_point_properties(
                    existing, new_count, spec.options
                )
            _keep_layer_interaction(
                existing, spec, visible=saved_visible, mode=saved_mode
            )
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
            _store_layer_set_metadata(existing, spec)
            _store_branch_hover_metadata(existing, spec)
            _maybe_store_z_filter_cache(existing, spec)
            _store_thick_thin_skeleton_metadata(existing, spec)
            _store_binary_render_metadata(existing, spec)
            _store_segmentation_cleanup_metadata(existing, spec)
            return

    if existing is not None:
        viewer.layers.remove(existing)

    adder = getattr(viewer, f"add_{spec.kind}")
    options = dict(spec.options)
    if spec.kind == "image":
        options = _image_options_for_napari(options)
    for key in _BRANCH_HOVER_OPTION_KEYS:
        options.pop(key, None)
    for key in _THICK_THIN_OPTION_KEYS:
        options.pop(key, None)
    for key in _SEGMENTATION_CLEANUP_OPTION_KEYS:
        options.pop(key, None)
    if spec.features:
        options["features"] = dict(spec.features)
    add_kwargs = {
        "name": spec.name,
        "scale": spec.scale,
        "visible": saved_visible if spec.visible else False,
        "metadata": {OURS: {"kind": spec.kind}},
        **options,
    }
    if spec.kind == "image" and spec.contrast_limits is not None:
        add_kwargs.setdefault("contrast_limits", spec.contrast_limits)
    layer = adder(spec.data, **add_kwargs)
    _keep_layer_interaction(layer, spec, visible=saved_visible, mode=saved_mode)
    _colour_layer(layer, spec.colour_by, spec.colour_kind,
                  spec.colour_cycle, spec.contrast_limits)
    _store_sweep_metadata(layer, spec)
    _store_layer_set_metadata(layer, spec)
    _store_branch_hover_metadata(layer, spec)
    _maybe_store_z_filter_cache(layer, spec)
    _store_thick_thin_skeleton_metadata(layer, spec)
    _store_binary_render_metadata(layer, spec)
    _store_segmentation_cleanup_metadata(layer, spec)


def _store_sweep_metadata(layer, spec) -> None:
    """Keep sweep grid + indexing on the layer so sliders can refresh features."""
    tag = dict(getattr(layer, "metadata", {}).get(OURS) or {})
    tag["kind"] = getattr(spec, "kind", tag.get("kind"))
    if getattr(spec, "sweep", None) is not None:
        tag["sweep"] = spec.sweep
        tag["segment_owner"] = spec.segment_owner
        tag["sweep_edge_index"] = spec.sweep_edge_index
        tag["sweep_directions"] = getattr(spec, "sweep_directions", None)
        tag["colour_by"] = spec.colour_by
        tag["contrast_limits"] = spec.contrast_limits
    else:
        tag.pop("sweep", None)
        tag.pop("segment_owner", None)
        tag.pop("sweep_edge_index", None)
        tag.pop("sweep_directions", None)
    if getattr(spec, "sweep_points", None) is not None:
        tag["sweep_points"] = spec.sweep_points
        tag["sweep_points_colour_cycle"] = spec.colour_cycle
    else:
        tag.pop("sweep_points", None)
        tag.pop("sweep_points_colour_cycle", None)
    metadata = dict(getattr(layer, "metadata", {}) or {})
    metadata[OURS] = tag
    layer.metadata = metadata


def _store_layer_set_metadata(layer, spec) -> None:
    """Remember which perturbation's network *layer* draws, if any."""
    tag = dict(getattr(layer, "metadata", {}).get(OURS) or {})
    key = getattr(spec, "layer_set", None)
    if key is None:
        tag.pop("layer_set", None)
    else:
        tag["layer_set"] = str(key)
    metadata = dict(getattr(layer, "metadata", {}) or {})
    metadata[OURS] = tag
    layer.metadata = metadata


def _layer_set_of(layer) -> str | None:
    """The perturbation whose network *layer* draws; None for anything else."""
    tag = getattr(layer, "metadata", {}).get(OURS) or {}
    key = tag.get("layer_set")
    return None if key is None else str(key)


def _perturbation_sets_in(viewer) -> list[str]:
    """The perturbations with a network in the viewer, in layer order."""
    if viewer is None:
        return []
    keys = (_layer_set_of(layer) for layer in viewer.layers if _is_ours(layer))
    return list(dict.fromkeys(key for key in keys if key is not None))


def _layer_set_shown(viewer) -> str | None:
    """The network the view panel shows: a perturbation's name, or None."""
    return getattr(viewer, "_haemolynx_layer_set", None)


def _apply_layer_set_visibility(viewer, key, roles) -> None:
    from haemolynx.gui.layer_sets import visibility_for

    present = [layer.name for layer in viewer.layers if _is_ours(layer)]
    keys = [None, *_perturbation_sets_in(viewer)]
    for name, visible in visibility_for(key, keys, roles, present).items():
        layer = viewer.layers[name]
        if bool(layer.visible) != visible:
            if not visible and _is_vessel_tubes_layer(layer):
                _hide_tube_surface(layer)
            else:
                layer.visible = visible


def show_layer_set(viewer, key: str | None) -> None:
    """Show one network -- the baseline (None) or a perturbation -- and hide
    every other network's vessels, nodes, flow direction and pericytes.

    Which kinds of layer were on carries across from the network shown
    before (see :mod:`haemolynx.gui.layer_sets`). A perturbation with no
    layers in the viewer shows the baseline instead.
    """
    from haemolynx.gui.layer_sets import carried_roles, role_visibility

    if viewer is None:
        return
    if key is not None and key not in _perturbation_sets_in(viewer):
        key = None
    shown = _layer_set_shown(viewer)
    visible = {layer.name: bool(layer.visible) for layer in viewer.layers if _is_ours(layer)}
    remembered = getattr(viewer, "_haemolynx_layer_set_roles", {}) or {}
    outgoing = role_visibility(shown, visible)
    if shown is not None and shown not in _perturbation_sets_in(viewer):
        # What was shown has gone: fall back to how the baseline was left.
        outgoing = remembered
    roles = carried_roles(role_visibility(key, visible), outgoing, remembered)
    viewer._haemolynx_layer_set = key
    viewer._haemolynx_layer_set_roles = roles
    _apply_layer_set_visibility(viewer, key, roles)
    _sync_vessel_tubes(viewer)


def _reapply_layer_set(viewer) -> None:
    """Keep the chosen network shown after layers were redrawn or removed.

    A redrawn perturbation layer comes back hidden, as every perturbation
    layer starts; a removed one leaves nothing to show, so the baseline
    comes back.
    """
    shown = _layer_set_shown(viewer)
    if shown is None:
        return
    if shown not in _perturbation_sets_in(viewer):
        show_layer_set(viewer, None)
        return
    _apply_layer_set_visibility(
        viewer, shown, getattr(viewer, "_haemolynx_layer_set_roles", {}) or {}
    )


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


def _store_thick_thin_skeleton_metadata(layer, spec) -> None:
    """Keep the boolean skeleton and its fat catchment on the layer.

    Read by the thick/thin debug toggle to recompute its display after every
    stage that re-emits this same layer (build_network, assign_boundaries,
    solve all redraw it), not only the skeletonise run that first computed
    it. Cleared, not left stale, on a run that did not use thickness-gated
    skeletonisation -- otherwise a later rerun with it off would still offer
    a toggle backed by a previous run's mask.
    """
    thick_vessel_mask = spec.options.get("thick_vessel_mask")
    metadata = dict(getattr(layer, "metadata", {}) or {})
    tag = dict(metadata.get(OURS) or {})
    if thick_vessel_mask is None:
        tag.pop("thick_vessel_mask", None)
        tag.pop("skeleton_bool", None)
    else:
        tag["thick_vessel_mask"] = thick_vessel_mask
        tag["skeleton_bool"] = np.asarray(spec.data, dtype=bool)
    metadata[OURS] = tag
    layer.metadata = metadata


def _store_binary_render_metadata(layer, spec) -> None:
    """Remember that *layer* was laid out with the transparent-background
    mask style (``results.BINARY_IMAGE_VOLUME_OPTIONS`` /
    ``MASK_VOLUME_OPTIONS``), independent of which colormap is currently
    applied.

    ``_focus_image_layer_rendering`` used to tell this apart from plain
    grayscale by checking ``layer.colormap.name`` against the custom
    ``haemolynx_vessel_mask`` colormap's own name -- which broke the moment
    a user picked any other colormap from napari's own dropdown: the check
    then read the layer as ordinary grayscale and overwrote its rendering
    back to MIP-family, turning a dense volume solid again (the exact bug
    the translucent rendering exists to avoid) regardless of which
    colormap they chose. A metadata flag set once at layer-creation time
    survives any later colormap change.
    """
    if spec.kind != "image":
        return
    tag = dict(getattr(layer, "metadata", {}).get(OURS) or {})
    if "mask_colour" in spec.options:
        tag["binary_render"] = True
    else:
        tag.pop("binary_render", None)
    metadata = dict(getattr(layer, "metadata", {}) or {})
    metadata[OURS] = tag
    layer.metadata = metadata


def _thick_thin_skeleton_state(layer) -> tuple[np.ndarray | None, np.ndarray | None]:
    """The ``(skeleton_bool, thick_vessel_mask)`` a toggle needs, or ``(None, None)``."""
    tag = getattr(layer, "metadata", {}).get(OURS) or {}
    return tag.get("skeleton_bool"), tag.get("thick_vessel_mask")


def _apply_thick_thin_skeleton_display(
    layer, *, show: bool, default_colormap,
    z_window: tuple[float, float, float | None] | None = None,
) -> None:
    """Recolour the skeleton layer to show thick vs. thin voxels, or revert.

    Reads the layer's own stashed skeleton/thick-mask fresh every time
    (rather than a value captured when the toggle was built), so it stays
    correct across reruns without any separate refresh bookkeeping.
    *default_colormap* is napari's own colormap object, captured once when
    the checkbox was first built (before this ever touched it) -- napari 0.9
    Labels layers have no "reset to default" short of handing the original
    object back. *z_window*, when given, is the Z-depth filter's current
    ``(z_min, z_max, z_extent)`` (see ``_current_z_depth_window``): both
    caches this reads are the full, unclipped volume, so without this the
    Z-depth window and this recolouring would each silently discard
    whichever one the other applied last.
    """
    skeleton_bool, thick_vessel_mask = _thick_thin_skeleton_state(layer)
    if show and skeleton_bool is not None and thick_vessel_mask is not None:
        if z_window is not None:
            z_min, z_max, z_extent = z_window
            dz = _layer_voxel_size_z(layer)
            skeleton_bool = np.asanyarray(
                clip_volume_to_z(skeleton_bool, dz, z_min, z_max, z_extent=z_extent),
                dtype=bool,
            )
            thick_vessel_mask = np.asanyarray(
                clip_volume_to_z(thick_vessel_mask, dz, z_min, z_max, z_extent=z_extent),
                dtype=bool,
            )
        layer.data = _thick_thin_skeleton_labels(skeleton_bool, thick_vessel_mask)
        layer.colormap = {
            1: THIN_SKELETON_COLOUR,
            2: THICK_SKELETON_COLOUR,
            None: (0.0, 0.0, 0.0, 0.0),
        }
        return
    # Revert the colormap even when there is no skeleton_bool to redraw from
    # (a rerun dropped the fat catchment) -- otherwise the layer is left
    # stuck on the two-colour debug scheme with no control left to fix it.
    layer.colormap = default_colormap
    if skeleton_bool is not None:
        data = _as_uint8_layer_data(skeleton_bool)
        if z_window is not None:
            z_min, z_max, z_extent = z_window
            dz = _layer_voxel_size_z(layer)
            data = clip_volume_to_z(data, dz, z_min, z_max, z_extent=z_extent)
        layer.data = data


def _attach_thick_thin_skeleton_toggle(viewer, layer) -> bool:
    """Put the thick/thin debug checkbox on the skeleton layer's controls, once.

    Only offered when this run's skeleton actually carries a fat catchment
    (``use_thick_vessel_skeletonisation`` was on) -- see
    _store_thick_thin_skeleton_metadata. The checkbox itself lives on the
    layer's own native controls, so napari already shows it only while this
    layer is the active selection. Scoped to the skeleton layer by name, not
    just by its napari Labels kind, so a future non-skeleton Labels layer
    does not silently pick up this checkbox and get recoloured from whatever
    unrelated thick_vessel_mask/skeleton_bool metadata happens to be on it.
    """
    from qtpy.QtWidgets import QCheckBox

    if layer.__class__.__name__.lower() != "labels":
        return False
    if getattr(layer, "name", None) != SKELETON:
        return False
    _skeleton_bool, thick_vessel_mask = _thick_thin_skeleton_state(layer)
    available = thick_vessel_mask is not None
    controls = _layer_controls(viewer, layer)
    if controls is None:
        return False
    checkbox = getattr(controls, "_haemolynx_thick_thin", None)
    if checkbox is not None:
        checkbox.setEnabled(available)
        if not available:
            checkbox.setChecked(False)
        _apply_thick_thin_skeleton_display(
            layer, show=checkbox.isChecked(),
            default_colormap=checkbox._haemolynx_default_colormap,
            z_window=_current_z_depth_window(viewer),
        )
        return True
    layout = controls.layout()
    if not hasattr(layout, "addRow"):
        return False
    checkbox = QCheckBox("Thick/thin skeleton (debug)")
    checkbox.setToolTip(
        "Colour skeleton voxels by whether thickness-gated skeletonisation "
        "treated them as fat (thick) or ridge (thin)"
    )
    checkbox.setEnabled(available)
    # Captured once, before this ever recolours the layer: napari 0.9 Labels
    # layers have no "reset to default" short of handing this object back.
    checkbox._haemolynx_default_colormap = layer.colormap
    checkbox.toggled.connect(
        lambda checked, l=layer, cb=checkbox: _apply_thick_thin_skeleton_display(
            l, show=checked, default_colormap=cb._haemolynx_default_colormap,
            z_window=_current_z_depth_window(viewer),
        )
    )
    layout.addRow(checkbox)
    controls._haemolynx_thick_thin = checkbox
    return True


def _resync_thick_thin_skeleton_after_z_change(viewer) -> None:
    """Re-apply the thick/thin recolouring on top of a just-changed Z-depth
    window, for whichever skeleton layer has the toggle checked.

    ``_apply_volume_z_display`` (called just before this, from
    ``apply_view_z``) overwrites the skeleton layer's data from its own
    plain, unclipped cache -- silently discarding the two-colour recolouring
    if the toggle is on. This puts it back, clipped to the same window, so
    the two features compose instead of clobbering each other.
    """
    if viewer is None:
        return
    z_window = _current_z_depth_window(viewer)
    for layer in viewer.layers:
        if layer.__class__.__name__.lower() != "labels":
            continue
        if getattr(layer, "name", None) != SKELETON:
            continue
        controls = _layer_controls(viewer, layer)
        checkbox = getattr(controls, "_haemolynx_thick_thin", None) if controls is not None else None
        if checkbox is None or not checkbox.isChecked():
            continue
        _apply_thick_thin_skeleton_display(
            layer, show=True, default_colormap=checkbox._haemolynx_default_colormap,
            z_window=z_window,
        )


def _store_segmentation_cleanup_metadata(layer, spec) -> None:
    """Keep the pre-cleanup mask on the IMAGE layer, plus the corrected mask
    it is being drawn with, for the raw-vs-corrected debug toggle.

    Mirrors ``_store_thick_thin_skeleton_metadata``. Cleared, not left
    stale, on a run whose segmentation_cleanup_* settings were all off --
    otherwise a later rerun without cleanup would still offer a toggle
    backed by a previous run's mask.
    """
    if spec.kind != "image":
        return
    raw_segmented_image = spec.options.get("raw_segmented_image")
    metadata = dict(getattr(layer, "metadata", {}) or {})
    tag = dict(metadata.get(OURS) or {})
    if raw_segmented_image is None:
        tag.pop("raw_segmented_image", None)
        tag.pop("corrected_mask_bool", None)
    else:
        tag["raw_segmented_image"] = np.asarray(raw_segmented_image, dtype=bool)
        tag["corrected_mask_bool"] = np.asarray(spec.data, dtype=bool)
    metadata[OURS] = tag
    layer.metadata = metadata


def _segmentation_cleanup_state(layer) -> tuple[np.ndarray | None, np.ndarray | None]:
    """The ``(corrected_mask_bool, raw_segmented_image)`` a toggle needs, or ``(None, None)``."""
    tag = getattr(layer, "metadata", {}).get(OURS) or {}
    return tag.get("corrected_mask_bool"), tag.get("raw_segmented_image")


def _segmentation_cleanup_diff_labels(corrected: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Four labels: 0 background, 1 unchanged, 2 added by cleanup, 3 removed
    by cleanup. Pure, shape-agnostic, like ``_thick_thin_skeleton_labels``."""
    def labels_for(corrected, raw):
        corr = np.asarray(corrected, dtype=bool)
        raw_bool = np.asarray(raw, dtype=bool)
        labels = np.zeros(corr.shape, dtype=np.uint8)
        labels[corr & raw_bool] = 1
        labels[corr & ~raw_bool] = 2
        labels[~corr & raw_bool] = 3
        return labels

    if _on_disk(corrected, raw) and np.shape(corrected) == np.shape(raw):
        return _labels_by_slab(labels_for, corrected, raw)
    return labels_for(corrected, raw)


def _apply_segmentation_cleanup_display(
    layer, *, show: bool, default_colormap, default_contrast_limits,
    z_window: tuple[float, float, float | None] | None = None,
) -> None:
    """Recolour the IMAGE layer to show cleanup's changes, or revert.

    ``IMAGE`` is a napari Image layer, not a Labels layer like ``SKELETON``,
    so the thick/thin toggle's ``layer.colormap = {1: ..., ...}`` dict trick
    (valid only for Labels) does not apply. Reuses this codebase's only
    existing Image-layer recolour precedent instead --
    ``_image_options_for_napari``'s two-stop ``Colormap`` trick for
    ``mask_colour`` -- extended to four stops, with
    ``contrast_limits=(0, 3)`` so each integer label lands exactly on a
    control point with no interpolation.
    """
    corrected_bool, raw_mask = _segmentation_cleanup_state(layer)
    if show and corrected_bool is not None and raw_mask is not None:
        if z_window is not None:
            z_min, z_max, z_extent = z_window
            dz = _layer_voxel_size_z(layer)
            corrected_bool = np.asanyarray(
                clip_volume_to_z(corrected_bool, dz, z_min, z_max, z_extent=z_extent),
                dtype=bool,
            )
            raw_mask = np.asanyarray(
                clip_volume_to_z(raw_mask, dz, z_min, z_max, z_extent=z_extent),
                dtype=bool,
            )
        from napari.utils.colormaps import Colormap

        layer.data = _segmentation_cleanup_diff_labels(corrected_bool, raw_mask)
        layer.colormap = Colormap(
            [
                [0.0, 0.0, 0.0, 0.0],
                list(SEGMENTATION_CLEANUP_UNCHANGED_COLOUR),
                list(SEGMENTATION_CLEANUP_ADDED_COLOUR),
                list(SEGMENTATION_CLEANUP_REMOVED_COLOUR),
            ],
            name="haemolynx_segmentation_cleanup_diff",
        )
        layer.contrast_limits = (0.0, 3.0)
        return
    layer.colormap = default_colormap
    layer.contrast_limits = default_contrast_limits
    if corrected_bool is not None:
        data = _as_uint8_layer_data(corrected_bool)
        if z_window is not None:
            z_min, z_max, z_extent = z_window
            dz = _layer_voxel_size_z(layer)
            data = clip_volume_to_z(data, dz, z_min, z_max, z_extent=z_extent)
        layer.data = data


def _attach_segmentation_cleanup_toggle(viewer, layer) -> bool:
    """Put the segmentation-cleanup debug checkbox on the IMAGE layer's
    controls, once.

    Only offered when this run's mask actually carries a pre-cleanup
    version (at least one segmentation_cleanup_* step was on) -- see
    ``_store_segmentation_cleanup_metadata``. Mirrors
    ``_attach_thick_thin_skeleton_toggle`` including its defensive
    ``hasattr(layout, "addRow")`` guard: if napari's Image-layer controls
    widget does not expose an addRow-capable layout the way Labels controls
    do, the toggle is simply not offered rather than raising.
    """
    from qtpy.QtWidgets import QCheckBox

    if layer.__class__.__name__.lower() != "image":
        return False
    if getattr(layer, "name", None) != IMAGE:
        return False
    _corrected, raw_mask = _segmentation_cleanup_state(layer)
    available = raw_mask is not None
    controls = _layer_controls(viewer, layer)
    if controls is None:
        return False
    checkbox = getattr(controls, "_haemolynx_segmentation_cleanup", None)
    if checkbox is not None:
        checkbox.setEnabled(available)
        if not available:
            checkbox.setChecked(False)
        _apply_segmentation_cleanup_display(
            layer, show=checkbox.isChecked(),
            default_colormap=checkbox._haemolynx_default_colormap,
            default_contrast_limits=checkbox._haemolynx_default_contrast_limits,
            z_window=_current_z_depth_window(viewer),
        )
        return True
    layout = controls.layout()
    if not hasattr(layout, "addRow"):
        return False
    checkbox = QCheckBox("Segmentation cleanup: show changes (debug)")
    checkbox.setToolTip(
        "Colour mask voxels by whether pre-skeletonisation cleanup "
        "(reconnect / smooth / remove-small) changed them"
    )
    checkbox.setEnabled(available)
    # Captured once, before this ever recolours the layer.
    checkbox._haemolynx_default_colormap = layer.colormap
    checkbox._haemolynx_default_contrast_limits = layer.contrast_limits
    checkbox.toggled.connect(
        lambda checked, l=layer, cb=checkbox: _apply_segmentation_cleanup_display(
            l, show=checked, default_colormap=cb._haemolynx_default_colormap,
            default_contrast_limits=cb._haemolynx_default_contrast_limits,
            z_window=_current_z_depth_window(viewer),
        )
    )
    layout.addRow(checkbox)
    controls._haemolynx_segmentation_cleanup = checkbox
    return True


def _resync_segmentation_cleanup_after_z_change(viewer) -> None:
    """Mirror of ``_resync_thick_thin_skeleton_after_z_change`` for the
    IMAGE layer's segmentation-cleanup toggle."""
    if viewer is None:
        return
    z_window = _current_z_depth_window(viewer)
    for layer in viewer.layers:
        if layer.__class__.__name__.lower() != "image":
            continue
        if getattr(layer, "name", None) != IMAGE:
            continue
        controls = _layer_controls(viewer, layer)
        checkbox = (
            getattr(controls, "_haemolynx_segmentation_cleanup", None)
            if controls is not None else None
        )
        if checkbox is None or not checkbox.isChecked():
            continue
        _apply_segmentation_cleanup_display(
            layer, show=True, default_colormap=checkbox._haemolynx_default_colormap,
            default_contrast_limits=checkbox._haemolynx_default_contrast_limits,
            z_window=z_window,
        )


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

    edge_index = np.asarray(edge_index, dtype=int)

    # This grid point's columns per drawable edge, then spread over segments.
    per_edge: dict[str, np.ndarray] = {}
    flow_abs = np.asarray(sweep.flow_abs_at(*indices), dtype=float)
    per_edge["flow_abs"] = flow_abs[edge_index]
    from haemolynx.haemodynamics.resistance import flow_abs_log10_value

    per_edge["flow_abs_log10"] = np.asarray(
        [flow_abs_log10_value(v) for v in flow_abs], dtype=float
    )[edge_index]
    signed = sweep.flow_signed_at(*indices)
    if signed is not None:
        per_edge["flow_signed"] = np.asarray(signed, dtype=float)[edge_index]
    directions = tag.get("sweep_directions")
    if signed is not None and directions is not None:
        from haemolynx.gui.results import sweep_direction_columns

        per_edge.update(
            (name, np.asarray(values))
            for name, values in sweep_direction_columns(
                directions, np.asarray(signed, dtype=float)[edge_index]
            ).items()
        )
    drop = sweep.pressure_drop_at(*indices)
    if drop is not None:
        per_edge["pressure_drop"] = np.asarray(drop, dtype=float)[edge_index]

    def _with_grid_point(features: Mapping[str, Any], segment_owner) -> dict[str, Any]:
        segment_owner = np.asarray(segment_owner, dtype=int)
        updated = dict(features)
        for name, values in per_edge.items():
            if name == "flow_abs" or name in updated:
                updated[name] = values[segment_owner]
        mask_unsolved_flow_columns(updated)
        return updated

    layer.features = _with_grid_point(layer.features, owner)
    # The Z-depth filter redraws the layer from its full cache, so the cache
    # has to hold this grid point too -- otherwise the next depth change
    # quietly puts back the flows of whichever point was cached.
    cache = tag.get("z_filter_full")
    if isinstance(cache, dict) and cache.get("segment_owner") is not None:
        tag = dict(tag)
        tag["z_filter_full"] = {
            **cache,
            "features": _with_grid_point(cache["features"], cache["segment_owner"]),
        }
        metadata = dict(getattr(layer, "metadata", {}) or {})
        metadata[OURS] = tag
        layer.metadata = metadata

    colour_by = tag.get("colour_by") or "flow_abs"
    limits = tag.get("contrast_limits")
    if limits is None and colour_by == "flow_abs":
        limits = sweep.global_flow_abs_limits()
    _colour_layer(layer, colour_by, "continuous", (), limits)


def _apply_sweep_points(viewer, layer_set: str | None, row: int) -> None:
    """Move network *layer_set*'s sweep-following Points layers to grid *row*.

    A spacing or length sweep moves the pericytes themselves, so this swaps
    the points, not a column. The Z-depth filter's cache takes the grid
    point's whole set, and the layer shows the part inside the current
    window, as the filter would have drawn it.
    """
    for layer in list(viewer.layers):
        if not _is_ours(layer) or _layer_set_of(layer) != layer_set:
            continue
        tag = getattr(layer, "metadata", {}).get(OURS) or {}
        per_row = tag.get("sweep_points")
        if per_row is None or not 0 <= row < len(per_row):
            continue
        data, features = per_row[row]
        rule = _colouring_rule(layer)
        _store_z_filter_cache(layer, data, features)
        shown, shown_features = data, features
        window = _current_z_depth_window(viewer)
        if window is not None:
            shown, shown_features = filter_points_by_z(data, features, window[0], window[1])
        # Fewer points than before recreates the layer (see
        # _set_z_filtered_layer_data), so its colour controls go back on as
        # the Z-depth filter puts them back.
        layer = _set_z_filtered_layer_data(
            viewer, layer, "points", shown, dict(shown_features), None
        )
        try:
            _attach_colour_scale(viewer, layer)
        except Exception:  # noqa: BLE001 - a missing colour bar is survivable
            logger.debug("could not reattach colour controls to %s", layer.name, exc_info=True)
        if len(layer.data):
            cycle = tag.get("sweep_points_colour_cycle")
            if rule is not None and rule["column"] == "branch_order" and cycle:
                # The cycle covers every grid point's branch orders, so a
                # pericyte keeps its colour as the slider moves.
                _colour_layer(layer, "branch_order", "categorical", cycle)
            else:
                _reapply_colouring(layer, rule, features)
        try:
            _refresh_layer_controls(viewer, layer)
        except Exception:  # noqa: BLE001 - a stale readout is survivable
            logger.debug("could not refresh colour controls on %s", layer.name, exc_info=True)


def _sweep_dock_name(layer_name: str) -> str:
    return f"{layer_name} sweep"


def _remove_sweep_sliders(viewer, layer_name: str) -> None:
    """Take down the sweep sliders for *layer_name*, if there are any.

    Sliders outlive their layer otherwise: Clear, a run from an earlier tab,
    or a perturbation that failed or changed type left sliders driving a
    layer that was gone or no longer a sweep. They live in the view panel
    (see :func:`_attach_sweep_sliders`), or in a dock of their own when the
    viewer has no panel.
    """
    host = getattr(viewer, "_haemolynx_sweep_host", None)
    if host is not None:
        try:
            host.remove(layer_name)
        except Exception:  # noqa: BLE001
            logger.debug("could not remove sweep sliders for %s", layer_name, exc_info=True)
    dock_name = _sweep_dock_name(layer_name)
    try:
        window = viewer.window
        docks = getattr(window, "_wrapped_dock_widgets", None)
        if docks is None:  # napari < 0.6.2
            docks = getattr(window, "_dock_widgets", {})
        for dock in list(docks.values()):
            if getattr(dock, "objectName", lambda: "")() == dock_name or (
                hasattr(dock, "windowTitle") and dock.windowTitle() == dock_name
            ):
                window.remove_dock_widget(dock)
    except Exception:  # noqa: BLE001
        logger.debug("could not remove sweep dock %s", dock_name, exc_info=True)


def _attach_sweep_sliders(viewer, layer, spec) -> None:
    """One or two integer sliders for a sweep Vectors layer.

    They go in the view panel's Sweep box, which shows only the sliders of
    the perturbation chosen under "Showing"; a viewer with no panel gets them
    in a dock of their own.
    """
    # Drop previous sliders for this layer on re-run -- and keep them dropped
    # when this layer is no longer a sweep.
    _remove_sweep_sliders(viewer, spec.name)
    sweep = getattr(spec, "sweep", None)
    if sweep is None:
        return

    from magicgui.widgets import Container, Label, Slider

    dock_name = _sweep_dock_name(spec.name)

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

    layer_name = layer.name
    network = getattr(spec, "layer_set", None)

    def on_change(_event=None) -> None:
        indices = tuple(int(slider.value) for _name, slider, _vals, _lab in sliders)
        for (_name, slider, values, readout), index in zip(sliders, indices):
            if 0 <= index < len(values):
                readout.value = _format_sweep_value(_name, values[index])
        try:
            _apply_sweep_points(viewer, network, sweep.flat_index(*indices))
        except Exception:  # noqa: BLE001 - the flows below still move
            logger.debug("could not move %s's sweep points", network, exc_info=True)
        # By name, not the layer object this slider was built for: narrowing
        # the Z-depth filter replaces the layer with a new one, and a slider
        # still holding the old object moved flows on a layer no longer shown.
        current = viewer.layers[layer_name] if layer_name in viewer.layers else None
        if current is None:
            return
        _apply_sweep_index(current, indices)
        try:
            _refresh_layer_controls(viewer, current)
            _attach_colour_scale(viewer, current)
        except Exception:  # noqa: BLE001
            logger.debug("sweep slider refresh failed for %s", layer_name, exc_info=True)

    for _name, slider, _values, _readout in sliders:
        slider.changed.connect(on_change)

    rows = []
    for _name, slider, _values, readout in sliders:
        rows.append(slider)
        rows.append(readout)
    container = Container(widgets=rows, layout="vertical", labels=True)
    host = getattr(viewer, "_haemolynx_sweep_host", None)
    if host is not None:
        host.add(spec.name, getattr(spec, "layer_set", None), container)
        return
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
    "vectors": "vectors", "shapes": "shapes", "surface": "surface",
}


#: Identifiers rather than quantities: colouring by one shows nothing.
#: Hover metric copies (`flow`, `order`, `tortuosity`) ride on the vessels
#: Vectors layer for the tooltip table and must not appear in Colour by.
NOT_WORTH_COLOURING_BY = frozenset(
    {
        "u", "v", "key", "edge_index", "node_id", "tooltip", "branch_id",
        "flow", "order", "tortuosity",
        # Not a colouring of its own: under any flow-based colouring the
        # unsolved vessels are the grey ones.
        FLOW_SOLUTION,
        # What the tubes draw a bridge by, not a quantity.
        IS_ZERO_RESISTANCE,
    }
)

#: The columns the colour-by dropdown lists first, in this order.
_FLOW_COLOUR_COLUMN_ORDER = (
    "flow_abs",
    "flow_abs_log10",
    # Branch order beside flow rather than alphabetically among ~30 analysis
    # columns: its rank along the flow path and its signed generations from
    # the capillary bed colour with a map, range and colour bar like flow;
    # the label with a fixed palette in the same flow-path order.
    "branch_order_rank",
    "branch_order_signed",
    "branch_order",
    "flow_signed",
    "flow_dir_rgb",
    "flow_heading_deg",
    "flow_dir_z",
    "flow_dir_y",
    "flow_dir_x",
    "pressure_drop",
    "pressure_u",
    "pressure_v",
)

#: Colormaps offered on Vectors/Points layer controls, grouped for scanning.
#: Names are matplotlib/vispy maps ``ensure_colormap`` accepts; unknown ones
#: are dropped when the combo is filled so an older napari still starts.
COLORMAP_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Sequential",
        ("viridis", "plasma", "inferno", "magma", "cividis", "turbo", "gray"),
    ),
    (
        "Diverging",
        ("coolwarm", "RdBu", "seismic", "PiYG", "bwr"),
    ),
    (
        "Cyclic",
        ("hsv", "twilight", "twilight_shifted"),
    ),
    (
        "Other",
        (
            "jet",
            "rainbow",
            "hot",
            "cool",
            "spring",
            "summer",
            "autumn",
            "winter",
            "bone",
            "copper",
            "pink",
        ),
    ),
)
_COLORMAP_GROUP_HEADERS = frozenset(title for title, _names in COLORMAP_GROUPS)


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
    if column in FLOW_DIR_COLUMNS or column == FLOW_HEADING_COLUMN:
        _ensure_flow_dir_features(layer)
    if column == FLOW_DIR_RGB_COLUMN:
        return None
    try:
        values = np.asarray(layer.features[column], dtype=float)
    except (TypeError, ValueError):
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    if column == FLOW_HEADING_COLUMN:
        return _flow_heading_contrast_limits(values)
    if column in FLOW_DIR_COLUMNS:
        return _flow_dir_contrast_limits(values)
    if column == BRANCH_ORDER_SIGNED:
        # Centred on the capillary bed, so a diverging map puts the arterial
        # and venous sides on its two halves whatever their depths.
        reach = max(float(np.max(np.abs(finite))), 1.0)
        return -reach, reach
    low = float(np.percentile(finite, low_percentile))
    high = float(np.percentile(finite, high_percentile))
    if high <= low:
        high = low + abs(low) * 1e-6 + 1e-30
    return low, high


def _contrast_limits_attribute(layer) -> str:
    return f"{_colour_attribute(layer).replace('_color', '')}_contrast_limits"


def _colormap_attribute(layer) -> str:
    """``edge_colormap`` / ``face_colormap`` -- not ``edge_color_colormap``.

    Napari accepts setattr of a name it does not have: the value lands on a
    stray attribute and the real colormap stays viridis. Same trap as
    ``edge_color_contrast_limits`` (see ``_contrast_limits_attribute``).
    """
    return f"{_colour_attribute(layer).replace('_color', '')}_colormap"


def _colormap_name(layer) -> str | None:
    """The LUT currently on *layer*, or None if it has none -- named as the
    map a grey-NaN copy was made from."""
    cmap = getattr(layer, _colormap_attribute(layer), None)
    name = getattr(cmap, "name", None)
    return _base_colormap_name(str(name)) if name else None


def _colormap_usable(layer) -> bool:
    """Whether a 1D LUT applies: colormap mode, not direct RGB or a cycle."""
    if layer is None:
        return False
    if not hasattr(layer, _colormap_attribute(layer)):
        return False
    column = _active_column(layer)
    if column in {None, "", "none", FLOW_DIR_RGB_COLUMN}:
        return False
    if _is_text_column(layer, column):
        return False
    mode = getattr(layer, f"{_colour_attribute(layer)}_mode", None)
    return mode == "colormap"


def _apply_colormap(layer, name: str) -> bool:
    """Set the layer's LUT and re-map the active column so the canvas updates."""
    applied = False
    for attribute in _colour_attributes(layer):
        cmap_attr = f"{attribute.replace('_color', '')}_colormap"
        if not hasattr(layer, cmap_attr):
            continue
        try:
            setattr(layer, cmap_attr, _grey_nan_colormap(name))
        except (ValueError, TypeError, KeyError):
            logger.debug("could not set %s = %s", cmap_attr, name, exc_info=True)
            return False
        applied = True
    if not applied:
        return False
    column = _active_column(layer)
    if not column:
        return True
    try:
        for attribute in _colour_attributes(layer):
            setattr(layer, attribute, column)
    except (KeyError, TypeError, ValueError):
        logger.debug("could not remap colours after colormap change", exc_info=True)
    return True


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
            and self._column != FLOW_DIR_RGB_COLUMN
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
        try:
            self.bar.setPixmap(
                _colorbar_pixmap(_colormap_name(layer) or "viridis")
            )
        except Exception:  # noqa: BLE001 - a missing bar is survivable
            logger.debug("could not draw colour bar pixmap", exc_info=True)
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


def _choose_colour_by(viewer, layer, column: str) -> None:
    """Colour *layer* by *column* as a user's pick does, then let every
    control showing that choice catch up.

    The one path for both pickers -- "Colour by" on the layer's own controls
    and on the view panel -- so neither colours differently from the other,
    or is left naming a column that is no longer the one on screen.
    """
    text = _is_text_column(layer, column)
    cycle = colour_cycle_for(layer.features[column]) if text else ()
    _colour_layer(layer, column, "categorical" if text else "continuous", cycle)
    _refresh_layer_controls(viewer, layer)
    refresh_view_panel = getattr(viewer, "_haemolynx_refresh_colour_by", None)
    if refresh_view_panel is not None:
        refresh_view_panel()


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
        _choose_colour_by(self._viewer, layer, column)


def _known_colormap(name: str) -> bool:
    """Whether napari will accept *name* as a LUT."""
    try:
        from napari.utils.colormaps import ensure_colormap

        ensure_colormap(name)
        return True
    except (KeyError, ValueError, TypeError):
        return False


def colormap_choices() -> tuple[str, ...]:
    """Flat, de-duplicated list of maps the dropdown offers, in group order."""
    seen: set[str] = set()
    names: list[str] = []
    for _title, group in COLORMAP_GROUPS:
        for name in group:
            if name not in seen:
                seen.add(name)
                names.append(name)
    return tuple(names)


def _fill_colormap_combo(combo) -> None:
    """Group headers (disabled) then the maps napari actually knows."""
    seen: set[str] = set()
    for title, group in COLORMAP_GROUPS:
        available = [
            name for name in group
            if name not in seen and _known_colormap(name)
        ]
        if not available:
            continue
        seen.update(available)
        combo.addItem(title)
        header = combo.model().item(combo.count() - 1)
        if header is not None:
            header.setEnabled(False)
        for name in available:
            combo.addItem(name)


class _ColormapChooser:
    """LUT picker on the same layer-controls panel as Colour by.

    Direct RGB (``flow_dir_rgb``) and categorical colourings do not use a
    colormap, so the combo is hidden then. Changing Colour by reapplies that
    column's default map; the user can override afterwards from here.
    """

    def __init__(self, viewer, layer_name: str) -> None:
        from qtpy.QtWidgets import QComboBox

        self._viewer = viewer
        self._layer_name = layer_name
        self._connected = None
        self.label = None
        self.shown = False
        self.native = QComboBox()
        _fill_colormap_combo(self.native)
        self.native.currentTextChanged.connect(self._chosen)

    def _layer(self):
        layers = getattr(self._viewer, "layers", {}) if self._viewer else {}
        layer = layers[self._layer_name] if self._layer_name in layers else None
        return layer if layer is not None and _is_ours(layer) else None

    def follow_the_layer(self) -> None:
        """Keep the combo in step with colourings chosen anywhere."""
        layer = self._layer()
        if layer is not None and layer is not self._connected:
            events = getattr(layer, "events", None)
            attribute = _colour_attribute(layer)
            signal = getattr(events, attribute, None) if events else None
            if signal is not None:
                signal.connect(lambda *_a: self.refresh())
            self._connected = layer
        self.refresh()

    def _offered_maps(self) -> list[str]:
        return [
            self.native.itemText(i)
            for i in range(self.native.count())
            if self.native.itemText(i)
            and self.native.itemText(i) not in _COLORMAP_GROUP_HEADERS
        ]

    def refresh(self) -> None:
        """Select the layer's current map, or hide when a LUT does not apply."""
        layer = self._layer()
        usable = _colormap_usable(layer)
        self.shown = bool(usable)
        self.native.setEnabled(usable)
        self.native.setVisible(usable)
        if self.label is not None:
            self.label.setVisible(usable)
        if not usable or layer is None:
            return
        name = _colormap_name(layer)
        if name and name in self._offered_maps():
            with _blocked(self.native):
                self.native.setCurrentText(name)

    def _chosen(self, name: str) -> None:
        if not name or name in _COLORMAP_GROUP_HEADERS:
            return
        layer = self._layer()
        if layer is None or not _colormap_usable(layer):
            return
        _apply_colormap(layer, name)


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
    if _is_vessel_tubes_layer(layer):
        return False
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
        map_label = QLabel("Colour map:")
        maps = _ColormapChooser(viewer, layer.name)
        maps.label = map_label
        layout.addRow(map_label, maps.native)
        controls._haemolynx_colormap = maps
        maps.follow_the_layer()
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
        "_haemolynx_colormap",
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


def _shown_vessels_layer(viewer):
    """The vessels Vectors layer of the network "Showing" names, or None.

    Tubes or lines, this is the layer to colour: the tube Surface takes its
    colours from it (see :func:`_retint_vessel_tubes`), hidden or not.
    """
    from haemolynx.gui.layer_sets import layer_set

    layers = getattr(viewer, "layers", None) if viewer is not None else None
    if layers is None:
        return None
    name = layer_set(_layer_set_shown(viewer))["vessels"]
    # With the name a clash with a user's layer gives ours (see _add_or_update).
    for candidate in (name, f"{name} (HaemoLynx)"):
        if candidate in layers:
            layer = layers[candidate]
            if _is_ours(layer) and layer.__class__.__name__ == "Vectors":
                return layer
    return None


class _VesselColourMenu:
    """"Colour by" for the vessels on screen, in the floating view panel.

    The same choice as the vessels layer's own "Colour by" (see
    :class:`_FeatureChooser`), without having to select the layer first. For
    tubes that was the only way: the tube Surface takes its colours from the
    vessels Vectors layer, which is hidden while tubes are drawn. It acts on
    the network "Showing" names, drawn either way.

    A menu on a button rather than a QComboBox, for the reason the Showing
    menu is one: a combo's popup over the floating dock missed clicks.
    """

    def __init__(self, viewer, on_resize=None) -> None:
        from qtpy.QtWidgets import QFormLayout, QLabel, QMenu, QToolButton, QWidget

        from haemolynx.gui.chrome_tooltips import COLOUR_BY_TOOLTIP

        self._viewer = viewer
        self._on_resize = on_resize
        self._connected = None
        #: Recorded for tests: Qt reports children of an unshown window as
        #: invisible, so callers check this rather than isVisible().
        self.shown = False
        self.label = QLabel("Colour by")
        self.label.setObjectName("haemolynx_colour_by_label")
        self.button = QToolButton()
        self.button.setObjectName("haemolynx_colour_by")
        self.button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.menu = QMenu(self.button)
        self.menu.setObjectName("haemolynx_colour_by_menu")
        self.button.setMenu(self.menu)
        # A later stage adds columns (flow after the solve) without always
        # recolouring, so the list is read afresh each time it opens.
        self.menu.aboutToShow.connect(self.refresh)
        for widget in (self.label, self.button):
            widget.setToolTip(COLOUR_BY_TOOLTIP)
        self.row = QWidget()
        self.row.setObjectName("haemolynx_colour_by_row")
        form = QFormLayout(self.row)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow(self.label, self.button)
        self.row.setVisible(False)

    def refresh(self, *_args) -> None:
        """Offer what the shown network's vessels layer offers on its own
        controls, tick the one on screen, and hide the row while there is no
        network to colour."""
        layer = _shown_vessels_layer(self._viewer)
        self._follow(layer)
        columns = _colour_by_columns(layer) if layer is not None else []
        active = _active_column(layer) if layer is not None else None
        try:
            offered = [action.text() for action in self.menu.actions()]
            if offered != columns:
                # Only when the list changed: a pick recolours, the recolour
                # lands here, and clearing would delete the very action whose
                # `triggered` is still being delivered.
                self.menu.clear()
                for column in columns:
                    action = self.menu.addAction(column)
                    action.setCheckable(True)
                    action.triggered.connect(
                        lambda _checked=False, column=column: self.choose(column)
                    )
            for action in self.menu.actions():
                action.setChecked(action.text() == active)
            self.button.setText(active if active in columns else "none")
            was_shown = self.shown
            self.shown = bool(columns)
            self.row.setVisible(self.shown)
        except RuntimeError:
            # The dock's Qt widgets outlive the panel on teardown.
            logger.debug("colour-by menu is gone", exc_info=True)
            return
        if self.shown != was_shown and self._on_resize is not None:
            self._on_resize()

    def choose(self, column: str) -> None:
        """Colour the shown network's vessels by *column*, tubes and all."""
        layer = _shown_vessels_layer(self._viewer)
        if layer is None or column not in getattr(layer, "features", {}):
            self.refresh()
            return
        # Which refreshes this menu too, through the viewer's hook.
        _choose_colour_by(self._viewer, layer, column)

    def _follow(self, layer) -> None:
        """Hear about a colouring chosen anywhere else -- napari's own controls.

        A bound method, which napari connects once however often it is asked,
        so going back to a network already followed adds nothing.
        """
        if layer is None or layer is self._connected:
            return
        events = getattr(layer, "events", None)
        signal = getattr(events, "edge_color", None) if events else None
        if signal is not None:
            signal.connect(self.refresh)
        self._connected = layer


def _layer_features(layer):
    """Feature table on *layer*, never a DataFrame used as a boolean."""
    features = getattr(layer, "features", None)
    if features is None:
        return {}
    return features


def _is_branch_hover_layer(layer) -> bool:
    """Whether *layer* carries branch-tooltip features (vessels / flow / leftover)."""
    if not _is_ours(layer):
        return False
    if _is_vessel_tubes_layer(layer):
        return False
    if layer.name == BRANCH_HOVER or layer.name.startswith(f"{BRANCH_HOVER} "):
        return True
    features = _layer_features(layer)
    return "tooltip" in features and "branch_id" in features


def _branch_hover_available(layer) -> tuple[str, ...]:
    """Optional metrics this hover layer can offer right now."""
    from haemolynx.gui.branch_hover import available_metrics_from_features

    tag = getattr(layer, "metadata", {}).get(OURS) or {}
    stored = tag.get("branch_hover_available")
    if stored is not None:
        return tuple(stored)
    return available_metrics_from_features(_layer_features(layer))


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
    tag = dict(getattr(layer, "metadata", {}).get(OURS) or {})
    # The tag records which selection the tooltip column was composed for --
    # set here, and by _store_branch_hover_metadata for a stage's own column --
    # and the Z-depth filter's cache is kept in step with it below, so a
    # filtered or recreated layer's tooltips are already right. Recomposing
    # them anyway on every Z-depth change is what made the slider take tens
    # of seconds per move.
    recorded = tag.get("branch_hover_selected")
    if (
        recorded is not None
        and tuple(recorded) == tuple(chosen)
        and "tooltip" in getattr(layer, "features", {})
    ):
        return
    features = dict(layer.features)
    features["tooltip"] = tooltips_from_feature_table(features, chosen)
    layer.features = features
    tag["branch_hover_selected"] = chosen
    cache = tag.get("z_filter_full")
    if isinstance(cache, dict) and cache.get("features") is not None:
        full_feats = dict(cache["features"])
        if full_feats:
            full_feats["tooltip"] = tooltips_from_feature_table(full_feats, chosen)
            tag["z_filter_full"] = {**cache, "features": full_feats}
    metadata = dict(getattr(layer, "metadata", {}) or {})
    metadata[OURS] = tag
    layer.metadata = metadata


def _hover_max_distance(layer) -> float:
    """Pickup radius in data coordinates; at least the old circle diameter."""
    from haemolynx.gui.branch_hover import BRANCH_HOVER_MAX_DISTANCE

    width = getattr(layer, "edge_width", None)
    try:
        edge_width = float(np.asarray(width).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        edge_width = 0.0
    if not np.isfinite(edge_width) or edge_width < 0.0:
        edge_width = 0.0
    return max(BRANCH_HOVER_MAX_DISTANCE, edge_width * 2.0)


def _hover_feature_index(layer, event) -> int | None:
    """Index into *layer.features['tooltip']* for the cursor, or ``None``."""
    from haemolynx.gui.branch_hover import nearest_vector_index

    kind = layer.__class__.__name__.lower()
    if kind == "vectors":
        position = getattr(event, "position", None)
        dims = list(getattr(event, "dims_displayed", ()) or ())
        view_direction = getattr(event, "view_direction", None)
        if len(dims) != 3:
            view_direction = None
        return nearest_vector_index(
            position,
            getattr(layer, "data", ()),
            max_distance=_hover_max_distance(layer),
            view_direction=view_direction,
        )
    try:
        index = layer.get_value(
            event.position,
            view_direction=getattr(event, "view_direction", None),
            dims_displayed=list(getattr(event, "dims_displayed", ())),
            world=True,
        )
    except TypeError:
        index = layer.get_value(event.position, world=True)
    if isinstance(index, tuple):
        index = index[0]
    if index is None:
        return None
    try:
        return int(index)
    except (TypeError, ValueError):
        return None


def _show_branch_tooltip(layer, index: int) -> None:
    from qtpy.QtGui import QCursor
    from qtpy.QtWidgets import QToolTip

    tips = _layer_features(layer).get("tooltip")
    if tips is None:
        return
    try:
        text = str(tips[int(index)])
    except (IndexError, TypeError, ValueError):
        return
    if text:
        QToolTip.showText(QCursor.pos(), text)


def _branch_hover_mouse_move(layer, event) -> None:
    """Show the composed tooltip when the cursor is over this layer's polyline.

    A miss does not hide the tooltip: the viewer-level callback owns
    hide-on-empty so a selected nodes layer cannot swallow a hit on the
    vessels underneath.
    """
    index = _hover_feature_index(layer, event)
    if index is None:
        return
    _show_branch_tooltip(layer, index)


def _branch_hover_viewer_mouse_move(viewer, event) -> None:
    """Hover the drawn branch even when another HaemoLynx layer is selected."""
    from qtpy.QtWidgets import QToolTip

    layers = getattr(viewer, "layers", ())
    for layer in reversed(list(layers)):
        if not _is_branch_hover_layer(layer):
            continue
        hidden_source = (
            not getattr(layer, "visible", False)
            and _is_vessel_vectors_layer(layer)
            and _vessel_draw_mode == VESSEL_DRAW_TUBES
        )
        if not getattr(layer, "visible", False) and not hidden_source:
            continue
        index = _hover_feature_index(layer, event)
        if index is None:
            continue
        _show_branch_tooltip(layer, index)
        return
    QToolTip.hideText()


def _ensure_branch_hover_callback(layer, viewer=None) -> None:
    """Install polyline tooltip callbacks once per layer and once per viewer."""
    callbacks = getattr(layer, "mouse_move_callbacks", None)
    if callbacks is not None and _branch_hover_mouse_move not in callbacks:
        callbacks.append(_branch_hover_mouse_move)
    if viewer is None:
        viewer = getattr(layer, "viewer", None)
    if viewer is None:
        return
    viewer_callbacks = getattr(viewer, "mouse_move_callbacks", None)
    if viewer_callbacks is None:
        return
    if _branch_hover_viewer_mouse_move not in viewer_callbacks:
        viewer_callbacks.append(_branch_hover_viewer_mouse_move)


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
            "What to show when hovering a branch. "
            "The graph edge index (branchID) is always included."
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
        _ensure_branch_hover_callback(layer, self._viewer)
        self.native.setVisible(True)
        # Recorded for tests: Qt reports children of an unshown window as
        # invisible, so callers check this rather than isVisible().
        self.offered = current_keys
        self.selected = selected

    def _toggled(self, *_args) -> None:
        global _branch_hover_session_selected
        selected = tuple(
            key for key, box in self._boxes.items() if box.isChecked()
        )
        _branch_hover_session_selected = selected
        viewer = self._viewer
        if viewer is not None:
            for layer in getattr(viewer, "layers", ()):
                if _is_branch_hover_layer(layer):
                    _apply_branch_hover_selection(layer, selected)
        else:
            layer = self._layer()
            if layer is not None:
                _apply_branch_hover_selection(layer, selected)
        layer = self._layer()
        if layer is not None:
            self.selected = _branch_hover_selected_for(layer)


def _attach_branch_hover_controls(viewer, layer) -> bool:
    """Put the metrics checkboxes on the branch-hover layer's controls, once."""
    from qtpy.QtWidgets import QLabel

    if not _is_branch_hover_layer(layer):
        return False
    controls = _layer_controls(viewer, layer)
    if controls is None:
        _ensure_branch_hover_callback(layer, viewer)
        _apply_branch_hover_selection(layer, _branch_hover_selected_for(layer))
        return False
    if getattr(controls, "_haemolynx_branch_hover", None) is not None:
        controls._haemolynx_branch_hover.refresh()
        return True
    layout = controls.layout()
    if not hasattr(layout, "addRow"):
        _ensure_branch_hover_callback(layer, viewer)
        return False
    panel = _BranchHoverPanel(viewer, layer.name)
    layout.addRow(QLabel("hover info:"), panel.native)
    controls._haemolynx_branch_hover = panel
    panel.refresh()
    return True


def _process_pending_qt_events() -> None:
    """Let Qt/vispy finish any already-queued GL work before more layer
    teardown queues more.

    A no-op when there is no live ``QApplication`` (some test contexts);
    otherwise just drains whatever is already pending -- it does not block
    waiting for new events, so this is cheap enough to call often.
    """
    try:
        from qtpy.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            app.processEvents()
    except Exception:  # noqa: BLE001 - never let this stop a clear
        logger.debug("could not process pending Qt events", exc_info=True)


def _clear_our_layers(viewer) -> int:
    """Remove every layer this plugin added. Leaves the user's alone.

    Processes pending Qt events before starting, and after every removal.
    ``viewer.layers.remove`` tears down that layer's GL resources
    synchronously (napari's vispy canvas -> glDeleteProgram and friends),
    but vispy's own GLIR command queue can still hold earlier GL work
    queued by recent user interaction (a camera move, a colormap change, a
    visibility toggle) that has not been flushed yet. Starting a
    synchronous teardown on top of that queue crashed the whole process
    with a Windows access violation deep inside vispy's own cleanup --
    below Python, so there is no exception to catch. Letting the queue
    drain before, and between, each removal is the standard mitigation for
    this class of vispy/GL race.
    """
    return _remove_our_layers(viewer, keep=frozenset())


def _layer_names_redrawn_by(groups) -> frozenset[str]:
    """Every layer name replaying *groups* puts back, tube Surfaces included.

    With the name a clash with a user's layer would give ours (see
    :func:`_add_or_update`), and the tube mesh drawn for each vessels layer
    (see :func:`_sync_vessel_tubes`).
    """
    names = {spec.name for group in groups for spec in getattr(group, "layers", ())}
    names |= {vessel_tubes_layer_name(name) for name in list(names)}
    names |= {f"{name} (HaemoLynx)" for name in list(names)}
    return frozenset(names)


def _remove_our_layers(viewer, keep: frozenset[str]) -> int:
    """Remove this plugin's layers except those named in *keep*.

    See :func:`_clear_our_layers` for why the removals are paced.
    """
    ours = [
        layer for layer in list(viewer.layers)
        if _is_ours(layer) and layer.name not in keep
    ]
    if not ours:
        return 0
    _process_pending_qt_events()
    for layer in ours:
        name = layer.name
        viewer.layers.remove(layer)
        _process_pending_qt_events()
        _remove_sweep_sliders(viewer, name)
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
    state=None, log=None, checkpoints=None, after_layers=None,
    start_from=None, resume=None, restored_stages=(), stop_after=None,
    on_paused=None):
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
    the panel keeps across a run so a tab can re-run from an earlier stage
    without rebuilding topology. Optional: without one, the run is unchanged.

    *start_from* / *resume* are forwarded to
    :func:`~haemolynx.pipeline.run_pipeline_stages` so "Run from this stage"
    can skip earlier work.

    *stop_after* is forwarded too: a run that stops there on purpose is
    paused, not finished -- the bars stay where it got to, *state* records
    where it paused, and *on_paused* (if given) is called, for the panel to
    bring up its Post processing tab.
    """
    from napari.qt.threading import thread_worker

    run_state = state if state is not None else RunState()
    cancel_flag = {"cancelled": False}
    bridge = _progress_bridge()
    show_layers = viewer is not None and results is not None
    #: The capture, while there is one. A namespace rather than a name, because
    #: it is set below `stopped`, which is what releases it.
    capture = SimpleNamespace(attachment=None)

    def still_ours() -> bool:
        return run_state.cancel_flag is cancel_flag

    def progressed(event: ProgressEvent) -> None:
        # A cancelled run's last few events are already on their way here. The
        # bars have been put back to nothing; moving them again would show a
        # run that is no longer going.
        if cancel_flag["cancelled"]:
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
        if cancel_flag["cancelled"]:
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
        between graph building's thirteen topology steps, so a run stops with
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
        run_state.check(cancel_flag)
        log_progress(event)
        bridge.event.emit(event)

    restored = frozenset(restored_stages)

    def produced(stage: str, output) -> None:
        """Build this stage's layers here, on the run's thread.

        A stage in *restored_stages* was restored from its checkpoint before a
        mid-pipeline run, and only passed through here: its outputs are the
        resumed ones -- graph building's is the previous tab's *finished*
        graph -- so it is neither recorded nor redrawn, or every earlier tab
        would take on a later tab's state. Its layers are not built either --
        they are on screen, replayed -- only the results' own bookkeeping
        follows it (see ``ResultLayers.stage_restored``).

        Eagerly, because every stage after `build_network` writes onto the same
        graph: convert later and the viewer shows a later stage's numbers under
        this stage's name. And guarded, because a fault in drawing a run must
        never end it -- an eight-hour whole-brain run least of all.

        The cancellation check is outside that guard, deliberately: stopping is
        the one thing that must get past it.
        """
        run_state.check(cancel_flag)
        if stage in restored:
            try:
                results.stage_restored(stage, output)
            except Exception:  # noqa: BLE001 - reported, never raised at the run
                logger.exception("could not follow restored stage %s", stage)
            return
        try:
            group = results.stage_finished(stage, output)
        except Exception:  # noqa: BLE001 - reported, never raised at the run
            logger.exception("could not build layers for stage %s", stage)
            return
        if cancel_flag["cancelled"]:
            return
        if checkpoints is not None:
            try:
                checkpoints.record(stage, group, results, settings=settings, output=output)
            except Exception:  # noqa: BLE001 - a bad snapshot must not end the run
                logger.exception("could not record checkpoint for stage %s", stage)
        if cancel_flag["cancelled"]:
            return
        bridge.layers.emit(group)

    @thread_worker
    def run():
        # `bridge` is captured here, which is also what keeps it alive for as
        # long as the run that emits through it.
        extra = {}
        if start_from is not None or resume is not None:
            extra["start_from"] = start_from
            extra["resume"] = resume
        if stop_after is not None:
            extra["stop_after"] = stop_after
        return run_pipeline_stages(
            settings,
            schema,
            progress=watched,
            on_stage_output=produced if show_layers else None,
            **extra,
        )

    def finished(graph) -> None:
        if not still_ours():
            return
        run_state.stopped()
        button.enabled = True
        if cancel_flag["cancelled"]:
            # It got to the end between being asked to stop and reaching the
            # next checkpoint. Saying "finished" beside an empty viewer, or
            # "cancelled" about a run that completed, would both be wrong.
            report.value = FINISHED_FIRST
            return
        if stop_after is not None and graph is not None:
            run_state.paused_after = stop_after
            if bars is not None:
                pause = getattr(bars, "pause", None)
                (pause if callable(pause) else bars.finish)(paused_bar_text(stop_after))
            report.value = paused_message(
                stop_after, graph.number_of_nodes(), graph.number_of_edges()
            )
            if on_paused is not None:
                on_paused()
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
        if not still_ours():
            if isinstance(error, RunCancelled):
                return
            logger.debug("stale worker failed after Clear", exc_info=error)
            return
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
        if not still_ours():
            return
        if not run_state.running:
            return
        run_state.stopped()
        button.enabled = True
        if cancel_flag["cancelled"]:
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
    run_state.start(worker=worker, results=results, cancel_flag=cancel_flag)
    if checkpoints is not None:
        checkpoints.unfreeze()
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


_OPTIMISATION_PROGRESS_BRIDGE_CLASS = None


def _optimisation_progress_bridge():
    """Same rationale as `_progress_bridge`: a QObject whose signal carries an
    `OptimisationEvent` from the optimiser's worker thread to the GUI thread.

    A separate bridge class from `_progress_bridge`'s, so the two event types
    (a pipeline run's `ProgressEvent`, the optimiser's `OptimisationEvent`)
    never share a Signal.
    """
    global _OPTIMISATION_PROGRESS_BRIDGE_CLASS
    if _OPTIMISATION_PROGRESS_BRIDGE_CLASS is None:
        from qtpy.QtCore import QObject, Signal

        class OptimisationProgressBridge(QObject):
            event = Signal(object)

        _OPTIMISATION_PROGRESS_BRIDGE_CLASS = OptimisationProgressBridge
    return _OPTIMISATION_PROGRESS_BRIDGE_CLASS()


def _names_with_prerequisite_closure(schema: Schema, names: Sequence[str]) -> tuple[str, ...]:
    """*names* plus every setting they (transitively) ``requires``.

    ``Schema.__init__`` validates each setting's ``requires`` targets against
    its own subset, not the parent schema -- building a documentation-only
    subset (like the optimiser's generated config schema) straight from a
    fixed list of "quality knobs" raises ``ConfigError`` the moment one of
    them, or one of *its* prerequisites, needs a stage toggle or another
    setting that fixed list never named. Walking the whole chain here means
    a new setting or a new layer of prerequisites never needs a matching
    edit at the call site.
    """
    closure: set[str] = set()
    stack = list(names)
    while stack:
        name = stack.pop()
        if name in closure:
            continue
        closure.add(name)
        for prerequisite in schema[name].requires:
            # "name=value" names a choice setting to check the value of, not
            # a nested prerequisite of its own -- the base name is what needs
            # to be in the closure.
            stack.append(prerequisite_name(prerequisite))
    return tuple(setting.name for setting in schema if setting.name in closure)


def _load_raw_reference_image_or_none(
    local_settings: dict[str, Any], *, expected_shape: tuple[int, ...]
) -> tuple["np.ndarray | None", "str | None"]:
    """Load the optional raw reference image from ``fwhm_raw_tiff_path``.

    Shared by "Check segmented image" and "Optimise settings" -- one setting
    feeds both (see the "Raw data file" row's own docstring) -- so a fix to
    how that load degrades only has to be made once, and both callers agree
    on what "the raw file didn't work out" means. Never raises: a missing
    setting, an unreadable file, or a raw image whose shape does not match
    *expected_shape* (the segmented mask's own shape) all degrade to
    ``(None, reason)`` rather than failing the caller's own real work --
    including the shape check itself, which every actual consumer of the
    returned image (`compare_segmentation_to_raw_image`, `_Search.__init__`)
    would otherwise raise on directly, one of them outside any try/except
    the caller has around this load.
    """
    from haemolynx.haemodynamics.automated import load_single_channel_tiff_volume
    from haemolynx.io import resolve_image_path_with_optional_zip

    raw_path = local_settings.get("fwhm_raw_tiff_path")
    if not raw_path:
        return None, None
    try:
        raw_path_resolved = resolve_image_path_with_optional_zip(Path(raw_path))
        channel = local_settings.get("fwhm_raw_channel")
        raw_image = load_single_channel_tiff_volume(
            raw_path_resolved,
            axis_order=local_settings["image_axis_order"],
            channel=None if channel is None else int(channel),
        )
        if raw_image.shape != expected_shape:
            raise ValueError(
                "raw_image shape does not match the segmented mask shape: "
                f"{raw_image.shape} != {expected_shape}"
            )
        return raw_image, None
    except Exception as error:  # noqa: BLE001 - degrade, never fail the caller
        return None, f"{type(error).__name__}: {error}"


def _cancellable_progress_bridge(run_state: "RunState", bars: "OptimiseProgressBars"):
    """The cancel-flag/progress-bridge plumbing every "Optimise ..."
    background run needs, shared by `_run_optimisation_in_background` and
    `_run_fwhm_optimisation_in_background` (previously copy-pasted between
    them verbatim).

    Returns ``(cancel_flag, still_ours, watched)`` for the caller's own
    `run()`/`finished()`/`failed()` closures: *still_ours* is how they check
    a later "Clear layers and state" press did not already claim
    *run_state* for something else, and *watched* is what the worker's
    `progress=` callback should be -- it runs on the worker's own thread,
    raises ``RunCancelled`` if this run was asked to stop, and forwards the
    event across the thread boundary via the Qt signal bridge so *bars* can
    draw it on the GUI thread.
    """
    bridge = _optimisation_progress_bridge()
    cancel_flag: dict[str, bool] = {"cancelled": False}

    def still_ours() -> bool:
        return run_state.cancel_flag is cancel_flag

    def progressed(event: OptimisationEvent) -> None:
        if cancel_flag["cancelled"]:
            return
        bars.show_event(event)

    bridge.event.connect(progressed)

    def watched(event: OptimisationEvent) -> None:
        run_state.check(cancel_flag)
        bridge.event.emit(event)

    return cancel_flag, still_ours, watched


def _start_optimisation_worker(
    run_thread_worker,
    *,
    run_state: "RunState",
    cancel_flag: dict,
    still_ours,
    button,
    bars: "OptimiseProgressBars",
    report,
    finished,
    failed,
    starting_message: str,
):
    """Wire up and start a `thread_worker`-wrapped optimisation run with the
    button/progress-bar/cancellation bookkeeping `_run_optimisation_in_background`
    and `_run_fwhm_optimisation_in_background` both need identically --
    including `stopped()`, whose body never differed between the two.
    """
    def stopped() -> None:
        if not still_ours():
            return
        if not run_state.running:
            return
        run_state.stopped()
        button.enabled = True
        if cancel_flag["cancelled"]:
            report.value = FINISHED_FIRST

    worker = run_thread_worker(_connect={"errored": failed}, _start_thread=False)
    worker.returned.connect(finished)
    worker.finished.connect(stopped)
    run_state.start(worker=worker, cancel_flag=cancel_flag)
    button.enabled = False
    bars.start()
    report.value = starting_message
    worker.start()
    return worker


def _run_optimisation_in_background(
    settings: dict[str, Any],
    schema: Schema,
    rows: dict[str, Any],
    report,
    button,
    bars: "OptimiseProgressBars",
    *,
    apply_prerequisites,
    run_state: RunState,
    downsample_factor: "int | None" = None,
    groups: "tuple[str, ...] | None" = None,
    max_passes: int = 1,
    propose,
):
    """Run the settings optimiser off the GUI thread, reporting progress as it goes.

    The run writes nothing into *rows* itself: when it finishes it hands
    *propose* the result and a function writing it into *rows*, and the panel
    shows the proposal for the user to Apply (which calls that function) or
    Discard.

    Structurally parallel to `_run_in_background`, but not a reuse of it: that
    one is tightly coupled to `run_pipeline_stages` / `ResultLayers` /
    checkpoints, none of which apply to a search that never produces a graph
    worth drawing in the viewer. Shares *run_state* with the pipeline Run
    button purely for mutual exclusion and cooperative cancellation --
    "Optimise settings" and "Run pipeline" must never run concurrently against
    the same `rows`.
    """
    from napari.qt.threading import thread_worker

    cancel_flag, still_ours, watched = _cancellable_progress_bridge(run_state, bars)

    @thread_worker
    def run():
        local_settings = dict(settings)
        inputs = segment(local_settings)
        image, metadata_voxel_size, voxel_meta_status = load_volume_for_skeletonise(
            local_settings, inputs.input_format
        )
        voxel_size_xyz, _source = resolve_voxel_size_xyz(
            metadata_voxel_size_xyz=metadata_voxel_size,
            metadata_status=voxel_meta_status,
            voxel_size_override_xyz=local_settings["voxel_size_override_xyz"],
            voxel_size_policy=local_settings["voxel_size_policy"],
        )
        raw_mask = _to_binary_volume_for_skeletonization(image)
        starting_values = {name: local_settings[name] for name in OPTIMISE_SETTING_NAMES}

        # Optional: same shared fwhm_raw_tiff_path the "Raw data file" row
        # feeds "Check segmented image" with -- when present, guards the
        # segmentation_cleanup group against inventing foreground the raw
        # signal does not support. Never fatal (see
        # `_load_raw_reference_image_or_none`): a missing/unreadable/
        # mismatched-shape raw file just means the search runs without this
        # extra guard, exactly like leaving the field empty.
        raw_image, raw_image_error = _load_raw_reference_image_or_none(
            local_settings, expected_shape=raw_mask.shape
        )

        result = optimise_skeleton_and_graph_settings(
            raw_mask,
            voxel_size_xyz=voxel_size_xyz,
            starting_values=starting_values,
            progress=watched,
            downsample_factor=downsample_factor,
            groups=groups,
            raw_image=raw_image,
            max_passes=max_passes,
        )
        return result, local_settings["input_path"], raw_image_error

    def finished(payload) -> None:
        if not still_ours():
            return
        run_state.stopped()
        button.enabled = True
        if cancel_flag["cancelled"]:
            report.value = FINISHED_FIRST
            return
        result, input_path, raw_image_error = payload

        def apply_result() -> None:
            for name, value in result.settings.items():
                if name in rows:
                    rows[name].value = display_value_for(schema[name], value)
            apply_prerequisites()

        out_path = Path(input_path).parent / config_filename(input_path)
        # OPTIMISE_SETTING_NAMES are quality knobs only -- several of them
        # (and "input_path" itself) `requires` a stage toggle or upstream
        # setting Schema validates against its own subset, not the parent
        # schema, so the documented config schema needs the whole closure,
        # not just the names the optimiser actually tunes. Every added
        # prerequisite defaults to what a config meant to be loaded and run
        # implies (skeletonisation/graph-building on, ilastik off), so
        # leaving them out of `values` below and letting `dump_config` fill
        # the schema default is correct.
        documented_schema = Schema(
            list(
                schema.subset(
                    _names_with_prerequisite_closure(
                        schema, (*OPTIMISE_SETTING_NAMES, "input_path")
                    )
                )
            ),
            title="HaemoLynx optimised settings",
            description=build_report_text(result),
        )
        try:
            dump_config(
                out_path, documented_schema, values={**result.settings, "input_path": input_path}
            )
            wrote_note = f" Wrote {out_path}."
        except Exception:  # noqa: BLE001 - the proposal stands without its file
            logger.exception("could not write optimised config to %s", out_path)
            wrote_note = f" Could not write {out_path} (see log)."
        bars.finish("Optimised")
        raw_note = (
            f" Raw-image cross-check skipped: {raw_image_error}" if raw_image_error else ""
        )
        report.value = propose(result, apply_result) + wrote_note + raw_note

    def failed(error: Exception) -> None:
        if not still_ours():
            if isinstance(error, RunCancelled):
                return
            logger.debug("stale optimisation worker failed after Clear", exc_info=error)
            return
        run_state.stopped()
        button.enabled = True
        if isinstance(error, RunCancelled):
            report.value = CANCELLED
            return
        bars.fail(f"Failed: {type(error).__name__}")
        report.value = f"{type(error).__name__}: {error}"
        logger.exception("settings optimisation failed", exc_info=error)
        raise error

    return _start_optimisation_worker(
        run,
        run_state=run_state,
        cancel_flag=cancel_flag,
        still_ours=still_ours,
        button=button,
        bars=bars,
        report=report,
        finished=finished,
        failed=failed,
        starting_message="Optimising settings...",
    )


#: Settings outside the ``fwhm_`` group the FWHM optimiser reads and never
#: changes: the image PSF its trials hold the blur at, and the FWHM/EDT ratio a
#: width is set aside at.
_FWHM_OPTIMISER_READS = (
    "raw_section_psf_sigma_xy_um",
    "raw_section_psf_sigma_z_um",
    "edt_fwhm_disagreement_warn_ratio",
)


def _run_fwhm_optimisation_in_background(
    graph: "nx.MultiGraph",
    voxel_size_zyx: tuple[float, float, float],
    settings: dict[str, Any],
    schema: Schema,
    rows: dict[str, Any],
    report,
    button,
    bars: "OptimiseProgressBars",
    *,
    apply_prerequisites,
    run_state: RunState,
    groups: "tuple[str, ...] | None" = None,
    vessel_mask: "np.ndarray | None" = None,
    sample_edge_count: "int | None" = None,
    max_passes: int = 1,
    propose,
):
    """Run the FWHM settings optimiser off the GUI thread, reporting progress
    as it goes -- the Diameters tab's own analogue of
    `_run_optimisation_in_background`, over
    `haemolynx.optimisation.optimise_fwhm_settings` instead of the
    skeleton/graph search, against the panel's own current in-memory graph
    rather than a mask reloaded from disk. Shares *run_state* with "Run
    pipeline" and "Optimise settings" purely for mutual exclusion and
    cooperative cancellation -- only one of the three may run at a time
    against the same `rows`. *vessel_mask*, the run's segmented image, lets
    every trial be checked against decoys and the mask's own widths. Like
    `_run_optimisation_in_background`, it proposes rather than applies: see
    *propose* there.
    """
    from napari.qt.threading import thread_worker

    cancel_flag, still_ours, watched = _cancellable_progress_bridge(run_state, bars)

    @thread_worker
    def run():
        local_settings = dict(settings)
        starting_values = {
            name: value
            for name, value in local_settings.items()
            if name.startswith("fwhm_") or name in _FWHM_OPTIMISER_READS
        }
        raw_path = local_settings["fwhm_raw_tiff_path"]
        result = optimise_fwhm_settings(
            graph,
            raw_tiff_path=raw_path,
            voxel_size_zyx=voxel_size_zyx,
            starting_values=starting_values,
            progress=watched,
            groups=groups,
            axis_order=local_settings["image_axis_order"],
            raw_channel=local_settings.get("fwhm_raw_channel"),
            vessel_mask=vessel_mask,
            sample_edge_count=sample_edge_count,
            max_passes=max_passes,
        )
        return result, raw_path

    def finished(payload) -> None:
        if not still_ours():
            return
        run_state.stopped()
        button.enabled = True
        if cancel_flag["cancelled"]:
            report.value = FINISHED_FIRST
            return
        result, raw_path = payload

        def apply_result() -> None:
            for name, value in result.settings.items():
                if name in rows:
                    rows[name].value = display_value_for(schema[name], value)
            apply_prerequisites()

        out_path = Path(raw_path).parent / config_filename(raw_path)
        # Same reasoning as `_run_optimisation_in_background`'s own
        # `documented_schema`: FWHM_SETTING_NAMES are quality knobs only, so
        # the closure pulls in their own prerequisites (use_fwhm_edge_diameters,
        # fwhm_raw_tiff_path, ...) for a config that is self-consistent on
        # its own rather than one that silently depends on values not shown.
        documented_schema = Schema(
            list(schema.subset(_names_with_prerequisite_closure(schema, FWHM_SETTING_NAMES))),
            title="HaemoLynx optimised FWHM settings",
            description=build_report_text(
                result,
                group_names=FWHM_GROUP_NAMES,
                heading="HaemoLynx FWHM settings, optimised on a sample of the network's "
                "vessels in the raw image:",
            ),
        )
        try:
            dump_config(out_path, documented_schema, values=result.settings)
            wrote_note = f" Wrote {out_path}."
        except Exception:  # noqa: BLE001 - the proposal stands without its file
            logger.exception("could not write optimised FWHM config to %s", out_path)
            wrote_note = f" Could not write {out_path} (see log)."
        bars.finish("Optimised")
        report.value = propose(result, apply_result) + wrote_note

    def failed(error: Exception) -> None:
        if not still_ours():
            if isinstance(error, RunCancelled):
                return
            logger.debug("stale FWHM optimisation worker failed after Clear", exc_info=error)
            return
        run_state.stopped()
        button.enabled = True
        if isinstance(error, RunCancelled):
            report.value = CANCELLED
            return
        bars.fail(f"Failed: {type(error).__name__}")
        report.value = f"{type(error).__name__}: {error}"
        logger.exception("FWHM settings optimisation failed", exc_info=error)
        raise error

    return _start_optimisation_worker(
        run,
        run_state=run_state,
        cancel_flag=cancel_flag,
        still_ours=still_ours,
        button=button,
        bars=bars,
        report=report,
        finished=finished,
        failed=failed,
        starting_message="Optimising FWHM settings...",
    )


def _run_segmentation_quality_check_in_background(
    settings: dict[str, Any],
    report,
    button,
    *,
    run_state: RunState,
):
    """Score the segmented input image off the GUI thread, printing the
    breakdown to *report* when done.

    Shares *run_state* with "Run pipeline" and "Optimise settings" purely
    for mutual exclusion -- a large volume's EDT/connected-components/
    smoothing pass is real work, so this must never race the other two
    against the same input. No progress bars: unlike the optimiser's dozens
    of pipeline sub-stage runs, this is one pass over the mask, reported as
    a single before/after message like "Run checks" already is.
    """
    from napari.qt.threading import thread_worker

    from haemolynx.io import voxel_size_zyx_from_xyz
    from haemolynx.preprocessing import (
        clean_segmented_mask_for_skeletonisation,
        compare_segmentation_to_raw_image,
        format_segmentation_quality_report,
        format_segmentation_raw_comparison_report,
        score_segmented_mask,
    )

    cancel_flag = {"cancelled": False}

    def still_ours() -> bool:
        return run_state.cancel_flag is cancel_flag

    @thread_worker
    def run():
        local_settings = dict(settings)
        inputs = segment(local_settings)
        image, metadata_voxel_size, voxel_meta_status = load_volume_for_skeletonise(
            local_settings, inputs.input_format
        )
        voxel_size_xyz, _source = resolve_voxel_size_xyz(
            metadata_voxel_size_xyz=metadata_voxel_size,
            metadata_status=voxel_meta_status,
            voxel_size_override_xyz=local_settings["voxel_size_override_xyz"],
            voxel_size_policy=local_settings["voxel_size_policy"],
        )
        mask = _to_binary_volume_for_skeletonization(image)
        voxel_size_zyx = voxel_size_zyx_from_xyz(tuple(float(v) for v in voxel_size_xyz))
        score = score_segmented_mask(
            mask,
            voxel_size_zyx=voxel_size_zyx,
            target_voxels_across_radius=local_settings.get("min_voxels_across_vessel_radius"),
            boundary_patch_allowance=local_settings.get("expected_boundary_vessel_count"),
        )

        # Also score the mask after the run's *currently configured*
        # segmentation_cleanup_* settings, so the report can show whether a
        # good number reflects the source data or just this pipeline's own
        # cleanup patching over it -- see segmentation_quality's own module
        # docstring for why that distinction matters. `raw_segmented_image`
        # is `None` (nothing to compare) when every cleanup step is off, or
        # when the segmentation_cleanup master switch itself is off -- a
        # real run's skeletonise() skips cleanup entirely in that case (see
        # pipeline.schema's segmentation_cleanup), and this preview must
        # show the same run the pipeline will actually do.
        if local_settings["segmentation_cleanup"]:
            cleanup_kwargs = prefixed_arguments(
                local_settings, "segmentation_cleanup_",
                parameters_of(clean_segmented_mask_for_skeletonisation),
            )
            cleaned_mask, raw_segmented_image = clean_segmented_mask_for_skeletonisation(
                mask, voxel_size_zyx=voxel_size_zyx, **cleanup_kwargs
            )
        else:
            cleaned_mask, raw_segmented_image = mask, None
        after_cleanup = None
        if raw_segmented_image is not None:
            after_cleanup = score_segmented_mask(
                cleaned_mask,
                voxel_size_zyx=voxel_size_zyx,
                target_voxels_across_radius=local_settings.get("min_voxels_across_vessel_radius"),
                boundary_patch_allowance=local_settings.get("expected_boundary_vessel_count"),
            )

        # Optional: cross-check against the raw (unsegmented) image, when
        # one is configured -- shares fwhm_raw_tiff_path with FWHM diameter
        # measurement (see the "Raw data file" row next to this button).
        # Never fatal: a missing/unreadable/mismatched-shape raw file
        # degrades to a report note, not a failed check -- the mask-only
        # score above is still useful on its own.
        raw_comparison = None
        raw_image, raw_comparison_error = _load_raw_reference_image_or_none(
            local_settings, expected_shape=mask.shape
        )
        if raw_image is not None:
            try:
                raw_comparison = compare_segmentation_to_raw_image(
                    mask, raw_image, voxel_size_zyx=voxel_size_zyx
                )
            except Exception as error:  # noqa: BLE001 - degrade to a report note
                raw_comparison_error = f"{type(error).__name__}: {error}"

        return score, after_cleanup, raw_comparison, raw_comparison_error

    def finished(payload) -> None:
        if not still_ours():
            return
        run_state.stopped()
        button.enabled = True
        if cancel_flag["cancelled"]:
            report.value = FINISHED_FIRST
            return
        score, after_cleanup, raw_comparison, raw_comparison_error = payload
        text = format_segmentation_quality_report(score, after_cleanup=after_cleanup)
        if raw_comparison is not None:
            text += "\n\n" + format_segmentation_raw_comparison_report(raw_comparison)
        elif raw_comparison_error is not None:
            text += f"\n\nRaw-image cross-check skipped: {raw_comparison_error}"
        report.value = text

    def failed(error: Exception) -> None:
        if not still_ours():
            if isinstance(error, RunCancelled):
                return
            logger.debug("stale segmented-image check worker failed after Clear", exc_info=error)
            return
        run_state.stopped()
        button.enabled = True
        if isinstance(error, RunCancelled):
            report.value = CANCELLED
            return
        report.value = f"{type(error).__name__}: {error}"
        logger.exception("segmented image quality check failed", exc_info=error)
        raise error

    def stopped() -> None:
        if not still_ours():
            return
        if not run_state.running:
            return
        run_state.stopped()
        button.enabled = True
        if cancel_flag["cancelled"]:
            report.value = FINISHED_FIRST

    worker = run(_connect={"errored": failed}, _start_thread=False)
    worker.returned.connect(finished)
    worker.finished.connect(stopped)
    run_state.start(worker=worker, cancel_flag=cancel_flag)
    button.enabled = False
    report.value = "Checking segmented image..."
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


def _box_tool_widgets(role: str) -> dict[str, Any]:
    """One role page's box tools, as magicgui widgets in the tab's own style.

    The size is three spin boxes on one labelled row and the six moves one row
    of buttons, each button named for its direction; the node list is a Qt
    table in a container of its own, reachable as ``.table`` (and its line of
    explanation as ``.note``).
    """
    from magicgui.widgets import ComboBox, Container, FloatSpinBox, Label, PushButton
    from qtpy.QtWidgets import QAbstractItemView, QHeaderView, QTableWidget

    from haemolynx.gui.boundary_picking import (
        BOX_NODE_COLUMNS,
        DEFAULT_BOX_SIZE_UM,
        MOVE_DIRECTIONS,
        role_title,
    )

    size = Container(
        widgets=[
            FloatSpinBox(value=v, min=0.5, max=5000.0, step=1.0, label=axis, name=f"size_{axis}")
            for axis, v in zip("zyx", DEFAULT_BOX_SIZE_UM)
        ],
        layout="horizontal", labels=True, label="Box size (um)",
    )
    arrows = {"left": "\u25c0 Left", "right": "Right \u25b6", "up": "\u25b2 Up",
              "down": "\u25bc Down", "back": "Back", "forward": "Forward"}
    move = Container(
        widgets=[PushButton(text=arrows[d], name=d) for d in MOVE_DIRECTIONS],
        layout="horizontal", labels=False, label="Move the box",
    )
    note = Label(value="Insert a box to list the nodes inside it.")
    nodes = Container(widgets=[note], labels=False, label="Nodes in the box")
    table = QTableWidget(0, len(BOX_NODE_COLUMNS))
    table.setObjectName(f"haemolynx_box_nodes_{role}")
    table.setHorizontalHeaderLabels(list(BOX_NODE_COLUMNS))
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setSelectionMode(QAbstractItemView.SingleSelection)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    table.setMinimumHeight(110)
    nodes.native.layout().addWidget(table)
    nodes.table = table
    nodes.note = note
    return {
        "insert_box": PushButton(text="Insert a box"),
        "box_choice": ComboBox(choices=["(no box yet)"], label="Box"),
        "box_size": size,
        "box_step": FloatSpinBox(value=5.0, min=0.1, max=1000.0, step=1.0,
                                 label="Move step (um)"),
        "box_move": move,
        "box_nodes": nodes,
        "use_node": PushButton(text=f"Use selected node as {role_title(role)}"),
        "remove_box": PushButton(text="Remove this box"),
    }


@functools.lru_cache(maxsize=512)
def _resolved_path(path: str) -> Path:
    """``Path(path).resolve()``, once per path.

    The panel compares every loaded path with its row on each read of the
    settings, and resolving touches the file system -- on a Mac, a run made on
    a server names ``/home/...`` paths, an automount point that makes each
    lookup slow enough that every settings change lagged by half a second.
    """
    return Path(path).resolve()


def _boundary_controls(viewer, rows, fields, schema, report, boundaries_input=None):
    """The Boundaries tab's "point at it instead of typing it" controls.

    Two layers, both editable, both holding exactly what the settings hold --
    napari world coordinates are already the microns the settings store, so the
    layers are the settings rather than a view of them. Which direction wins is
    the only rule worth stating: *the layers are authoritative while the user
    is editing them; the rows are authoritative when a config is loaded or
    "Show" is pressed.* The two directions are therefore never both live, which
    is why there is no cycle to break -- the `applying` guard below is for the
    fact that `events.data` fires twice per edit, not for a feedback loop.

    *boundaries_input* (optional) returns the graph the Boundaries stage would
    run on -- the '3. Graph' network, before any large-vessel cut -- which is
    what the box tools list nodes from: a node chosen there has an ID the next
    Boundaries run will find.

    Returns None without a viewer: the panel is buildable outside napari.
    """
    if viewer is None:
        return None

    import napari
    from magicgui.widgets import (
        ComboBox,
        Container,
        FloatSlider,
        FloatSpinBox,
        Label,
        PushButton,
    )

    from haemolynx.gui.boundary_picking import (
        AUTOMATED_OVERRIDES_MANUAL_NOTE,
        BC_BOX_NAMES,
        BC_COORDINATES,
        BC_LAYER_NAMES,
        BC_NODE_IDS,
        BC_REGION_NAMES,
        DISABLED_ROLE_TOOLTIP,
        band_boxes,
        box_colours,
        box_from_view_drag,
        box_mesh,
        boxes_name,
        role_boxes,
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
        node_id_note,
        node_id_setting,
        settings_for_method,
        settings_from_layers,
        regions_name,
        snap,
        terminal_axis_span,
        terminal_points,
        toggle_node_id,
        volume_setting,
        wanted_rows,
        BC_BOX_NODES,
        BOX_NODE_COLUMNS,
        rectangle_from_box,
        DEFAULT_BOX_SIZE_UM,
        MOVE_DIRECTIONS,
        box_around,
        box_node_rows,
        box_nodes_spec,
        box_size,
        move_box,
        nodes_in_box,
        resize_box,
        use_node_for_role,
        view_centre_zyx,
    )
    from haemolynx.graph.boundaries import inlets_outlets_from_vessel_masks
    from haemolynx.gui.results import BOUNDARY_NODES
    from haemolynx.gui.chrome_tooltips import (
        ACTION_TOOLTIPS,
        SHOW_BOUNDARIES_TOOLTIP,
        SNAP_BOUNDARIES_TOOLTIP,
        STOP_PICKING_NODES_TOOLTIP,
    )
    from haemolynx.gui.graph_click import hit_test_nodes, nearest_node_hit
    from haemolynx.gui.results import role_colours

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

    #: The node-picking button says which way it will go: one press starts
    #: clicking nodes for the role, the next stops it.
    PICK_NODES_TEXT = "Pick nodes in the viewer"
    STOP_PICKING_NODES_TEXT = "Stop picking nodes"

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
            pick_nodes=PushButton(text=PICK_NODES_TEXT),
            clear_nodes=PushButton(text="Clear this role's node IDs"),
            **_box_tool_widgets(name),
        )
        for name in ROLES
    }
    for _action in actions.values():
        for _control, _tip in ACTION_TOOLTIPS.items():
            getattr(_action, _control).tooltip = _tip

    #: The box tools: insert a box, size and move it, list the nodes in it
    #: and choose one. A volume role's page is these; a node-ID role keeps
    #: them as a way to find the node to list.
    BOX_TOOLS = ("insert_box", "box_choice", "box_size", "box_step", "box_move",
                 "box_nodes", "use_node", "remove_box")
    #: Which of a role's controls its chosen method has any use for.
    ACTIONS_FOR_METHOD = {
        "coordinates": ("pick", "move", "assign"),
        "volume": (*BOX_TOOLS, "clear"),
        "node_ids": ("pick_nodes", "clear_nodes", *BOX_TOOLS),
    }
    CONTROLS = ("pick", "draw", "depth", "move", "assign", "clear",
                "pick_nodes", "clear_nodes", *BOX_TOOLS)

    #: `node_pick` is the role a click on a graph node goes to, or None when
    #: clicks are the camera's alone.
    #: `active_box` is, per role, which of its boxes the box tools act on;
    #: `box_nodes` the nodes listed for each role's box, in table order.
    state = SimpleNamespace(applying=False, results=None, connected=set(),
                        visible=frozenset(), hidden=frozenset(), tabs=None,
                        actions={}, draw3d=None, node_pick=None,
                        active_box={}, box_nodes={}, pending_boxes={})

    #: Each role's page, and where each shared row currently sits. Filled in
    #: by `page`; empty until the panel has been laid out.
    holders: dict[str, Any] = {}
    shared_home: dict[str, Any] = {}

    def depth_slider():
        """The region depth on the page the user is looking at."""
        return actions[str(role.value)].depth

    def current_values() -> dict[str, Any]:
        return {name: fields[name].to_setting_value(_safe_widget_value(widget))
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

    def box_graph():
        """The graph boundary nodes are chosen from: what Boundaries runs on.

        Not the network on screen, which after a run is the final one -- cut
        at the large vessels, pruned -- and lacks nodes (and IDs) the
        Boundaries stage will have. Falls back to it before 3. Graph is saved.
        """
        if boundaries_input is not None:
            try:
                found = boundaries_input()
            except Exception:  # noqa: BLE001 - a missing checkpoint is not an error here
                found = None
            if found is not None:
                return found
        return graph()

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
        listed_ids = BoundaryPicks.from_settings(values).node_ids
        for name in ROLES:
            method = values.get(method_setting(name))
            picked = len(values.get(coordinate_setting(name)) or ())
            boxed = len(values.get(volume_setting(name)) or ())
            listed = len(listed_ids.get(name, ()))
            if picked and method != "coordinates":
                notes.append(f"{picked} {name} coordinate(s) but "
                             f"{method_setting(name)} is {method!r}")
            # Under node_ids a box is how the nodes were found, not a pick.
            if boxed and method not in ("volume", "node_ids"):
                notes.append(f"{boxed} {name} region(s) but "
                             f"{method_setting(name)} is {method!r}")
            if listed and method != "node_ids":
                notes.append(f"{listed} {name} node ID(s) but "
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
        bands = band_boxes(values, *box, axis_span=span)
        # A role the masks (or another automation) choose for never reads its
        # band: drawing one would show a selection the run does not make.
        bands = {owner: band for owner, band in bands.items()
                 if role_manual_controls_enabled(owner, values)}
        return bands, span is not None

    def masks_choose_note(values) -> str:
        """While the masks choose the inlets/outlets, say which ones they chose.

        Shows the run's own boundary-nodes layer (hidden by default) -- the
        nodes the masks picked -- since there is no band or box to draw for a
        choice the run makes from the masks.
        """
        if not inlets_outlets_from_vessel_masks(values):
            return ""
        chosen = layer(BOUNDARY_NODES)
        if chosen is None or not len(chosen.data):
            return ("  Inlets and outlets: chosen by the large-vessel masks when a run "
                    "reaches '4. Boundaries' -- none chosen yet.")
        chosen.visible = True
        roles = [str(r) for r in _layer_features(chosen).get("role", ())]
        return (f"  Inlets and outlets: chosen by the large-vessel masks -- "
                f"{roles.count('inlet')} inlet(s) and {roles.count('outlet')} "
                f"outlet(s), shown in '{BOUNDARY_NODES}'. Untick 'Inlets outlets "
                "from vessel masks' to choose them yourself.")

    def redraw() -> None:
        """Settings -> layers. One of the two authoritative directions.

        Guarded, because writing a layer fires its own `events.data`: without
        this the redraw's half-applied layer would be read straight back as if
        the user had edited it, and a role whose features had not been written
        yet would lose its boxes.
        """
        if state.applying:
            return
        # A box mid-move is written first, so the redraw reads where it is.
        flush_boxes()
        state.applying = True
        try:
            values = current_values()
            bands, measured = bands_now(values)
            group = group_for(values, bands, graph())
            # Straight to `_add_or_update`, not through `_apply_layers`: that
            # also rebuilds the vessel tube mesh and re-runs the Z-depth
            # filter, neither of which a boundary edit touches, and doing both
            # on every edit is what made picking lag.
            for spec in group.layers:
                if spec.name in BC_REGION_NAMES and region_layer_holds(spec.name, values):
                    # Already what the settings say -- typically because the
                    # settings came *from* it. Rewriting it anyway would reset
                    # napari's selection and drawing state underneath the user.
                    continue
                try:
                    _add_or_update(viewer, spec)
                except Exception:  # noqa: BLE001 - one bad layer must not stop the rest
                    logger.exception("could not show layer %s", spec.name)
            report.value = (f"Boundary conditions: {group.note}"
                            f"{band_note(bands, measured)}"
                            f"{masks_choose_note(values)}"
                            f"{unused_warning(values)}{offscreen_warning(values)}")
            for name in our_layer_names():
                listen(layer(name))
            drawn = {spec.name for spec in group.layers}
            for name in BC_REGION_NAMES:
                # `_add_or_update` only ever adds and updates, so a role that
                # has nothing left to draw would keep the layer it had. Pressing
                # Draw makes an empty layer to draw into, and taking that away
                # would leave the next click with nowhere to land -- so a layer
                # a tool is pointed at is emptied rather than removed. Being
                # merely selected is not enough: napari selects whatever was
                # added last.
                stale = layer(name)
                if name in drawn or stale is None or not _is_ours(stale):
                    continue
                if getattr(stale, "mode", "pan_zoom") == "pan_zoom":
                    viewer.layers.remove(stale)
                elif len(stale.data):
                    stale.data = []
            # Drawn from the settings and never edited, so it simply goes
            # once no role lists a node.
            stale = layer(BC_NODE_IDS)
            if BC_NODE_IDS not in drawn and stale is not None and _is_ours(stale):
                viewer.layers.remove(stale)
            draw_boxes(values, bands)
            set_depth_range()
            if state.node_pick is not None:
                focus_nodes()
        finally:
            state.applying = False
        refresh_box_tools()

    #: What a layer's `events.data` says once an edit is complete. The same
    #: edit also fires "adding"/"changing"/"removing" first, and napari's
    #: rectangle tool fires "adding" before the rectangle exists -- reading
    #: that one back did nothing but cost a full sync per edit.
    FINISHED_EDITS = frozenset({"added", "changed", "removed"})

    def sync(event=None, *_args) -> None:
        """Layers -> settings. The other authoritative direction."""
        if state.applying:
            return
        action = getattr(event, "action", None)
        if action is not None and str(getattr(action, "value", action)) not in FINISHED_EDITS:
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
                    found = read_regions(target, owner)
                    rectangles += found[0]
                    rectangle_roles += found[1]
                    depths += found[2]
                proposed.update(settings_from_layers(
                    rectangles=rectangles,
                    rectangle_roles=rectangle_roles,
                    depths=depths,
                ))
            write_rows(proposed)
            values = current_values()
            # The boxes are drawn from the settings, so they follow an edit
            # here; the layer that was edited is left exactly as it is.
            draw_boxes(values)
            picks = BoundaryPicks.from_settings(values)
            report.value = (f"Boundary conditions: {picks.summary()}"
                            f"{unused_warning(values)}{offscreen_warning(values)}")
        finally:
            state.applying = False

    def handle_indices(target) -> list[int]:
        """Which shapes can be read as a region: anything with an area.

        Only rectangles are drawn here, but napari's layer controls offer the
        other tools too; an ellipse or polygon still reads as the box around
        it, while a line has no area to make one from.
        """
        kinds = list(target.shape_type)
        return [
            index
            for index in range(len(target.data))
            if kinds[index] not in ("line", "path")
            and len(target.data[index]) >= 3
        ]

    def read_regions(target, owner) -> tuple[list, list[str], list[float]]:
        """A regions layer's rectangles, with each one's role and depth."""
        handles = handle_indices(target)
        roles = roles_of(target, len(target.data), default=owner)
        found = depths_of(target, len(target.data))
        return ([target.data[index] for index in handles],
                [roles[index] for index in handles],
                [found[index] for index in handles])

    def region_layer_holds(name: str, values) -> bool:
        """Whether a role's regions layer already shows what *values* say."""
        target = layer(name)
        owner = next((r for r in ROLES if regions_name(r) == name), None)
        if target is None or owner is None:
            return False
        rectangles, roles, depths = read_regions(target, owner)
        if len(rectangles) != len(target.data) or any(r != owner for r in roles):
            return False
        held = settings_from_layers(rectangles=rectangles, rectangle_roles=roles,
                                    depths=depths)[volume_setting(owner)]
        wanted = BoundaryPicks.from_settings(values).to_settings()[volume_setting(owner)]
        return held == wanted

    def draw_boxes(values=None, bands=None, extra=None) -> None:
        """Each role's regions as translucent solid boxes, one shade per box.

        Its own Surface layer per role rather than more shapes in the regions
        layer: a Surface draws in 3D, where a Shapes layer can only show flat
        rectangles, and it is never edited, so it can be rewritten freely
        without disturbing a tool working on the regions layer. *extra* is a
        box still being dragged out in 3D.
        """
        values = current_values() if values is None else values
        if bands is None:
            bands, _measured = bands_now(values)
        boxes = role_boxes(BoundaryPicks.from_settings(values), bands, extra=extra)
        colours = dict(role_colours())
        for owner in ROLES:
            name = boxes_name(owner)
            existing = layer(name)
            if existing is not None and not _is_ours(existing):
                continue
            mine = boxes.get(owner)
            if not mine:
                if existing is not None:
                    viewer.layers.remove(existing)
                continue
            vertices, faces, which = box_mesh(mine)
            shades = np.asarray(box_colours(colours[owner], len(mine)), dtype=float)[which]
            if existing is not None and existing.__class__.__name__.lower() == "surface":
                _set_tube_mesh(existing, vertices, faces, shades)
                continue
            # Adding a layer makes it the active one, and the camera follows
            # the active layer's `mouse_pan` -- so a box appearing mid-drag
            # would hand the drag to the camera. Put the selection back.
            active = viewer.layers.selection.active
            viewer.add_surface(
                (vertices, faces), name=name, vertex_colors=shades,
                opacity=0.35, shading="flat",
                # Every face of every box, front and back: a box is only
                # readable as a volume if you can see through to its far side.
                blending="translucent_no_depth",
                metadata={OURS: {"kind": "surface", "role": "boundary_boxes"}},
            )
            if active is not None and active in viewer.layers:
                viewer.layers.selection.active = active

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
        # Only the data event. There used to be a drag callback here too,
        # redrawing the layer when the mouse came up -- which ran before
        # napari's own tool had finished adding the rectangle, so it wiped
        # the new region and left the tool holding a stale selection (the
        # `_selected_box` TypeError on the next drag).
        target.events.data.connect(sync)
        state.connected.add(id(target))

    def set_defaults(target) -> None:
        """Tag whatever napari adds next as this role's, at this role's depth.

        Every column the layer has, not just the ones this cares about: a
        default that names a subset is refused outright, and the failure is
        silent -- the next shape then arrives as the role that was chosen
        before, with no `part`, which stops it being read back at all.
        """
        known = {
            "role": str(role.value),
            "depth": float(depth_slider().value),
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
            # the others you need to fill in. An advanced row (the node IDs a
            # run fills in) belongs to the role page's own Advanced button,
            # which no method choice hides.
            if name in schema and schema[name].advanced:
                continue
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

        The masks choosing the inlets and outlets (``automated_vessel_assignment``
        and ``inlets_outlets_from_vessel_masks``) disables Inlet and Outlet;
        small-vessel auto disables Arteriole and Venule. Tabs stay
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
                tabs.setTabToolTip(index, DISABLED_ROLE_TOOLTIP[name])
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
        for name, action in actions.items():
            method = str(values.get(method_setting(name)))
            useful = set(ACTIONS_FOR_METHOD.get(method, ()))
            # Automated assignment overrides the matching manual role tabs.
            overridden = not role_manual_controls_enabled(name, values)
            for control in CONTROLS:
                widget = getattr(action, control)
                widget.visible = control in useful
                if overridden:
                    widget.enabled = False
                    widget.tooltip = DISABLED_ROLE_TOOLTIP[name]
                else:
                    widget.enabled = True
                    widget.tooltip = ACTION_TOOLTIPS[control]
            state.actions[name] = frozenset(useful)
            if state.node_pick == name and ("pick_nodes" not in useful or overridden):
                # The method moved off node_ids, or automation took the role
                # over: a click must not go on writing IDs nothing reads.
                disarm_nodes()
        label_pick_nodes()

    def on_settings_changed(*_args) -> None:
        """Follow the form: the layers show what the settings currently say."""
        refresh_rows()
        if state.applying:
            return
        if any(name in viewer.layers for name in (*our_layer_names(), BC_NODE_IDS)):
            redraw()
        else:
            refresh_box_tools()

    def on_show() -> None:
        redraw()

    def on_pick() -> None:
        disarm_3d()
        disarm_nodes()
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
        disarm_3d()
        disarm_nodes()
        if viewer.dims.ndisplay == 3 and image_extent() is None:
            report.value = (
                "Open an image first: a box drawn in 3D runs through the image "
                "along the line of sight, so it needs an image to run through."
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
        if viewer.dims.ndisplay == 3:
            arm_3d(target)
            report.value = (
                f"Drag a rectangle across the 3D view for a {role.value} region. "
                "Across the screen the box takes what you drag; along the line "
                "of sight it runs through the whole image (looking down z, the "
                f"Region depth below: {depth_slider().value:.0f} um). Snap the "
                "view to XY, XZ or YZ for an exact box -- at an angle it takes "
                "the smallest box holding the rectangle. The view stops turning "
                "until the drag is done; press Draw again for another box."
            )
            return
        target.mode = "add_rectangle"
        report.value = (
            f"Draw rectangles for {role.value} regions -- as many as you like. "
            f"Each takes the depth below ({depth_slider().value:.0f} um), "
            "centred on the slice you draw it on."
        )

    def arm_3d(target) -> None:
        """Let the next drag on *target* make a box instead of turning the view.

        The camera follows the active layer's `mouse_pan`, which is how napari's
        own 3D plane dragging keeps the view still; the same switch here.
        """
        state.draw3d = SimpleNamespace(layer=target, owner=str(role.value))
        target.mouse_pan = False
        if draw_in_3d not in target.mouse_drag_callbacks:
            target.mouse_drag_callbacks.append(draw_in_3d)

    def disarm_3d() -> None:
        """Give the mouse back to the camera."""
        armed, state.draw3d = state.draw3d, None
        if armed is None:
            return
        try:
            armed.layer.mouse_pan = True
            if draw_in_3d in armed.layer.mouse_drag_callbacks:
                armed.layer.mouse_drag_callbacks.remove(draw_in_3d)
        except Exception:  # noqa: BLE001 - a layer removed meanwhile has nothing to restore
            logger.debug("could not disarm 3D region drawing", exc_info=True)

    def draw_in_3d(target, event):
        """One box per drag: previewed as it grows, written when the mouse is up."""
        armed = state.draw3d
        extent = image_extent()
        if (armed is None or armed.layer is not target
                or viewer.dims.ndisplay != 3 or extent is None):
            return
        start = np.asarray(event.position, dtype=float)
        view = getattr(event, "view_direction", None)
        up = getattr(event, "up_direction", None)
        if view is None or up is None:
            view, up = viewer.camera.view_direction, viewer.camera.up_direction
        depth = float(actions[armed.owner].depth.value)

        def box_to(position):
            return box_from_view_drag(start, position, view, up, *extent, depth=depth)

        yield
        while event.type == "mouse_move":
            box = box_to(event.position)
            draw_boxes(extra={armed.owner: [box]} if box is not None else None)
            yield
        box = box_to(event.position)
        disarm_3d()
        if box is None:
            draw_boxes()
            report.value = (
                "That drag made no box -- it had no area across the screen, or "
                "missed the image. Press Draw and drag across the image."
            )
            return
        name = volume_setting(armed.owner)
        held = BoundaryPicks.from_settings(current_values()).to_settings()[name]
        # Writing the row redraws: the settings are the one place a box lives.
        write_rows({name: [*held, box]})
        report.value = (
            f"Added a box for {armed.owner}: {box[0]} to {box[1]} um (z, y, x). "
            "Press Draw for another; the depth slider trims its z extent."
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
                "Regions can only be moved or resized in the 2D view -- napari "
                "does not allow editing a Shapes layer in 3D. In 3D, Draw adds "
                "a box and Clear removes this role's boxes. Coordinates can be "
                "moved in either view."
            )
            return
        disarm_3d()
        disarm_nodes()
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

    # --- the box tools: insert a box, size and move it, choose a node in it.

    def boxes_of(owner: str, values=None) -> list:
        """*owner*'s boxes: the ones being moved if any, else the setting's."""
        if owner in state.pending_boxes:
            return [list(map(list, box)) for box in state.pending_boxes[owner]]
        values = current_values() if values is None else values
        return BoundaryPicks.from_settings(values).to_settings()[volume_setting(owner)]

    from qtpy.QtCore import QTimer

    #: Moving or resizing a box redraws only its own two layers at once; the
    #: setting, the node list and the box's nodes follow once the clicks stop.
    #: Writing the row on every click redrew every boundary layer twice and
    #: rebuilt six roles' node lists, which in a full viewer -- vessel tubes
    #: and all, redrawn per layer change -- made each press lag.
    BOX_SETTLE_MS = 250
    settle_timer = QTimer()
    settle_timer.setSingleShot(True)
    settle_timer.setInterval(BOX_SETTLE_MS)

    def preview_boxes(owner: str, boxes) -> None:
        """Show *owner*'s boxes where they now are: its rectangle and 3D box only."""
        state.pending_boxes[owner] = boxes
        was = state.applying
        state.applying = True
        try:
            regions = layer(regions_name(owner))
            if regions is not None and _is_ours(regions) and len(regions.data) == len(boxes):
                regions.data = [rectangle_from_box(lo, hi)[0] for lo, hi in boxes]
            solids = layer(boxes_name(owner))
            if solids is not None and _is_ours(solids) and boxes:
                vertices, faces, which = box_mesh(boxes)
                shades = np.asarray(
                    box_colours(dict(role_colours())[owner], len(boxes)), dtype=float
                )[which]
                _set_tube_mesh(solids, vertices, faces, shades)
        finally:
            state.applying = was
        settle_timer.start()

    def flush_boxes() -> None:
        """Write the boxes moved since the last write, then list their nodes."""
        settle_timer.stop()
        pending, state.pending_boxes = state.pending_boxes, {}
        if not pending:
            return
        was = state.applying
        state.applying = True
        try:
            write_rows({volume_setting(owner): boxes for owner, boxes in pending.items()})
        finally:
            state.applying = was
        refresh_box_tools(only=tuple(pending))
        owner = str(role.value)
        if owner in pending:
            boxes = pending[owner]
            index = active_index(owner, boxes)
            found = state.box_nodes.get(owner, [])
            if index is not None:
                report.value = (
                    f"{owner} box {index + 1}: {boxes[index][0]} to {boxes[index][1]} um "
                    f"(z, y, x). {len(found)} node(s) in it, "
                    f"{sum(n.open_end for n in found)} open end(s)."
                )

    settle_timer.timeout.connect(flush_boxes)

    def active_index(owner: str, boxes) -> int | None:
        """Which of *owner*'s boxes the tools act on: the chosen one, else the last."""
        if not boxes:
            return None
        index = state.active_box.get(owner, len(boxes) - 1)
        index = min(max(int(index), 0), len(boxes) - 1)
        state.active_box[owner] = index
        return index

    def entered_size(owner: str) -> list[float]:
        return [float(w.value) for w in actions[owner].box_size]

    def view_centre() -> list[float]:
        dims = viewer.dims
        return view_centre_zyx(
            dims.point, tuple(int(a) for a in dims.displayed), viewer.camera.center
        )

    def write_boxes(owner: str, boxes, active: int | None) -> None:
        if active is not None:
            state.active_box[owner] = active
        write_rows({volume_setting(owner): boxes})
        # Writing the row only redraws once the BC layers are on screen.
        redraw()

    def on_insert_box() -> None:
        owner = str(role.value)
        flush_boxes()
        disarm_3d()
        disarm_nodes()
        try:
            box = box_around(view_centre(), entered_size(owner))
        except ValueError as error:
            report.value = f"Could not insert a box: {error}"
            return
        boxes = [*boxes_of(owner), box]
        write_boxes(owner, boxes, len(boxes) - 1)
        found = state.box_nodes.get(owner, [])
        report.value = (
            f"Inserted box {len(boxes)} for {owner} at the middle of the view: "
            f"{box[0]} to {box[1]} um (z, y, x). Move it with the arrows, change "
            f"its size above, then choose a node from the list "
            f"({len(found)} in it, {sum(n.open_end for n in found)} open end(s))."
        )

    def on_box_choice() -> None:
        owner = str(role.value)
        flush_boxes()
        choice = str(actions[owner].box_choice.value or "")
        if choice.startswith("Box "):
            state.active_box[owner] = int(choice.split()[1]) - 1
        refresh_box_tools()

    def on_box_size() -> None:
        owner = str(role.value)
        boxes = boxes_of(owner)
        index = active_index(owner, boxes)
        if index is None:
            return
        try:
            boxes[index] = resize_box(boxes[index], entered_size(owner))
        except ValueError as error:
            report.value = f"Could not resize the box: {error}"
            return
        preview_boxes(owner, boxes)

    def on_move_box(direction: str) -> None:
        owner = str(role.value)
        boxes = boxes_of(owner)
        index = active_index(owner, boxes)
        if index is None:
            report.value = "Insert a box first, then move it."
            return
        displayed = [int(a) - max(0, viewer.dims.ndim - 3) for a in viewer.dims.displayed]
        camera = {}
        if viewer.dims.ndisplay == 3:
            # Relative to how the 3D view is turned, not to the image's axes.
            camera = {"view_direction": viewer.camera.view_direction,
                      "up_direction": viewer.camera.up_direction}
        boxes[index] = move_box(boxes[index], direction, float(actions[owner].box_step.value),
                                displayed, **camera)
        preview_boxes(owner, boxes)

    def on_remove_box() -> None:
        owner = str(role.value)
        flush_boxes()
        boxes = boxes_of(owner)
        index = active_index(owner, boxes)
        if index is None:
            report.value = f"{owner} has no box to remove."
            return
        del boxes[index]
        write_boxes(owner, boxes, max(index - 1, 0) if boxes else None)
        if not boxes:
            state.active_box.pop(owner, None)
        report.value = f"Removed {owner} box {index + 1}."

    def selected_box_node(owner: str):
        table = actions[owner].box_nodes.table
        rows_chosen = sorted({i.row() for i in table.selectionModel().selectedRows()})
        listed = state.box_nodes.get(owner, [])
        return listed[rows_chosen[0]] if rows_chosen and rows_chosen[0] < len(listed) else None

    def on_box_node_selected(owner: str) -> None:
        """Mark the chosen row's node in the viewer, among the box's nodes."""
        target = layer(BC_BOX_NODES)
        chosen = selected_box_node(owner)
        if target is None or chosen is None:
            return
        listed = state.box_nodes.get(owner, [])
        target.selected_data = {listed.index(chosen)}

    def on_use_node() -> None:
        owner = str(role.value)
        flush_boxes()
        chosen = selected_box_node(owner)
        if chosen is None:
            report.value = "Click a node in the list of nodes in the box first."
            return
        if not role_manual_controls_enabled(owner, current_values()):
            report.value = DISABLED_ROLE_TOOLTIP[owner]
            return
        proposed, message = use_node_for_role(current_values(), owner, chosen.node_id)
        write_rows(proposed)
        redraw()
        listed = BoundaryPicks.from_settings(current_values()).node_ids.get(owner, ())
        kind = "an open end" if chosen.open_end else f"a junction of {chosen.degree} vessels"
        report.value = (
            f"{message} It is {kind}. {role_title(owner)} now takes exactly node(s) "
            f"{list(listed)} (method node_ids); the box stays as a finder."
        )

    def refresh_box_tools(only: Sequence[str] | None = None) -> None:
        """Each role's box list, size, node table, and the box's nodes drawn.

        *only* limits it to those roles -- after a move, the one role moved.
        """
        from qtpy.QtWidgets import QTableWidgetItem

        values = current_values()
        g = box_graph()
        was = state.applying
        state.applying = True
        try:
            for owner, action in actions.items():
                if only is not None and owner not in only:
                    continue
                boxes = boxes_of(owner, values)
                index = active_index(owner, boxes)
                choices = [f"Box {i + 1}" for i in range(len(boxes))] or ["(no box yet)"]
                action.box_choice.choices = choices
                action.box_choice.value = choices[index if index is not None else 0]
                if index is not None:
                    for widget, size in zip(action.box_size, box_size(boxes[index])):
                        widget.value = max(float(widget.min), min(float(widget.max), size))
                found = nodes_in_box(g, boxes[index]) if index is not None else []
                state.box_nodes[owner] = found
                table = action.box_nodes.table
                table.setRowCount(len(found))
                for r, cells in enumerate(box_node_rows(found)):
                    for c, text in enumerate(cells):
                        table.setItem(r, c, QTableWidgetItem(text))
                if index is None:
                    note = "Insert a box to list the nodes inside it."
                elif g is None:
                    note = "Run at least '3. Graph' to list the nodes inside the box."
                else:
                    note = (f"{len(found)} node(s) in box {index + 1}, "
                            f"{sum(n.open_end for n in found)} of them open ends "
                            "(listed first). Click one, then Use selected node.")
                action.box_nodes.note.value = note
        finally:
            state.applying = was
        draw_box_nodes(values)

    def draw_box_nodes(values) -> None:
        """The open role's box nodes in the viewer, if its page shows the box tools."""
        owner = str(role.value)
        shows = "box_nodes" in state.actions.get(owner, ()) and \
            role_manual_controls_enabled(owner, values)
        found = state.box_nodes.get(owner, []) if shows else []
        existing = layer(BC_BOX_NODES)
        if not shows:
            if existing is not None and _is_ours(existing):
                viewer.layers.remove(existing)
            return
        if not found and existing is None:
            return
        # Updated in place, never removed and re-added: removing a layer
        # rebuilds napari's whole scene graph and forces a garbage collection
        # (~0.4 s on a full run), and `_add_or_update` removes a Points layer
        # whenever it shrinks -- which a box moving off a node does every time.
        spec = box_nodes_spec(found)
        try:
            if existing is not None and _is_ours(existing) and \
                    existing.__class__.__name__ == "Points":
                options = spec.options
                existing.data = spec.data
                existing.features = dict(spec.features)
                if len(spec.data):
                    existing.size = options["size"]
                    existing.face_color = options["face_color"]
                    existing.border_color = options["border_color"]
                existing.visible = True
            else:
                _add_or_update(viewer, spec)
        except Exception:  # noqa: BLE001 - drawing must never stop the tab
            logger.exception("could not draw the nodes in the box")

    # --- node_ids: click a node of the graph to list it for the role. The
    # click is caught at the viewer rather than on the nodes layer, because a
    # run replaces that layer whenever its node count changes and a callback
    # on the old one would silently stop answering.

    #: How far, in screen pixels, the mouse may wander between press and
    #: release and still be a click rather than a drag that turns the view.
    CLICK_SLOP_PX = 4.0
    #: How far from a node dot, in screen pixels, a 3D click may land and
    #: still take that node.
    NODE_PICK_SLOP_PX = 8.0

    def label_pick_nodes() -> None:
        """Each role's button says what pressing it will do."""
        for name, action in actions.items():
            armed = state.node_pick == name
            action.pick_nodes.text = STOP_PICKING_NODES_TEXT if armed else PICK_NODES_TEXT
            if action.pick_nodes.enabled:
                action.pick_nodes.tooltip = (STOP_PICKING_NODES_TOOLTIP if armed
                                             else ACTION_TOOLTIPS["pick_nodes"])

    def focus_nodes() -> None:
        """Show the nodes layer and make it the active one.

        Active, so napari's status bar names the node under the cursor --
        that is how the ID being clicked can be read before clicking it.
        """
        nodes = layer(NODES)
        if nodes is None:
            return
        nodes.visible = True
        if viewer.layers.selection.active is not nodes:
            viewer.layers.selection.active = nodes
        if getattr(nodes, "mode", "pan_zoom") != "pan_zoom":
            nodes.mode = "pan_zoom"

    def arm_nodes(owner: str) -> None:
        state.node_pick = owner
        if pick_node_click not in viewer.mouse_drag_callbacks:
            viewer.mouse_drag_callbacks.append(pick_node_click)
        label_pick_nodes()

    def disarm_nodes() -> None:
        """Give clicks back to the camera."""
        state.node_pick = None
        if pick_node_click in viewer.mouse_drag_callbacks:
            viewer.mouse_drag_callbacks.remove(pick_node_click)
        label_pick_nodes()

    def on_pick_nodes() -> None:
        """Start (or stop) clicking nodes for the open role's node IDs."""
        owner = str(role.value)
        if state.node_pick == owner:
            disarm_nodes()
            report.value = f"Stopped picking {owner} nodes."
            return
        nodes = layer(NODES)
        if graph() is None or nodes is None or not len(nodes.data):
            report.value = (
                "Nothing to pick yet: node IDs name the nodes of the graph a run "
                "builds, so run at least '3. Graph' first, then click its nodes."
            )
            return
        disarm_3d()
        arm_nodes(owner)
        redraw()
        focus_nodes()
        report.value = (
            f"Click a node in the viewer to add it to {owner}'s node IDs; click "
            "it again to take it off. The status bar names the node under the "
            "cursor. Dragging still turns the view -- only a click picks. Press "
            f"'{STOP_PICKING_NODES_TEXT}' when done."
        )

    def pick_node_at(position, view_direction=None, dims_displayed=None) -> None:
        """Toggle the node under *position* in the armed role's node IDs."""
        owner = state.node_pick
        nodes = layer(NODES)
        if owner is None or nodes is None:
            return
        if not role_manual_controls_enabled(owner, current_values()):
            disarm_nodes()
            report.value = DISABLED_ROLE_TOOLTIP[owner]
            return
        dims = list(dims_displayed or ())
        try:
            index = nodes.get_value(
                position, view_direction=view_direction if len(dims) == 3 else None,
                dims_displayed=dims or None, world=True,
            )
        except TypeError:
            index = nodes.get_value(position, world=True)
        hit = hit_test_nodes(index, _layer_features(nodes))
        if hit is None and len(dims) == 3 and view_direction is not None:
            # A whole volume in 3D makes each dot a few pixels across, so a
            # near miss takes the node it was aimed at. Not in 2D, where the
            # dot is on the slice and a projected miss could reach through
            # to another slice's node.
            zoom = float(getattr(viewer.camera, "zoom", 0.0) or 0.0)
            hit = nearest_node_hit(
                nodes.data, _layer_features(nodes), position,
                max_distance=NODE_PICK_SLOP_PX / zoom if zoom > 0 else 0.0,
                view_direction=view_direction, dims=dims,
            )
        if hit is None:
            report.value = (
                f"No node under that click. Click on one of the node dots to "
                f"add it to {owner}'s node IDs."
            )
            return
        proposed, action = toggle_node_id(current_values(), owner, hit.node_id)
        write_rows(proposed)
        if BC_NODE_IDS not in viewer.layers:
            # `on_settings_changed` redraws only once something is drawn.
            redraw()
        focus_nodes()
        listed = BoundaryPicks.from_settings(current_values()).node_ids.get(owner, ())
        report.value = (
            f"{action} {owner} node IDs: {list(listed)}. Click another node, click "
            f"one again to take it off, or press '{STOP_PICKING_NODES_TEXT}'."
            f"{node_id_note(graph(), current_values())}"
        )

    def _moved_far(start, now) -> bool:
        if start is None or now is None:
            return False
        return float(np.hypot(*(np.asarray(now, dtype=float)[:2]
                                - np.asarray(start, dtype=float)[:2]))) > CLICK_SLOP_PX

    def pick_node_click(_viewer, event):
        """A press and release with no drag between them picks a node."""
        if state.node_pick is None or getattr(event, "button", None) not in (None, 1):
            return
        # napari hands the generator the same event object with each later
        # event swapped in, so the press has to be copied now.
        position = tuple(float(v) for v in event.position)
        view_direction = getattr(event, "view_direction", None)
        if view_direction is not None:
            view_direction = tuple(float(v) for v in view_direction)
        dims = list(getattr(event, "dims_displayed", ()) or ())
        start_pixel = getattr(event, "pos", None)
        start_pixel = None if start_pixel is None else tuple(start_pixel)
        dragged = False
        yield
        while event.type == "mouse_move":
            pixel = getattr(event, "pos", None)
            if _moved_far(start_pixel, pixel) or (
                pixel is None and tuple(float(v) for v in event.position) != position
            ):
                dragged = True
            yield
        if dragged:
            return
        try:
            pick_node_at(position, view_direction, dims)
        except Exception as exc:  # noqa: BLE001 - report it, don't crash the viewer
            logger.exception("picking a boundary node failed")
            report.value = f"Picking a node failed: {exc}"

    def on_clear_nodes() -> None:
        write_rows({node_id_setting(str(role.value)): []})
        redraw()

    for _name in (
        *(name for _role in ROLES
          for name in (method_setting(_role), coordinate_setting(_role),
                       volume_setting(_role), node_id_setting(_role))),
        # The band settings draw a box too, so they move the picture as much
        # as a coordinate does.
        *shared_settings(),
        "automated_vessel_assignment",
        "inlets_outlets_from_vessel_masks",
        "use_small_vessel_masks_for_boundary_assignment",
        "assign_large_vessel_branch_orders",
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
        wire(_name, _action.pick_nodes, on_pick_nodes)
        wire(_name, _action.clear_nodes, on_clear_nodes)
        wire(_name, _action.insert_box, on_insert_box)
        wire(_name, _action.box_choice, on_box_choice)
        wire(_name, _action.remove_box, on_remove_box)
        wire(_name, _action.use_node, on_use_node)
        for _spin in _action.box_size:
            wire(_name, _spin, on_box_size)
        for _button in _action.box_move:
            wire(_name, _button, lambda d=_button.name: on_move_box(d))
        _action.box_nodes.table.itemSelectionChanged.connect(
            lambda owner=_name: on_box_node_selected(owner))
        _action.box_nodes.table.itemDoubleClicked.connect(
            lambda _item, owner=_name: (setattr(role, "value", owner), on_use_node()))

    def on_ndisplay(*_args) -> None:
        # A drag armed in 3D would, back in 2D, fight napari's own tools for
        # the mouse; and napari resets `mouse_pan` on the mode change anyway.
        disarm_3d()
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

        state.disclosures = {}
        state.group_boxes = {}
        role_tabs = QTabWidget()
        placed: set[str] = set()
        for name in ROLES:
            mine = [n for n in role_settings(name) if n in rows]
            placed.update(mine)
            # The node IDs a run fills in sit behind the page's own Advanced
            # button, below the pick/draw controls.
            advanced = [n for n in mine if schema[n].advanced]
            ordinary = [n for n in mine if n not in advanced]
            action = actions[name]
            nodes = None
            if advanced:
                nodes = _AdvancedDisclosure(role_nodes_disclosure(name, advanced), rows)
            holder = Container(
                widgets=[
                    *(rows[n] for n in ordinary),
                    action.pick,
                    action.insert_box, action.box_choice, action.box_size,
                    action.box_step, action.box_move, action.box_nodes,
                    action.use_node, action.remove_box,
                    action.draw, action.depth,
                    action.move, action.assign, action.clear,
                    action.pick_nodes, action.clear_nodes,
                    *((nodes.container,) if nodes is not None else ()),
                ],
                labels=True,
            )
            if nodes is not None:
                _hide_row_label(nodes.container)
                state.disclosures[nodes.spec.key] = nodes
            holders[name] = holder
            role_tabs.addTab(holder.native, role_title(name))
            role_tabs.setTabToolTip(role_tabs.count() - 1,
                                    fields[method_setting(name)].help)

        shared = [n for n in shared_settings() if n in rows]
        rest = [n for n in names if n not in placed and n not in shared]
        # The vessel masks, boxed large and small; the node-ID rows no role
        # page owns go last, under the role tabs.
        laid_out = layout_for("assign_boundaries", rest, schema)
        trailing = tuple(box for box in laid_out.boxes if box.key == "boundary_nodes")
        leading = TabLayout(boxes=tuple(b for b in laid_out.boxes if b not in trailing))
        # Shared main/large/small ilastik knobs are reparented here when only
        # vessel-mask ilastik is on (declared under Input; same widgets).
        shared_ilastik_holder = Container(widgets=[], labels=False)
        _flush(shared_ilastik_holder)

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(summary.native)
        if leading.boxes:
            masks, _ = _lay_out_boxes(
                leading, rows, disclosures=state.disclosures,
                group_boxes=state.group_boxes,
            )
            layout.addWidget(masks)
        layout.addWidget(shared_ilastik_holder.native)
        layout.addWidget(Label(value=AUTOMATED_OVERRIDES_MANUAL_NOTE).native)
        layout.addWidget(widget.native)
        layout.addWidget(role_tabs)
        if trailing:
            node_ids, _ = _lay_out_boxes(
                TabLayout(boxes=trailing), rows, disclosures=state.disclosures,
                group_boxes=state.group_boxes,
            )
            layout.addWidget(node_ids)
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
        if state.draw3d is not None and state.draw3d.owner != str(role.value):
            disarm_3d()
        if state.node_pick is not None and state.node_pick != str(role.value):
            # Clicks follow the page being looked at, as a picked point does:
            # a node clicked with the Outlet page open is an outlet.
            if "pick_nodes" in state.actions.get(str(role.value), ()) and \
                    role_manual_controls_enabled(str(role.value), current_values()):
                arm_nodes(str(role.value))
            else:
                disarm_nodes()
        for name in our_layer_names():
            target = layer(name)
            if target is not None:
                set_defaults(target)
        draw_box_nodes(current_values())

    role.changed.connect(on_role_changed)

    return SimpleNamespace(
        widget=widget, role=role, actions=actions, depth_slider=depth_slider,
        holders=holders, shared_home=shared_home,
        state=state, page=page,
        row_order=row_order, refresh_rows=refresh_rows,
        show=on_show, pick=on_pick, draw=on_draw, snap=on_snap, move=on_move,
        assign=on_assign, clear=on_clear, redraw=redraw, sync=sync,
        draw_in_3d=draw_in_3d,
        pick_nodes=on_pick_nodes, clear_nodes=on_clear_nodes,
        insert_box=on_insert_box, remove_box=on_remove_box, flush_boxes=flush_boxes,
        # Moves wait for the clicks to stop before they are written; called
        # directly (a script, a test) a move is written at once.
        move_box=lambda direction: (on_move_box(direction), flush_boxes()),
        use_node=on_use_node, refresh_box_tools=refresh_box_tools,
        pick_node_at=pick_node_at, pick_node_click=pick_node_click,
        layer_names=(BC_COORDINATES, BC_NODE_IDS, *BC_REGION_NAMES, *BC_BOX_NAMES),
        shared_ilastik_holder=lambda: getattr(state, "shared_ilastik_holder", None),
        disclosures=lambda: dict(getattr(state, "disclosures", {})),
        group_boxes=lambda: dict(getattr(state, "group_boxes", {})),
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
            {LIST_SETTING: fields[LIST_SETTING].to_setting_value(_safe_widget_value(widget))}
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
        lingering as a value nothing applies -- and so has one the entry's own
        choices hide (a capillary block's probability once it blocks listed
        vessels instead).

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
        values = entry_values(editor)
        return {
            name: value
            for name, value in read.items()
            if value is not None and entry_shows(name, values)
        }

    def entry_shows(name: str, values: Mapping[str, Any]) -> bool:
        """Whether an entry's option applies, given the whole chain of gates above it."""
        return fields[name].is_visible(values) and all(
            is_prerequisite_met(rule, values) for rule in prerequisite_chain(schema, name)
        )

    def entry_values(editor) -> dict[str, Any]:
        """What this entry's run would read: the baseline, then its own options."""
        values = {
            name: fields[name].to_setting_value(_safe_widget_value(widget))
            for name, widget in rows.items()
        }
        for name in rows_for_type(editor.type.value):
            if name in editor.editors:
                values[name] = fields[name].to_setting_value(
                    _safe_widget_value(editor.editors[name])
                )
        return values

    def refresh_entry(editor) -> None:
        """Show this type's options whose own prerequisites this entry meets.

        The same rule a tab row follows, read against the entry's values
        rather than the baseline's: a pericyte mask path shows only while that
        entry places its sites from a mask.
        """
        values = entry_values(editor)
        shown = set(rows_for_type(editor.type.value))
        for name, widget in editor.editors.items():
            widget.visible = name in shown and entry_shows(name, values)
        editor.advanced.refresh(
            lambda n: n in shown and entry_shows(n, values), values, schema
        )

    def lay_out_editors(editor) -> None:
        """Name, type, this type's knobs in SETTINGS_FOR_TYPE order, Remove.

        Visibility alone is not enough: editors are built once for every type,
        so a naive append order buried ``arteriole_diameter_change_percent``
        under pericyte geometry rows on the combined type. The type's advanced
        options go behind the entry's own Advanced button, after the rest.
        """
        chosen = editor.type.value
        shown = set(rows_for_type(chosen))
        order = [name for name in editor_layout_order(chosen) if name in editor.editors]
        behind = [name for name in order if name in shown and schema[name].advanced]
        editor.container.clear()
        editor.advanced.set_rows(behind, editor.editors)
        editor.container.append(editor.name)
        editor.container.append(editor.type)
        for name in order:
            if name not in behind:
                editor.container.append(editor.editors[name])
        editor.container.append(editor.advanced.container)
        _hide_row_label(editor.advanced.container)
        editor.container.append(editor.remove)
        refresh_entry(editor)
        editor.shown = frozenset(name for name in editor.editors if name in shown)
        editor.hidden = frozenset(
            name for name in editor.editors if name not in shown
        )
        editor.layout_order = tuple(
            name for name in editor_layout_order(chosen) if name in editor.editors
        )
        # Appending re-unified the label widths over every row, hidden ones
        # included; cap them again so the editor fits the tab.
        _wrap_row_labels(editor.container.native)
        editor.container.native.updateGeometry()

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
            # The chosen type's advanced options; lay_out_editors fills it.
            advanced=_AdvancedDisclosure(entry_disclosure(index, ()), editors),
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
            # An option's own prerequisites follow the entry even while the
            # entries are being rebuilt from a loaded config.
            refresh_entry(editor)
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
        from qtpy.QtWidgets import QSizePolicy

        flat = [name for name in visible_tab_settings(names) if name in rows]
        state.disclosures = {}
        boxes = _box_containers(
            layout_for("run_perturbations", flat, schema), rows, state.disclosures
        )
        settings = Container(widgets=[container for _box, container in boxes], labels=False)
        _flush(settings)
        explanation = Label(
            value=(
                "Each perturbation re-solves the network from the same "
                "baseline. The values above are what one starts from; the "
                "options below are what it changes."
            )
        )
        body = Container(
            widgets=[stage_summary, settings, explanation, holder, add_button],
            labels=False,
        )
        body.native.layout().setContentsMargins(0, 0, 0, 0)
        # Vertical policy Minimum: the page may grow but never shrinks below
        # its natural height, so a short dock scrolls the tab instead of the
        # scroll area squashing every row (and the wrapped explanation) to
        # its bare minimum height -- the compressed, overlapping rows this
        # tab showed while each perturbation's editors were being added.
        body.native.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        return body.native

    def refresh_entries() -> None:
        """Re-read every entry's options against the baseline as it is now."""
        for editor in state.editors:
            refresh_entry(editor)

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
        refresh_entries=refresh_entries,
        disclosures=lambda: dict(getattr(state, "disclosures", {})),
    )


#: Height a tab scroll area reports, not the height of its rows. Qt's
#: QScrollArea.sizeHint adds the inner widget; that became a 1057px window
#: minimum on Windows, taller than a 1080p work area with a taskbar.
TAB_SCROLL_HINT_WIDTH = 420
TAB_SCROLL_HINT_HEIGHT = 240
TAB_SCROLL_MIN_HEIGHT = 120


def _fitting_scroll_area():
    """A scroll area whose size hint does not include the widget inside it."""
    from qtpy.QtCore import QSize, Qt
    from qtpy.QtWidgets import QScrollArea, QSizePolicy

    class FittingScrollArea(QScrollArea):
        def sizeHint(self):
            return QSize(TAB_SCROLL_HINT_WIDTH, TAB_SCROLL_HINT_HEIGHT)

        def minimumSizeHint(self):
            return QSize(200, TAB_SCROLL_MIN_HEIGHT)

    scroller = FittingScrollArea()
    scroller.setWidgetResizable(True)
    scroller.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    scroller.setMinimumHeight(TAB_SCROLL_MIN_HEIGHT)
    # A tab scrolls up/down only: a long row label (several settings' names
    # run past 50 characters) must wrap onto a second line rather than widen
    # the row and force a sideways scrollbar.
    scroller.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    return scroller


#: Widest a row's label column may be before its text wraps. magicgui pins
#: every label in a Container to the widest label's width -- hidden rows
#: included -- which put a 490px label column on Diameters and a wider one
#: on each perturbation editor (every type's rows are built, most hidden).
ROW_LABEL_MAX_WIDTH = 220

#: Characters a dropdown keeps room for once it may shrink below its
#: longest choice, and the width it may shrink to.
COMBO_MIN_CONTENTS_LENGTH = 16
COMBO_MIN_WIDTH = 140


def _wrap_row_labels(native) -> None:
    """Let every row's label wrap, and every dropdown shrink, to fit the tab.

    magicgui gives each row's ``QLabel`` word wrap off, which is fine for a
    short name but not for e.g. "Segmentation cleanup reconnect max axis
    angle degrees (degrees)" -- long enough on its own to push a row past a
    docked panel's width and force the horizontal scrollbar
    :func:`_fitting_scroll_area` turns off. Wrapping is the general fix,
    applied to every tab rather than only the ones with today's longest
    labels, since a future setting name is not something to special-case for.
    """
    from qtpy.QtWidgets import QComboBox, QLabel, QSizePolicy, QWidget

    for label in native.findChildren(QLabel):
        label.setWordWrap(True)
        # magicgui makes its labels Fixed x Fixed, so a wrapped label still
        # demanded its whole one-line width (the summaries widened tabs 1
        # and 5 past the dock) and a single line's height (the Perturbations
        # tab's wrapped text overlapped the rows below it).
        policy = label.sizePolicy()
        if policy.horizontalPolicy() == QSizePolicy.Fixed:
            label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        # Capped, not cleared: magicgui's unified width still lines the
        # value column up while every label fits under the cap.
        if label.minimumWidth() > ROW_LABEL_MAX_WIDTH:
            label.setMinimumWidth(ROW_LABEL_MAX_WIDTH)
            # A row is as tall as its value widget, not its wrapped label;
            # reserve the lines the label needs at the capped width.
            height = label.heightForWidth(ROW_LABEL_MAX_WIDTH)
            if height > 0:  # -1 when Qt has no wrapped height to give
                label.setMinimumHeight(height)
    # A dropdown otherwise demands the width of its longest choice (450px
    # for the perturbation type); it may shrink now, and its popup still
    # lists every choice in full.
    for combo in native.findChildren(QComboBox):
        combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(COMBO_MIN_CONTENTS_LENGTH)
        # An explicit minimum is what a layout honours over the hint, which
        # under napari's stylesheet can still measure the longest choice.
        if combo.minimumWidth() == 0:
            combo.setMinimumWidth(COMBO_MIN_WIDTH)
    # A layout caches each widget's size, and Qt skips the relayout for a
    # hidden widget, so a tab (or a boundary role's page) not on screen kept
    # the width it had before any of the above. updateGeometry drops a
    # widget's cached size even while hidden; invalidate drops the layout's.
    for child in [native, *native.findChildren(QWidget)]:
        child.updateGeometry()
        layout = child.layout()
        if layout is not None:
            layout.invalidate()


#: Status text for each GraphEditorState.click_add / mode-arm outcome.
_GRAPH_EDITOR_STATUS = {
    "rejected": "First click must land on an existing node or edge.",
    "started": "Branch started -- click along the vessel to extend it, or Finish.",
    "extended": "Extended -- click again, or press Finish.",
    "finished": "Branch finished.",
    "add-armed": "Add branch: click an existing node or edge to start.",
    "delete-armed": "Delete edge: click an edge to remove it.",
    "idle": "Edit graph, then Regenerate to catch up haemodynamics onward.",
}


class _GraphEditorWindow:
    """The floating "Edit" window: Add edge / Finish / Delete edge / Regenerate.

    Plain Qt glue and nothing else -- see :class:`haemolynx.gui.graph_editor.GraphEditorState`
    for what each button actually does to the graph, and :mod:`haemolynx.gui._widget`'s
    own ``on_open_graph_editor`` for how a click on the vessels/nodes layers
    reaches it. A standalone top-level window (not a docked tab), since
    editing needs the viewer in full view underneath it, not a panel
    replacing the settings form.
    """

    def __init__(self, *, on_add, on_finish, on_delete, on_regenerate) -> None:
        from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

        self.native = QWidget()
        self.native.setWindowTitle("HaemoLynx: Edit graph")
        self.native.setObjectName("haemolynx_graph_editor")
        layout = QVBoxLayout(self.native)

        self._status = QLabel(_GRAPH_EDITOR_STATUS["idle"])
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        add_button = QPushButton("Add edge")
        add_button.setToolTip(
            "Click an existing node or edge to start a new branch, then click "
            "along the segmented volume to extend it -- A* routed, preferring "
            "the segmented mask but able to cross a real gap in it. Ends when "
            "a click lands on existing structure, or when Finish is pressed."
        )
        finish_button = QPushButton("Finish")
        finish_button.setToolTip("End the branch being drawn as a new dangling terminal.")
        delete_button = QPushButton("Delete edge")
        delete_button.setToolTip(
            "Click an edge to remove it. A node left at degree 2 by the "
            "removal is collapsed into one continuous edge; a node left with "
            "no edges at all is removed too."
        )
        regenerate_button = QPushButton("Regenerate")
        regenerate_button.setToolTip(
            "Re-run diameters, haemodynamics, perturbations, additional "
            "measurements and export from the edited graph, and refresh "
            "every dependent layer."
        )

        add_button.clicked.connect(lambda: self._run(on_add, "add-armed"))
        finish_button.clicked.connect(lambda: self._run(on_finish, "finished"))
        delete_button.clicked.connect(lambda: self._run(on_delete, "delete-armed"))
        regenerate_button.clicked.connect(lambda: on_regenerate())

        for button in (add_button, finish_button, delete_button, regenerate_button):
            layout.addWidget(button)

    def _run(self, action, status_key: str) -> None:
        action()
        self.report(status_key)

    def report(self, status_key: str) -> None:
        self._status.setText(_GRAPH_EDITOR_STATUS.get(status_key, status_key))

    def show(self) -> None:
        self.native.show()

    def is_open(self) -> bool:
        return bool(self.native.isVisible())


#: The Post processing stage's tab, between Haemodynamics and Perturbations.
#: Its page is its own (no setting rows), and it has no "Run from this stage": its
#: Regenerate graph / Continue / Regenerate from the edited network are that.
POST_PROCESSING_TAB = post_processing_tab_title()

#: How wide a box, in microns, zooming to a junction fits on screen.
JUNCTION_ZOOM_BOX_UM = 60.0

#: Status line of the post-processing tab's click-in-the-viewer edits.
_POST_PROCESSING_EDIT_STATUS = {
    "idle": "Not editing. Press Delete vessel or Add vessel, then click in the viewer.",
    "delete-armed": (
        "Delete vessel: click a vessel in the viewer to remove it. Press Delete "
        "vessel again to stop."
    ),
    "add-armed": (
        "Add vessel: click a node to start from, or a point on a vessel where a "
        "new node should form."
    ),
    "deleted": "Vessel deleted. Click another, or press Delete vessel again to stop.",
}

#: A press that moves further than this (screen pixels) before it is let go
#: is a drag -- turning or panning the view -- not a click.
_CLICK_DRAG_TOLERANCE_PX = 4.0

def _zoom_viewer_to(viewer, position_zyx, box_um: float = JUNCTION_ZOOM_BOX_UM) -> None:
    """Centre the camera on *position_zyx* (microns) and zoom to a *box_um* box.

    In a 2D view the slider of every axis not on screen moves to the point
    too, so the junction is in the slice being shown. The zoom comes from the
    canvas size, not from resetting the view first: a reset redraws the whole
    scene once more, on every click.
    """
    from haemolynx.gui.post_processing import camera_center_for, zoom_for_canvas

    if viewer is None:
        return
    try:
        offset = max(0, int(viewer.dims.ndim) - 3)
        position = [0.0] * offset + [float(c) for c in np.asarray(position_zyx, dtype=float)[:3]]
        displayed = tuple(int(axis) for axis in viewer.dims.displayed)
        if int(viewer.dims.ndisplay) == 2:
            for axis in range(offset, len(position)):
                if axis not in displayed:
                    viewer.dims.set_point(axis, position[axis])
        viewer.camera.center = camera_center_for(position, displayed)
        zoom = zoom_for_canvas(getattr(viewer, "_canvas_size", ()) or (), box_um)
        if zoom is not None:
            viewer.camera.zoom = zoom
    except Exception:  # noqa: BLE001 - zooming is a convenience, never fatal
        logger.exception("could not zoom the viewer to %s", position_zyx)


def _post_processing_controls(
    viewer, report, *, results, boundary_roles, regenerate, running, settings=None,
    paused=None, complete=None,
):
    """The "7. Post processing" page: fix the solved network by hand.

    *results* returns the panel's current ResultLayers (or None), *boundary_roles*
    the run's boundary node ids by role, *regenerate(graph, stop_after=...)*
    runs the Post processing stage on an edited graph (and on to *stop_after*,
    or the end) and says whether it started, *running* says whether a run is
    under way and *settings* (optional) returns the panel's settings -- where
    Add vessel finds the raw data file and the run's centreline smoothing.
    *paused* says whether a run is paused here (Mid-run postprocessing) and
    *complete* whether one has been past here. Callables, because the panel
    builds them before what they read.

    Three buttons at the foot of the page take the edits on. Regenerate graph
    brings them in line -- branch orders, lengths, diameters by the run's own
    methods, thick-vessel bridges -- and reruns Haemodynamics on the edited
    network (the pipeline's ``post_process`` stage); when a run is paused
    here it stays paused, and after a finished run it goes on through
    Perturbations to Export, as Regenerate from the edited network does.
    Continue, while paused, does the same and carries the run on.

    The page edits one working copy of the graph: the junction table's
    Delete and Split, and the edit box's click-in-the-viewer Delete vessel and
    Add vessel. The junction list, its table and their buttons sit behind the
    "Manual 4+ vessel junction correction" checkbox, off at first; while it
    is off a scan neither marks the 4+ junctions in the viewer nor zooms to
    one. Add vessel traces a vessel click by click: from a node, or a
    point on a vessel (where a node will form), through each point clicked
    after it -- every leg found by A* through the segmented mask or the raw
    data, as the box's dropdown says, and drawn as soon as it is -- until a
    click lands on another node or vessel, which finishes it. The graph rules
    live in :mod:`haemolynx.graph.post_processing` and the colours and click
    geometry in :mod:`haemolynx.gui.post_processing`; this is only the Qt glue.
    Clicks only recolour the vessels layer already on screen (and its tubes);
    the layer is rebuilt only after an edit changes the network.
    """
    from qtpy.QtWidgets import (
        QAbstractItemView,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QListWidget,
        QPlainTextEdit,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    from haemolynx.graph import (
        DEFAULT_SPLIT_CONNECTOR_LENGTH_UM,
        IntensityCostField,
        VesselEnd,
        add_traced_vessel,
        connectivity_rows,
        delete_vessels,
        edge_keys,
        has_pending_edits,
        mask_cost_field,
        prune_disconnected_branches,
        split_junction,
        trace_path,
        vessel_path,
        write_connectivity_csv,
    )
    from haemolynx.gui.chrome_tooltips import POST_PROCESSING_TOOLTIPS as tips
    from haemolynx.gui.post_processing import (
        ADDED_NODES,
        HIGH_DEGREE_JUNCTIONS,
        JUNCTION_TABLE_COLUMNS,
        NEW_VESSEL_POINTS,
        NEW_VESSEL_TRACE,
        POST_PROCESSING_LAYERS,
        TRACE_SOURCES,
        TRACE_THROUGH_RAW,
        VesselTrace,
        added_nodes_layer,
        added_vessel_ids,
        branch_id_of,
        CONNECTIVITY_EXPORT_CHOICES,
        INLET_TO_OUTLET_ONLY,
        default_connectivity_csv_path,
        describe_vessels,
        junction_label,
        junction_marker_layer,
        junction_table_rows,
        nearest_node,
        parse_branch_ids,
        point_on_path_under_click,
        point_under_click,
        scan_network,
        status_colours,
        trace_layers,
        vessel_status,
    )
    from haemolynx.gui.results import FWHM_RAW

    state = SimpleNamespace(
        graph=None,
        scan=None,
        inlets=(),
        outlets=(),
        protected=frozenset(),
        decisions={},
        node=None,
        vessels=[],
        #: "idle", "delete" (click a vessel) or "add" (trace a new one).
        mode="idle",
        #: The vessel Add vessel is tracing (a VesselTrace), until it finishes.
        trace=None,
        #: Nodes Add vessel formed on existing vessels, drawn until Regenerate.
        added_nodes=[],
        #: (image data, its cost field): made once, the first time a vessel
        #: is traced through it, and reused while the image stays the same.
        routing=None,
        #: The same for the raw data, and why it could not be had, if not.
        raw=None,
        raw_problem=None,
        #: The connectivity CSV exported last, for Open 2D connectivity map.
        connectivity_csv=None,
    )

    page = QWidget()
    page.setObjectName("haemolynx_post_processing")
    layout = QVBoxLayout(page)
    intro = QLabel(
        "Fix the network by hand: vessels to add or delete and, with Manual 4+ "
        "vessel junction correction on, junctions where four or more vessels "
        "meet. Edits stay in the viewer until Regenerate graph brings them in "
        "line with the rest of the network and reruns Haemodynamics on it. With "
        "Mid-run postprocessing on (1. Input), a run pauses here after "
        "Haemodynamics and Continue runs Perturbations onwards."
    )
    intro.setWordWrap(True)
    scan_button = QPushButton("Scan network")
    scan_button.setToolTip(tips["scan"])
    status = QLabel("Not scanned yet.")
    status.setWordWrap(True)
    junction_toggle = QCheckBox("Manual 4+ vessel junction correction")
    junction_toggle.setObjectName("haemolynx_post_processing_junction_toggle")
    junction_toggle.setToolTip(tips["junction_correction"])
    junction_list = QListWidget()
    junction_list.setObjectName("haemolynx_post_processing_junctions")
    junction_list.setToolTip(tips["junctions"])
    junction_list.setMinimumHeight(80)
    table = QTableWidget(0, len(JUNCTION_TABLE_COLUMNS))
    table.setObjectName("haemolynx_post_processing_vessels")
    table.setToolTip(tips["table"])
    table.setHorizontalHeaderLabels(list(JUNCTION_TABLE_COLUMNS))
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setSelectionMode(QAbstractItemView.ExtendedSelection)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    table.setMinimumHeight(120)
    delete_button = QPushButton("Delete selected vessels")
    delete_button.setToolTip(tips["delete"])
    leave_button = QPushButton("Leave as is")
    leave_button.setToolTip(tips["leave"])
    split_button = QPushButton("Split into bifurcations with a vessel of")
    split_button.setToolTip(tips["split"])
    connector = QDoubleSpinBox()
    connector.setToolTip(tips["connector"])
    connector.setRange(0.5, 500.0)
    connector.setDecimals(1)
    connector.setSuffix(" µm")
    connector.setValue(DEFAULT_SPLIT_CONNECTOR_LENGTH_UM)
    edit_box = QGroupBox("Edit by clicking in the viewer")
    edit_layout = QVBoxLayout(edit_box)
    edit_status = QLabel(_POST_PROCESSING_EDIT_STATUS["idle"])
    edit_status.setWordWrap(True)
    # Checkable: a lit button shows what a click in the viewer does, and
    # pressing it again stops (an Add vessel stops by itself when it finishes).
    click_delete_button = QPushButton("Delete vessel")
    click_delete_button.setToolTip(tips["click_delete"])
    click_delete_button.setCheckable(True)
    add_button = QPushButton("Add vessel")
    add_button.setToolTip(tips["add"])
    add_button.setCheckable(True)
    trace_source = QComboBox()
    trace_source.setObjectName("haemolynx_post_processing_trace_source")
    trace_source.addItems(list(TRACE_SOURCES))
    trace_source.setToolTip(tips["trace_source"])
    edit_layout.addWidget(edit_status)
    edit_row = QHBoxLayout()
    for button in (click_delete_button, add_button, trace_source):
        edit_row.addWidget(button)
    edit_layout.addLayout(edit_row)
    branch_ids = QLineEdit()
    branch_ids.setObjectName("haemolynx_post_processing_branch_ids")
    branch_ids.setPlaceholderText("branchIDs, e.g. 12, 40")
    branch_ids.setToolTip(tips["branch_ids"])
    delete_ids_button = QPushButton("Delete by branch ID")
    delete_ids_button.setToolTip(tips["delete_ids"])
    ids_row = QHBoxLayout()
    ids_row.addWidget(branch_ids, 1)
    ids_row.addWidget(delete_ids_button)
    edit_layout.addLayout(ids_row)

    prune_button = QPushButton("Prune disconnected and dead-end branches")
    prune_button.setToolTip(tips["prune"])
    # The connectivity export: built here, because it exports the network
    # this page edits, but placed on the Export tab with the run's other files.
    connectivity_box = QGroupBox("Network connectivity")
    connectivity_box.setObjectName("haemolynx_connectivity_box")
    connectivity_layout = QVBoxLayout(connectivity_box)
    connectivity_choice = QComboBox()
    connectivity_choice.setObjectName("haemolynx_connectivity_choice")
    connectivity_choice.addItems(list(CONNECTIVITY_EXPORT_CHOICES))
    connectivity_choice.setToolTip(tips["connectivity_choice"])
    export_connectivity_button = QPushButton("Export connectivity CSV")
    export_connectivity_button.setObjectName("haemolynx_post_processing_export_connectivity")
    export_connectivity_button.setToolTip(tips["export_connectivity"])
    connectivity_map_button = QPushButton("Open 2D connectivity map")
    connectivity_map_button.setObjectName("haemolynx_connectivity_map")
    connectivity_map_button.setToolTip(tips["connectivity_map"])
    connectivity_status = QLabel("")
    connectivity_status.setObjectName("haemolynx_connectivity_status")
    connectivity_status.setWordWrap(True)
    connectivity_layout.addWidget(connectivity_choice)
    connectivity_row = QHBoxLayout()
    connectivity_row.addWidget(export_connectivity_button)
    connectivity_row.addWidget(connectivity_map_button)
    connectivity_layout.addLayout(connectivity_row)
    connectivity_layout.addWidget(connectivity_status)
    regenerate_button = QPushButton("Regenerate from the edited network")
    regenerate_button.setObjectName("haemolynx_post_processing_regenerate")
    regenerate_button.setToolTip(tips["regenerate"])
    regenerate_graph_button = QPushButton("Regenerate graph")
    regenerate_graph_button.setObjectName("haemolynx_post_processing_regenerate_graph")
    regenerate_graph_button.setToolTip(tips["regenerate_graph"])
    continue_button = QPushButton("Continue")
    continue_button.setObjectName("haemolynx_post_processing_continue")
    continue_button.setToolTip(tips["continue"])
    commit_status = QLabel("")
    commit_status.setObjectName("haemolynx_post_processing_commit_status")
    commit_status.setWordWrap(True)
    log_box = QPlainTextEdit()
    log_box.setObjectName("haemolynx_post_processing_log")
    log_box.setReadOnly(True)
    log_box.setToolTip(tips["log"])
    log_box.setMinimumHeight(110)

    # Everything the 4+ junction correction needs, nested under its checkbox
    # and hidden until it is ticked.
    junction_box = QWidget()
    junction_box.setObjectName("haemolynx_post_processing_junction_box")
    junction_layout = QVBoxLayout(junction_box)
    junction_layout.setContentsMargins(ADVANCED_INDENT_PX, 0, 0, 0)
    colours = QLabel(
        "The chosen junction's vessels are cyan and the ones selected in the "
        "table yellow."
    )
    colours.setWordWrap(True)
    junction_layout.addWidget(colours)
    junction_layout.addWidget(QLabel("Junctions where 4+ vessels meet:"))
    junction_layout.addWidget(junction_list)
    junction_layout.addWidget(QLabel("Vessels at the selected junction:"))
    junction_layout.addWidget(table)
    row = QHBoxLayout()
    row.addWidget(delete_button)
    row.addWidget(leave_button)
    junction_layout.addLayout(row)
    row = QHBoxLayout()
    row.addWidget(split_button)
    row.addWidget(connector)
    junction_layout.addLayout(row)
    junction_box.setVisible(False)

    layout.addWidget(intro)
    layout.addWidget(scan_button)
    layout.addWidget(status)
    layout.addWidget(junction_toggle)
    layout.addWidget(junction_box)
    layout.addWidget(edit_box)
    layout.addWidget(prune_button)
    layout.addWidget(QLabel("What was changed:"))
    layout.addWidget(log_box)
    row = QHBoxLayout()
    row.addWidget(regenerate_graph_button)
    row.addWidget(regenerate_button)
    layout.addLayout(row)
    layout.addWidget(commit_status)
    layout.addWidget(continue_button)

    def layer(name):
        return viewer.layers[name] if viewer is not None and name in viewer.layers else None

    def log_edit(text: str) -> None:
        """One line in the tab's own log, and the same in napari's run log."""
        log_box.appendPlainText(f"{datetime.now():%H:%M:%S}  {text}")
        logger.info("Post processing: %s", text)

    def selected_rows() -> list[int]:
        rows = sorted({index.row() for index in table.selectionModel().selectedRows()})
        return [r for r in rows if r < len(state.vessels)]

    def recolour() -> None:
        """Colour the vessels layer (and its tubes) by status: cheap, no rebuild."""
        vessels = layer(VESSELS)
        if vessels is None or state.scan is None:
            return
        features = _layer_features(vessels)
        if "edge_index" not in features or len(features["edge_index"]) != len(vessels.data):
            return
        chosen = selected_rows()
        labels = vessel_status(
            np.asarray(features["edge_index"]),
            at_junction=[v.branch_id for v in state.vessels],
            selected=[state.vessels[r].branch_id for r in chosen],
            added=added_vessel_ids(state.graph) if state.graph is not None else (),
        )
        vessels.edge_color = status_colours(labels)
        vessels.visible = True
        # A vessels layer drawn as tubes retints them on its own edge_color
        # event; retinting again rebuilds the whole mesh a second time.
        if not getattr(vessels, "_haemolynx_follow_tubes", False):
            _maybe_retint_vessel_tubes(vessels)

    def junctions_on() -> bool:
        return junction_toggle.isChecked()

    def draw_markers() -> None:
        if viewer is None or state.graph is None or state.scan is None:
            return
        drop_layer(HIGH_DEGREE_JUNCTIONS)
        if not junctions_on():
            return
        spec = junction_marker_layer(state.graph, state.scan)
        if len(spec.data):
            viewer.add_points(
                spec.data,
                name=spec.name,
                features=dict(spec.features),
                metadata={OURS: {"post_processing": True}},
                **spec.options,
            )
            # napari selects a layer it has just added, and a click goes to the
            # selected layer: hand selection back to the vessels.
            vessels = layer(VESSELS)
            if vessels is not None:
                viewer.layers.selection.active = vessels
            attach_click_callbacks()

    def drop_layer(name: str) -> None:
        existing = layer(name)
        if existing is not None:
            _process_pending_qt_events()
            viewer.layers.remove(existing)
            _process_pending_qt_events()

    def put_layers(specs, names) -> None:
        """Show the tab's own layers *specs*; any of *names* not among them goes.

        Each is replaced whole: they are a handful of points or segments. napari
        selects a layer it has just added, and a click goes to the selected
        layer, so selection is handed back to the vessels afterwards.
        """
        if viewer is None:
            return
        wanted = {spec.name: spec for spec in specs if len(spec.data)}
        for name in names:
            drop_layer(name)
            spec = wanted.get(name)
            if spec is None:
                continue
            add = viewer.add_points if spec.kind == "points" else viewer.add_vectors
            add(
                spec.data,
                name=spec.name,
                features=dict(spec.features),
                metadata={OURS: {"post_processing": True}},
                **spec.options,
            )
        vessels = layer(VESSELS)
        if vessels is not None and wanted:
            viewer.layers.selection.active = vessels
        attach_click_callbacks()

    def draw_trace() -> None:
        """Draw the vessel being traced as it stands, or take it away."""
        specs = trace_layers(state.trace) if state.trace is not None else ()
        put_layers(specs, (NEW_VESSEL_TRACE, NEW_VESSEL_POINTS))

    def draw_added_nodes() -> None:
        if state.graph is None:
            return
        state.added_nodes = [n for n in state.added_nodes if n in state.graph]
        put_layers([added_nodes_layer(state.graph, state.added_nodes)], (ADDED_NODES,))

    def redraw_network() -> None:
        """Rebuild the vessels/nodes layers from the edited graph, then recolour."""
        current = results()
        if viewer is None or current is None or state.graph is None:
            return
        _apply_layers(viewer, current.layers_for_graph(state.graph))
        attach_click_callbacks()
        recolour()

    def show_vessels(node) -> None:
        state.node = node
        table.blockSignals(True)
        table.clearSelection()
        if node is None or state.graph is None or node not in state.graph:
            state.vessels = []
            table.setRowCount(0)
        else:
            state.vessels, rows = junction_table_rows(state.graph, node)
            table.setRowCount(len(rows))
            for r, cells in enumerate(rows):
                for c, text in enumerate(cells):
                    table.setItem(r, c, QTableWidgetItem(text))
        table.blockSignals(False)

    def fill_list(prefer=None) -> None:
        junctions = list(state.scan.junctions)
        status.setText(state.scan.summary)
        junction_list.blockSignals(True)
        junction_list.clear()
        for node in junctions:
            junction_list.addItem(junction_label(state.graph, node, state.decisions.get(node)))
        junction_list.blockSignals(False)
        if not junctions_on():
            # Picking one would list its vessels, turn them cyan and zoom to it.
            show_vessels(None)
            recolour()
            return
        if prefer not in junctions:
            prefer = next((n for n in junctions if n not in state.decisions), None)
            if prefer is None and junctions:
                prefer = junctions[0]
        if prefer is None:
            show_vessels(None)
            recolour()
            return
        junction_list.setCurrentRow(junctions.index(prefer))

    def rescan(prefer=None, *, redraw: bool = True) -> None:
        state.scan = scan_network(state.graph)
        state.decisions = {n: d for n, d in state.decisions.items() if n in state.scan.junctions}
        show_vessels(None)
        if redraw:
            redraw_network()
        draw_markers()
        draw_added_nodes()
        fill_list(prefer)
        refresh_buttons()

    def is_paused() -> bool:
        return bool(paused()) if paused is not None else False

    def refresh_buttons() -> None:
        """Which of the three hand-over buttons applies now, and why."""
        busy = bool(running())
        held = is_paused()
        finished = bool(complete()) if complete is not None else False
        pending = has_pending_edits(state.graph)
        regenerate_graph_button.setEnabled(not busy and pending)
        continue_button.setEnabled(not busy and held)
        regenerate_button.setEnabled(not busy and pending and finished and not held)
        if busy:
            text = "A run is going."
        elif held and pending:
            text = (
                "Paused after Haemodynamics, with edits not yet in the model. "
                "Regenerate graph brings them in line, reruns Haemodynamics and "
                "stays paused; Continue does the same and runs Perturbations onwards."
            )
        elif held:
            text = "Paused after Haemodynamics. Continue runs Perturbations onwards."
        elif pending:
            text = (
                "Edits not yet in the model. Regenerate graph brings them in line, "
                "reruns Haemodynamics on them and runs Perturbations through Export."
            )
        else:
            text = ""
        commit_status.setText(text)

    def on_junction_correction_toggled(on: bool) -> None:
        """Show the 4+ junctions and their edits, picking one; or put them away."""
        junction_box.setVisible(on)
        if state.scan is None:
            return
        draw_markers()
        fill_list(prefer=state.node)

    def on_junction_changed(row: int) -> None:
        if state.scan is None or not 0 <= row < len(state.scan.junctions):
            return
        node = state.scan.junctions[row]
        show_vessels(node)
        recolour()
        pos = state.graph.nodes[node].get("pos")
        if pos is not None:
            _zoom_viewer_to(viewer, pos)

    def on_scan() -> None:
        if running():
            report.value = ALREADY_RUNNING
            return
        current = results()
        graph = getattr(current, "_graph", None) if current is not None else None
        if graph is None:
            status.setText("Nothing to check yet: run the pipeline through at least Boundaries first.")
            return
        roles = boundary_roles() or {}
        state.inlets = tuple(roles.get("inlet", ()) or ())
        state.outlets = tuple(roles.get("outlet", ()) or ())
        state.protected = frozenset(n for nodes in roles.values() for n in (nodes or ()))
        stop_editing()
        # The viewer shows this tab's own edited graph until Regenerate: its
        # added nodes are still the ones to mark. Any other graph is a new run.
        if graph is not state.graph:
            state.added_nodes = []
        state.raw = state.raw_problem = None
        state.graph = copy_graph(graph)
        state.decisions = {}
        # The vessels on screen are already this graph: recolour, no rebuild.
        rescan(redraw=False)
        attach_click_callbacks()
        report.value = f"Post processing: {state.scan.summary}"
        log_edit(f"Scanned the network: {state.scan.summary}")

    def export_connectivity(path, *, only_inlet_to_outlet: bool | None = None) -> Path | None:
        """Write how the network on screen is connected to *path*.

        *only_inlet_to_outlet* defaults to the box's dropdown.
        """
        if only_inlet_to_outlet is None:
            only_inlet_to_outlet = connectivity_choice.currentText() == INLET_TO_OUTLET_ONLY
        graph = state.graph
        if graph is None:
            current = results()
            graph = getattr(current, "_graph", None) if current is not None else None
        if graph is None:
            connectivity_status.setText(
                "Nothing to export yet: run the pipeline through at least Boundaries first."
            )
            return None
        roles = boundary_roles() or {}
        inlets = tuple(roles.get("inlet", ()) or ())
        outlets = tuple(roles.get("outlet", ()) or ())
        if only_inlet_to_outlet and not (
            any(n in graph for n in inlets) and any(n in graph for n in outlets)
        ):
            connectivity_status.setText(
                "This network has no inlet or no outlet yet, so there is nothing "
                "between them to export; run through Boundaries first."
            )
            return None
        rows = connectivity_rows(
            graph,
            inlets,
            outlets,
            arteriole_boundary_nodes=tuple(roles.get("arteriole_boundary", ()) or ()),
            venule_boundary_nodes=tuple(roles.get("venule_boundary", ()) or ()),
            only_inlet_to_outlet=only_inlet_to_outlet,
        )
        try:
            written = write_connectivity_csv(path, rows)
        except OSError as error:
            connectivity_status.setText(f"Could not write {path}: {error.strerror or error}")
            return None
        state.connectivity_csv = written
        which = (
            f"{len(rows)} of {graph.number_of_edges()} vessels (inlet to outlet only)"
            if only_inlet_to_outlet else f"all {len(rows)} vessels"
        )
        stale = (
            " Edits not yet regenerated: their lengths and diameters may be out of date."
            if has_pending_edits(graph) else ""
        )
        log_edit(f"Exported the connectivity of {which} to {written}")
        connectivity_status.setText(f"Wrote the connectivity of {which} to {written}.{stale}")
        report.value = f"Connectivity CSV written to {written}."
        return written

    def open_connectivity_map(csv_path=None) -> Path | None:
        """Draw a connectivity CSV as a 2D map and open it in the browser.

        *csv_path* defaults to the CSV exported last, else one asked for.
        """
        import tempfile
        import webbrowser

        from haemolynx.visualization.connectivity_map import write_connectivity_map

        source = csv_path or state.connectivity_csv
        if source is None:
            from qtpy.QtWidgets import QFileDialog

            source, _filter = QFileDialog.getOpenFileName(
                _dialog_parent(viewer), "Draw a connectivity CSV", "", "CSV (*.csv)"
            )
            if not source:
                return None
        source = Path(source)
        try:
            try:
                html_path = write_connectivity_map(source)
            except OSError:
                html_path = write_connectivity_map(
                    source, Path(tempfile.gettempdir()) / f"{source.stem}_map.html"
                )
        except Exception as error:  # noqa: BLE001 - a bad file must not crash the panel
            connectivity_status.setText(f"Could not draw {source}: {error}")
            return None
        webbrowser.open(html_path.resolve().as_uri())
        connectivity_status.setText(f"Opened the 2D connectivity map: {html_path}")
        return html_path

    def on_export_connectivity() -> None:
        from qtpy.QtWidgets import QFileDialog

        values = None
        if settings is not None:
            try:
                values = settings()
            except Exception:  # noqa: BLE001 - only the suggested filename depends on it
                values = None
        path, _filter = QFileDialog.getSaveFileName(
            _dialog_parent(viewer),
            "Export network connectivity",
            default_connectivity_csv_path(values),
            "CSV (*.csv)",
        )
        if path:
            export_connectivity(path)

    def on_connectivity_map() -> None:
        open_connectivity_map()

    def on_delete() -> None:
        if state.graph is None or state.node is None:
            status.setText("Scan the network and pick a junction first.")
            return
        rows = selected_rows()
        if not rows:
            status.setText("Select one or more vessels in the table first.")
            return
        node = state.node
        edges = [state.vessels[r].edge for r in rows]
        degree = state.graph.degree(node)
        described = describe_vessels(state.graph, edges)
        try:
            delete_vessels(state.graph, edges, protected=state.protected)
        except ValueError as error:
            status.setText(str(error))
            log_edit(f"Node {node} ({degree} vessels): delete refused, {error}")
            return
        log_edit(f"Node {node} ({degree} vessels): deleted {described}")
        drop_trace_after_edit()
        rescan(prefer=node)
        report.value = (
            f"Post processing: deleted {len(edges)} vessel(s) at node {node}. "
            f"{state.scan.summary}"
        )

    def on_split() -> None:
        if state.graph is None or state.node is None:
            status.setText("Scan the network and pick a junction first.")
            return
        node = state.node
        degree = state.graph.degree(node)
        length = float(connector.value())
        try:
            new_nodes = split_junction(
                state.graph,
                node,
                connector_length_um=length,
                reserved_ids=state.protected,
            )
        except ValueError as error:
            status.setText(str(error))
            log_edit(f"Node {node} ({degree} vessels): split refused, {error}")
            return
        moves = []
        for new in new_nodes:
            ends = sorted((o for _, o in state.graph.edges(new) if o != node), key=str)
            moves.append(
                f"vessels to node(s) {', '.join(map(str, ends))} moved to new node {new}"
            )
        diameter = next(
            (d.get("diameter_um") for n in new_nodes[:1] for d in state.graph.get_edge_data(node, n).values()),
            None,
        )
        size = f", {diameter:.3g} µm across" if diameter is not None else ""
        log_edit(
            f"Node {node} ({degree} vessels): split into bifurcations with "
            f"{length:g} µm connector(s){size}; {'; '.join(moves)}"
        )
        drop_trace_after_edit()
        rescan()
        report.value = (
            f"Post processing: split node {node} into bifurcations "
            f"(new node(s) {', '.join(str(n) for n in new_nodes)}). {state.scan.summary}"
        )

    def on_prune() -> None:
        if state.graph is None:
            status.setText("Scan the network first.")
            return
        if not state.inlets or not state.outlets:
            status.setText(
                "This run has no inlet or no outlet nodes, so there is nothing to "
                "tell a connected branch from a disconnected one."
            )
            return
        try:
            pruned, stats = prune_disconnected_branches(state.graph, state.inlets, state.outlets)
        except ValueError as error:
            status.setText(str(error))
            log_edit(f"Prune refused, {error}")
            return
        if not stats["removed_vessels"] and not stats["removed_nodes"]:
            status.setText("Nothing to prune: every vessel is on an inlet-to-outlet path.")
            log_edit("Prune: nothing to remove, every vessel is on an inlet-to-outlet path")
            return
        removed = [e for e in edge_keys(state.graph) if not pruned.has_edge(*e)]
        lost = stats["removed_boundary_nodes"]
        what = (
            f"{stats['removed_components']} disconnected piece(s) and "
            f"{stats['removed_dead_end_vessels']} dead-end vessel(s), "
            f"{stats['removed_vessels']} vessel(s)"
        )
        log_edit(
            f"Pruned {what}: {describe_vessels(state.graph, removed)}"
            + (f"; boundary node(s) removed with them: {', '.join(map(str, lost))}" if lost else "")
        )
        stop_editing()
        state.graph = pruned
        state.inlets = tuple(n for n in state.inlets if n in pruned)
        state.outlets = tuple(n for n in state.outlets if n in pruned)
        rescan(prefer=state.node)
        dropped = stats["removed_boundary_nodes"]
        extra = (
            f", with boundary node(s) {', '.join(str(n) for n in dropped)}" if dropped else ""
        )
        report.value = f"Post processing: pruned {what}{extra}. {state.scan.summary}"

    def on_leave() -> None:
        if state.scan is None or state.node is None:
            return
        state.decisions[state.node] = "left as is"
        log_edit(f"Node {state.node} ({state.graph.degree(state.node)} vessels): left as is")
        fill_list()

    # --- editing by clicking in the viewer ------------------------------------

    def set_edit_status(key: str) -> None:
        edit_status.setText(_POST_PROCESSING_EDIT_STATUS.get(key, key))

    def show_mode() -> None:
        """Light the button of the mode clicks are in, and only that one."""
        for button, mode in ((click_delete_button, "delete"), (add_button, "add")):
            button.blockSignals(True)
            button.setChecked(state.mode == mode)
            button.blockSignals(False)

    def drop_trace() -> None:
        """Forget the vessel being traced; the network never had it."""
        if state.trace is not None:
            state.trace = None
            draw_trace()

    def drop_trace_after_edit() -> None:
        """Another edit changed the network under a trace: start it again."""
        if state.trace is None:
            return
        drop_trace()
        set_edit_status(
            "The network changed, so the vessel being traced was dropped: click "
            "where to start it again."
        )

    def stop_editing() -> None:
        state.mode = "idle"
        drop_trace()
        show_mode()
        set_edit_status("idle")

    def attach_click_callbacks() -> None:
        """(Re-)attach to the vessels/nodes layers, which a rebuild can swap,
        and to the tab's own layers, in case one of them is selected."""
        for name in (VESSELS, NODES, *POST_PROCESSING_LAYERS):
            target = layer(name)
            if target is not None and on_press not in target.mouse_drag_callbacks:
                target.mouse_drag_callbacks.append(on_press)

    def arm(mode: str) -> None:
        if state.graph is None:
            show_mode()
            set_edit_status("Scan the network first.")
            return
        if state.mode == mode:
            # The lit button pressed again: stop, dropping an unfinished vessel.
            tracing = state.trace is not None
            stop_editing()
            if tracing:
                set_edit_status("Add vessel cancelled: the network is as it was before.")
            return
        drop_trace()
        state.mode = mode
        attach_click_callbacks()
        # napari hands a click to the selected layer: make it one that listens.
        vessels = layer(VESSELS)
        if vessels is not None:
            viewer.layers.selection.active = vessels
        show_mode()
        set_edit_status(f"{mode}-armed")

    def voxel_size() -> tuple[float, float, float]:
        return tuple(float(v) for v in getattr(results(), "_voxel_size_zyx", (1.0, 1.0, 1.0)))

    def read_settings():
        if settings is None:
            return None
        try:
            return settings()
        except Exception:  # noqa: BLE001 - the defaults do without them
            logger.exception("post processing could not read the settings")
            return None

    def smoothing_settings():
        """The run's centreline smoothing, for a traced vessel: None when the run
        does not smooth, the pipeline's defaults when there are no settings."""
        local = read_settings()
        if local is None:
            return {}
        if not local.get("smooth_centrelines", True):
            return None
        return {
            "method": local.get("centreline_smoothing_method", "taubin"),
            "iterations": int(local.get("centreline_smoothing_iterations", 10)),
            "max_deviation": float(local.get("centreline_max_deviation", 1.0)),
        }

    def mask_routing():
        """The segmented image and its cost field, made once per image."""
        image = layer(IMAGE)
        if image is None:
            return None, None
        data = image.data
        if state.routing is None or state.routing[0] is not data:
            set_edit_status("Preparing the segmented image to trace new vessels through...")
            _process_pending_qt_events()
            state.routing = (
                data,
                mask_cost_field(np.asanyarray(data), use_memmap=True, voxel_size_zyx=voxel_size()),
            )
        return state.routing

    def load_raw():
        """The run's Raw data file: ``(volume, None)``, or ``(None, why not)``."""
        from haemolynx.haemodynamics.automated import load_single_channel_tiff_volume
        from haemolynx.io import resolve_image_path_with_optional_zip

        local = read_settings() or {}
        path = local.get("fwhm_raw_tiff_path")
        if not path:
            return None, "no Raw data file is set on the Input tab"
        set_edit_status("Reading the raw data file to trace new vessels through...")
        _process_pending_qt_events()
        channel = local.get("fwhm_raw_channel")
        try:
            raw = load_single_channel_tiff_volume(
                resolve_image_path_with_optional_zip(Path(path)),
                axis_order=local.get("image_axis_order") or "zyx",
                use_memmap=bool(local.get("use_memmap_loading")),
                memmap_directory=local.get("memmap_directory"),
                channel=None if channel is None else int(channel),
            )
        except Exception as error:  # noqa: BLE001 - reported; the mask is used instead
            return None, f"could not read {path}: {error}"
        image = layer(IMAGE)
        if image is not None and tuple(np.shape(raw)) != tuple(np.shape(image.data)):
            return None, (
                f"the raw data is {tuple(np.shape(raw))} voxels but the segmented "
                f"image {tuple(np.shape(image.data))}"
            )
        return raw, None

    def raw_routing():
        """The raw data and its cost field -- FWHM's own copy when it is in the
        viewer, else the Raw data file, read once -- and why not, if not."""
        fwhm = layer(FWHM_RAW)
        if fwhm is not None:
            data = fwhm.data
            if state.raw is None or state.raw[0] is not data:
                state.raw = (data, IntensityCostField(data))
            return state.raw, None
        if state.raw is None and state.raw_problem is None:
            data, state.raw_problem = load_raw()
            if data is not None:
                state.raw = (data, IntensityCostField(data))
        return state.raw, state.raw_problem

    def trace_route():
        """How the next leg of a trace is found, as the dropdown asks.

        ``cost`` is the A* cost field (None: a straight line), ``volume`` and
        ``pick`` how a 3D click's depth is read, ``how`` the words for it, and
        ``note`` why it is not what the dropdown asked, if it is not.
        """
        note = ""
        if trace_source.currentText() == TRACE_THROUGH_RAW:
            raw, problem = raw_routing()
            if raw is not None:
                return SimpleNamespace(
                    cost=raw[1], volume=raw[0], pick="brightest",
                    how="through the raw data", note="",
                )
            note = f" (no raw data to trace through: {problem})"
        mask, cost = mask_routing()
        if cost is not None:
            return SimpleNamespace(
                cost=cost, volume=mask, pick="first", how="through the segmented mask", note=note,
            )
        return SimpleNamespace(
            cost=None, volume=None, pick="first", how="as a straight line",
            note=note or " (no segmented image in the viewer to trace through)",
        )

    def vessel_hit(event):
        from haemolynx.gui.graph_click import hit_test_vessels

        vessels = layer(VESSELS)
        if vessels is None:
            return None
        return hit_test_vessels(
            vessels.data, _layer_features(vessels), event.position,
            view_direction=getattr(event, "view_direction", None),
            dims=list(getattr(event, "dims_displayed", ()) or ()),
        )

    def end_under_click(event, position, view, dims):
        """The node, or the point on a vessel, a click landed on -- or None."""
        node = nearest_node(state.graph, position, view_direction=view, dims=dims or None)
        if node is not None:
            return VesselEnd.at_node(node)
        hit = vessel_hit(event)
        if hit is None:
            return None
        edge = tuple(c.item() if isinstance(c, np.generic) else c for c in (hit.u, hit.v, hit.key))
        if not state.graph.has_edge(*edge):
            return None
        point = point_on_path_under_click(
            vessel_path(state.graph, edge), position, view_direction=view, dims=dims or None
        )
        return VesselEnd.on_vessel(edge, point)

    def delete_clicked(hit) -> None:
        edge = (hit.u, hit.v, hit.key)
        described = describe_vessels(state.graph, [edge])
        try:
            delete_vessels(state.graph, [edge], protected=state.protected)
        except ValueError as error:
            set_edit_status(str(error))
            log_edit(f"Delete of {described} (clicked) refused, {error}")
            return
        log_edit(f"Deleted {described} (clicked in the viewer)")
        rescan(prefer=state.node)
        set_edit_status("deleted")

    def add_clicked(event) -> None:
        """One click of Add vessel: start, trace on to a point, or finish."""
        position = np.asarray(event.position, dtype=float)
        dims = list(getattr(event, "dims_displayed", ()) or ())
        view = getattr(event, "view_direction", None) if len(dims) == 3 else None
        end = end_under_click(event, position, view, dims)
        if state.trace is None:
            if end is None:
                set_edit_status(
                    "Add vessel: that missed. Click a node, or a point on a vessel, "
                    "to start from."
                )
                return
            state.trace = VesselTrace.starting_at(end, state.graph)
            draw_trace()
            where = (
                f"a new node on {branch_label(end.edge)} (green; it forms when the "
                "vessel is finished)"
                if end.on_a_vessel
                else f"node {end.node}"
            )
            set_edit_status(
                f"Starting at {where}. Click along the vessel to trace it; click "
                "another node or vessel to finish it."
            )
            return
        route = trace_route()
        if end is not None:
            finish_trace(end, route)
            return
        target = point_under_click(
            position, view_direction=view, dims=dims or None, volume=route.volume,
            voxel_size_zyx=voxel_size(), pick=route.pick, near_um=state.trace.last_point,
        )
        leg = trace_path(route.cost, state.trace.last_point, target, voxel_size_zyx=voxel_size())
        if len(leg) < 2:
            set_edit_status("The trace is already there: click further along the vessel.")
            return
        state.trace.extend(leg, target, route.how)
        draw_trace()
        set_edit_status(
            f"Point {len(state.trace.waypoints_um)} traced {route.how}{route.note}. "
            "Click on along the vessel, or on a node or vessel to finish it; press "
            "Add vessel again to cancel."
        )

    def branch_label(edge) -> str:
        branch_id = branch_id_of(state.graph, edge)
        if branch_id is None:
            return f"the vessel {edge[0]}-{edge[1]}"
        return f"branchID {branch_id}"

    def finish_trace(end, route) -> None:
        """The click landed on a node or vessel: add the traced vessel, ending there."""
        graph, trace = state.graph, state.trace
        leg = trace_path(
            route.cost, trace.last_point, end.position(graph), voxel_size_zyx=voxel_size()
        )
        cut = [e.edge for e in (trace.start, end) if e.on_a_vessel]
        described_cut = describe_vessels(graph, cut)
        try:
            added = add_traced_vessel(
                graph, trace.start, end, trace.extended(leg),
                reserved_ids=state.protected, voxel_size_zyx=voxel_size(),
                smoothing=smoothing_settings(),
            )
        except ValueError as error:
            set_edit_status(f"{error}. Or press Add vessel again to cancel.")
            return
        routes = " then ".join(dict.fromkeys([*trace.routes, route.how]))
        clicks = len(trace.waypoints_um) + 2
        state.added_nodes.extend(added.new_nodes)
        state.trace = None
        state.mode = "idle"
        show_mode()
        draw_trace()
        new_nodes = ", ".join(str(n) for n in added.new_nodes)
        log_edit(
            f"Added {describe_vessels(graph, [added.edge])}, traced {routes} in {clicks} clicks"
            + (f"; new node(s) {new_nodes} cut {described_cut} in two" if added.new_nodes else "")
        )
        rescan(prefer=state.node)
        u, v, _key = added.edge
        size = (
            f"{added.diameter_um:.3g} µm across" if added.diameter_um is not None
            else "the table's diameter"
        )
        set_edit_status(
            f"Added {branch_label(added.edge)}: {graph.edges[added.edge]['length']:.3g} µm "
            f"from node {u} to node {v}, traced {routes}{route.note}, {size}."
            + (
                f" New node(s) {new_nodes} (green) cut the vessel(s) it joined in two."
                if added.new_nodes else ""
            )
            + " Press Add vessel to add another."
        )

    def on_delete_ids() -> None:
        if state.graph is None:
            set_edit_status("Scan the network first.")
            return
        keys = edge_keys(state.graph)
        try:
            ids = parse_branch_ids(branch_ids.text(), len(keys))
        except ValueError as error:
            set_edit_status(str(error))
            return
        described = describe_vessels(state.graph, [keys[i] for i in ids])
        try:
            delete_vessels(state.graph, [keys[i] for i in ids], protected=state.protected)
        except ValueError as error:
            set_edit_status(str(error))
            log_edit(f"Delete by branch ID of {described} refused, {error}")
            return
        log_edit(f"Deleted by branch ID: {described}")
        drop_trace_after_edit()
        branch_ids.clear()
        rescan(prefer=state.node)
        set_edit_status(
            f"Deleted branchID(s) {', '.join(str(i) for i in ids)}. branchIDs of the "
            "vessels after them have moved down: hover a vessel for its new one."
        )

    def on_click(_layer, event) -> None:
        """One click in the viewer: a press let go where it was pressed."""
        if state.mode == "idle" or state.graph is None:
            return
        try:
            if state.mode == "delete":
                hit = vessel_hit(event)
                if hit is not None:
                    delete_clicked(hit)
            else:
                add_clicked(event)
        except Exception as error:  # noqa: BLE001 - a click must not crash the viewer
            logger.exception("post-processing click failed")
            set_edit_status(f"error: {error}")

    def on_press(layer_, event):
        """napari's press-drag-release callback, handing :func:`on_click` only clicks.

        A press that moves before it is let go is turning or panning the view --
        between the clicks of a trace, often -- and must not add a point or
        delete a vessel.
        """
        if state.mode == "idle" or state.graph is None:
            return
        if getattr(event, "button", 1) not in (None, 1):
            return
        pressed = getattr(event, "pos", None)
        pressed = None if pressed is None else np.asarray(pressed, dtype=float)
        dragged = False
        yield
        while getattr(event, "type", None) == "mouse_move":
            pos = getattr(event, "pos", None)
            if (
                pressed is not None
                and pos is not None
                and float(np.linalg.norm(np.asarray(pos, dtype=float) - pressed))
                > _CLICK_DRAG_TOLERANCE_PX
            ):
                dragged = True
            yield
        if not dragged:
            on_click(layer_, event)

    # --- leaving -----------------------------------------------------------------

    def remove_layers() -> None:
        if viewer is None:
            return
        for name in POST_PROCESSING_LAYERS:
            if name in viewer.layers:
                _process_pending_qt_events()
                viewer.layers.remove(viewer.layers[name])
                _process_pending_qt_events()
        for name in (VESSELS, NODES):
            target = layer(name)
            if target is not None and on_press in target.mouse_drag_callbacks:
                target.mouse_drag_callbacks.remove(on_press)

    def hand_over(graph, *, stop_after, note: str, waiting: str) -> None:
        """Start the run that takes *graph* on, and clear the tab for it.

        The tab keeps its edits when the run does not start (a failed check
        says why in the report box): nothing is lost to a refused run.
        """
        stop_editing()
        if not regenerate(graph, stop_after=stop_after):
            refresh_buttons()
            return
        log_edit(note)
        remove_layers()
        state.graph = state.scan = state.node = None
        state.vessels = []
        state.added_nodes = []
        junction_list.clear()
        table.setRowCount(0)
        status.setText(waiting)
        refresh_buttons()

    def on_regenerate() -> None:
        """After a finished run: the edits in line, Haemodynamics rerun on
        them, then Perturbations to Export."""
        if state.graph is None:
            status.setText("Scan the network first; there is nothing edited to regenerate from.")
            return
        if running():
            report.value = ALREADY_RUNNING
            return
        hand_over(
            state.graph,
            stop_after=None,
            note=(
                f"Regenerating from the edited network: the edits in line, "
                f"Haemodynamics rerun on it, then Perturbations to Export "
                f"({state.graph.number_of_edges()} vessels)"
            ),
            waiting="Regenerating from the edited network; scan again once it finishes.",
        )

    def on_regenerate_graph() -> None:
        """The edits in line and Haemodynamics rerun; paused, the run stays
        paused, else it goes on to Export."""
        if state.graph is None or not has_pending_edits(state.graph):
            status.setText("No outstanding edits to regenerate: the network is in line.")
            return
        if running():
            report.value = ALREADY_RUNNING
            return
        held = is_paused()
        hand_over(
            state.graph,
            stop_after=regenerate_stop_after(held),
            note=(
                f"Regenerating the graph ({state.graph.number_of_edges()} vessels): "
                "branch orders, lengths, diameters and bridges of the edited vessels, "
                "then Haemodynamics rerun on it"
                + ("; the run stays paused" if held else "; then Perturbations to Export")
            ),
            waiting=(
                "Regenerating the graph; the tab scans it again once it is done."
                if held
                else "Regenerating the graph and re-solving; scan again once it finishes."
            ),
        )

    def on_continue() -> None:
        """A paused run: the edits (if any) in line with Haemodynamics rerun
        on them, then Perturbations onwards."""
        if not is_paused():
            status.setText(
                "No run is paused here. Turn on Mid-run postprocessing (1. Input) "
                "for a run to stop here after Haemodynamics."
            )
            return
        if running():
            report.value = ALREADY_RUNNING
            return
        graph = state.graph
        if graph is None:
            current = results()
            graph = getattr(current, "_graph", None) if current is not None else None
        if graph is None:
            status.setText("Nothing to continue from: run the pipeline again.")
            return
        hand_over(
            graph,
            stop_after=None,
            note=(
                f"Continuing the run from the edited network ({graph.number_of_edges()} "
                "vessels): the edits in line and Haemodynamics rerun on them, then "
                "Perturbations onwards"
            ),
            waiting="Continuing the run; scan again once it finishes.",
        )

    scan_button.clicked.connect(on_scan)
    junction_toggle.toggled.connect(on_junction_correction_toggled)
    junction_list.currentRowChanged.connect(on_junction_changed)
    table.itemSelectionChanged.connect(recolour)
    delete_button.clicked.connect(on_delete)
    split_button.clicked.connect(on_split)
    leave_button.clicked.connect(on_leave)
    prune_button.clicked.connect(on_prune)
    export_connectivity_button.clicked.connect(on_export_connectivity)
    connectivity_map_button.clicked.connect(on_connectivity_map)
    click_delete_button.clicked.connect(lambda: arm("delete"))
    add_button.clicked.connect(lambda: arm("add"))

    def on_trace_source_changed(_text) -> None:
        # Choosing again retries raw data that could not be had (a Raw data
        # file set since, say); raw data already read is kept.
        state.raw_problem = None

    trace_source.currentTextChanged.connect(on_trace_source_changed)
    delete_ids_button.clicked.connect(on_delete_ids)
    branch_ids.returnPressed.connect(on_delete_ids)
    regenerate_button.clicked.connect(on_regenerate)
    regenerate_graph_button.clicked.connect(on_regenerate_graph)
    continue_button.clicked.connect(on_continue)
    # Off until there is something to take on; the panel refreshes them once
    # what `running`, `paused` and `complete` read exists.
    for button in (regenerate_graph_button, continue_button, regenerate_button):
        button.setEnabled(False)

    def forget() -> None:
        """The panel was cleared: drop the working graph and its edits."""
        stop_editing()
        state.graph = state.scan = state.node = None
        state.vessels = []
        state.added_nodes = []
        state.decisions = {}
        state.raw = state.raw_problem = None
        junction_list.clear()
        table.setRowCount(0)
        status.setText("Not scanned yet.")
        refresh_buttons()

    return SimpleNamespace(
        page=page,
        state=state,
        scan=on_scan,
        refresh=refresh_buttons,
        forget=forget,
        regenerate_graph_button=regenerate_graph_button,
        continue_button=continue_button,
        commit_status=commit_status,
        scan_button=scan_button,
        junction_toggle=junction_toggle,
        junction_box=junction_box,
        junction_list=junction_list,
        table=table,
        delete_button=delete_button,
        split_button=split_button,
        connector=connector,
        leave_button=leave_button,
        click_delete_button=click_delete_button,
        add_button=add_button,
        trace_source=trace_source,
        branch_ids=branch_ids,
        delete_ids_button=delete_ids_button,
        prune_button=prune_button,
        export_connectivity_button=export_connectivity_button,
        export_connectivity=export_connectivity,
        connectivity_box=connectivity_box,
        connectivity_choice=connectivity_choice,
        connectivity_map_button=connectivity_map_button,
        connectivity_status=connectivity_status,
        open_connectivity_map=open_connectivity_map,
        regenerate_button=regenerate_button,
        log_box=log_box,
        status=status,
        edit_status=edit_status,
        on_click=on_click,
        on_press=on_press,
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
    from magicgui.widgets import (
        CheckBox,
        ComboBox,
        Container,
        FileEdit,
        Label,
        PushButton,
        SpinBox,
        TextEdit,
    )
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (
        QGroupBox,
        QHBoxLayout,
        QSizePolicy,
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
    tab_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    tab_widget.setMinimumHeight(120)

    report = TextEdit(value="Ready.")
    report.read_only = True
    # QTextEdit's default hint is a multi-line slab; the status line should not
    # be what pushes the dock taller than a 1080p work area.
    report.native.setMinimumHeight(48)
    report.native.setMaximumHeight(96)

    # Two passes. Every row has to exist before anything that reads them can be
    # built, and the boundary controls read them -- so rows first, pages second.
    for tab in tabs:
        for field in tab.fields:
            rows[field.name] = _build_row(field)
            fields[field.name] = field

    # A channel drop-down lists the channels of the file its image setting
    # names, and follows it as that changes.
    for channel_name, path_name in CHANNEL_SETTINGS.items():
        if channel_name in rows and path_name in rows:
            _follow_image_channels(rows[channel_name], rows[path_name])

    #: Every Advanced button on the panel, by key (see haemolynx.gui.layout).
    advanced_disclosures: dict[str, _AdvancedDisclosure] = {}

    # Shared ilastik knobs are one block -- the ordinary rows, then their own
    # Advanced button -- that place_shared_ilastik moves between Input and
    # Boundaries. The block stays hidden while it has no parent: a visible
    # widget with no parent is a floating top-level window.
    shared_ilastik_names = [name for name in SHARED_ILASTIK_SETTINGS if name in rows]
    shared_ilastik_rows = Container(
        widgets=[rows[n] for n in shared_ilastik_names if not schema[n].advanced],
        labels=True,
    )
    _flush(shared_ilastik_rows)
    shared_ilastik_advanced = _AdvancedDisclosure(
        shared_ilastik_disclosure(
            [n for n in shared_ilastik_names if schema[n].advanced]
        ),
        rows,
        always_visible=True,
    )
    advanced_disclosures[shared_ilastik_advanced.spec.key] = shared_ilastik_advanced
    shared_ilastik_block = Container(
        widgets=[shared_ilastik_rows, shared_ilastik_advanced.container], labels=False
    )
    _flush(shared_ilastik_block)
    shared_ilastik_block.visible = False

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

    # FORCED_HIDDEN_SETTINGS: never parented as a flat tab row for
    # the same reason as the perturbation rows above -- park them in the
    # same hidden holder rather than leaving them as a floating window.
    for name in FORCED_HIDDEN_SETTINGS:
        if name in rows:
            rows[name].visible = False
            orphan_holder.append(rows[name])

    # FWHM and the endothelial measurement are alternatives, so the Diameters
    # tab shows one "Diameter measurement" choice where their two checkboxes
    # were (see haemolynx.gui.diameter_source). The checkboxes still hold the
    # two settings a config file saves -- parked, never shown -- and the
    # choice writes them.
    diameter_source = ComboBox(
        label=DIAMETER_SOURCE_LABEL,
        choices=list(DIAMETER_SOURCES),
        value=source_from({name: rows[name].value for name in DIAMETER_SOURCE_SETTINGS}),
    )
    from haemolynx.gui.chrome_tooltips import DIAMETER_SOURCE_TOOLTIP

    diameter_source.tooltip = DIAMETER_SOURCE_TOOLTIP
    diameter_source.native.setObjectName("haemolynx_diameter_source")
    for name in DIAMETER_SOURCE_SETTINGS:
        rows[name].visible = False
        orphan_holder.append(rows[name])
    #: The first checkbox's place takes the choice; the second's takes nothing.
    diameter_source_places = {
        DIAMETER_SOURCE_SETTINGS[0]: diameter_source,
        DIAMETER_SOURCE_SETTINGS[1]: None,
    }

    #: Stages that lay their own page out, keyed by the stage function they
    #: belong to rather than by the tab's title, so renaming a tab cannot
    #: silently drop them. Any future stage-specific page has a home here.
    pages: dict[str, Any] = {}
    boundaries = _boundary_controls(
        viewer, rows, fields, schema, report,
        # `checkpoints` is created below; read only when a box is listed.
        boundaries_input=lambda: getattr(checkpoints.get("build_network"), "graph", None),
    )
    if boundaries is not None:
        pages["assign_boundaries"] = boundaries.page
    perturbations = _perturbation_controls(viewer, rows, fields, schema, report)
    if perturbations is not None:
        pages["run_perturbations"] = perturbations.page
    # Hand edits between Haemodynamics and Perturbations: the post_process stage's
    # page, which has no setting rows of its own. Its callables read `view`,
    # `checkpoints` and `run_state` when clicked, all of which exist by then.
    post_processing = _post_processing_controls(
        viewer,
        report,
        results=lambda: view.results,
        boundary_roles=lambda: checkpoints._carried_boundary_roles(),
        regenerate=lambda graph, stop_after=None: regenerate_from_graph(
            graph, from_post_processing=True, stop_after=stop_after
        ),
        running=lambda: run_state.running,
        settings=lambda: _settings(),
        paused=lambda: run_state.paused,
        complete=lambda: not run_state.paused and checkpoints.has(POST_PROCESS),
    )
    pages[POST_PROCESS] = lambda _summary, _names: post_processing.page

    #: Snapshots from the last run that showed layers: what "Run from this
    #: stage" on a tab needs from the previous tab. Cleared when the layers
    #: are, and replaced when a new run that shows layers starts.
    checkpoints = StageCheckpoints()
    revert_buttons: dict[str, Any] = {}
    from haemolynx.gui.chrome_tooltips import REVERT_STAGE_TOOLTIP

    #: The Input tab's boxes by key, so the chrome built further down (the
    #: check and optimise buttons) can join the box it belongs to.
    input_boxes: dict[str, Any] = {}
    #: Input-tab holder that receives the shared ilastik block when main
    #: segmentation uses ilastik. Boundaries gets a holder of its own.
    input_shared_ilastik_holder: Any = None
    #: Diameters-tab container, so "Optimise FWHM settings" (built further
    #: down, once its own click handler exists) can be inserted right after
    #: the fwhm_raw_tiff_path row.
    diameters_settings: Any = None
    #: Which tab currently parents the shared ilastik rows.
    shared_ilastik_placement: dict[str, str | None] = {"host": None}
    #: Predeclared so `apply_prerequisites` (defined below, but already
    #: called once before these are built) can safely check them on its very
    #: first call -- see the `thick_vessel_row` guard just below for the
    #: same "not built yet" pattern.
    optimise_fwhm_button: Any = None
    #: The FWHM optimiser's options and its review box, shown with the button.
    fwhm_optimise_box: Any = None
    #: A nested QGroupBox (e.g. "Connectivity/Network Analysis" on "8.
    #: Additional measurements") boxes a run of settings that all share one
    #: `requires` gate on the section above it. Left always visible, an
    #: unmet gate hides every row inside but not the box itself, so a user
    #: sees an empty frame with a title and nothing in it rather than the
    #: box simply not being there -- apply_prerequisites (below) hides the
    #: box itself once none of its own rows are visible.
    section_group_boxes: dict[str, tuple[Any, list[str]]] = {}

    for tab in tabs:
        summary = Label(value=tab.stage.summary)
        build = pages.get(tab.stage.call or "")
        # Shared ilastik knobs start unparented; place_shared_ilastik hosts
        # them. FORCED_HIDDEN_SETTINGS are parked in orphan_holder.
        names = [
            field.name
            for field in tab.fields
            if field.name not in SHARED_ILASTIK_SETTING_SET
            and field.name not in FORCED_HIDDEN_SETTINGS
        ]
        if build is not None:
            native = build(summary, names)
        else:
            # Boxes and Advanced buttons are decided by haemolynx.gui.layout:
            # a later section on a tab is a nested checkbox's own children
            # (e.g. "Connectivity/Network Analysis" under "Statistics and
            # measurements") and gets a group box of its own, and every
            # advanced row goes behind a button under the row it belongs to.
            native, containers = _lay_out_boxes(
                layout_for(tab.stage.call, names, schema),
                rows,
                disclosures=advanced_disclosures,
                group_boxes=section_group_boxes,
                lead=[summary],
                replacements=diameter_source_places,
            )
            if tab.stage.call == "segment":
                input_boxes = containers
                input_shared_ilastik_holder = Container(widgets=[], labels=False)
                _flush(input_shared_ilastik_holder)
                leading_input = containers.get("input")
                if leading_input is not None:
                    leading_input.append(input_shared_ilastik_holder)
                    _hide_row_label(input_shared_ilastik_holder)
            elif tab.stage.call == "assign_diameters":
                diameters_settings = containers.get("diameters")
        # A bounded QScrollArea rather than `Container(scrollable=True)`: the
        # magicgui one reports the full height of its contents, so a tab with
        # 39 rows stretches the whole napari window instead of scrolling.
        # Qt's QScrollArea.sizeHint *adds* the inner widget's hint, which on
        # Windows becomes the dock's minimum height -- taller than a 1080p
        # work area, then QWindowsWindow::setGeometry warns and clamps.
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(native)
        if tab.stage.call == "export_results":
            # Post processing owns the button (it exports the network edited
            # there, if any); it sits here with the run's other exports.
            page_layout.addWidget(post_processing.connectivity_box)
        # "Run from this stage" lives in shared chrome below "Show each
        # topology step", not inside the tab page: one button per tab that
        # has a predecessor, shown for the active tab and centered on the
        # panel. After a full run it drops this tab and later work, then
        # reruns from here using the previous tab's checkpoint.
        if previous_tab(tab.stage.title) is not None and tab.stage.call != POST_PROCESS:
            revert = PushButton(text="Run from this stage")
            revert.enabled = False
            revert.tooltip = REVERT_STAGE_TOOLTIP
            revert.native.setObjectName("haemolynx_revert")
            revert_buttons[tab.stage.title] = revert
        page_layout.addStretch(1)
        scroller = _fitting_scroll_area()
        scroller.setWidget(page)
        tab_widget.addTab(scroller, tab.stage.title)
        if tab.stage.call:
            index = tab_widget.count() - 1
            tab_widget.setTabToolTip(index, f"{tab.stage.call}(settings, ...)")

    # The pages that lay themselves out keep their own Advanced buttons and
    # boxes; the panel refreshes them with everything else.
    for controls in (boundaries, perturbations):
        if controls is None:
            continue
        advanced_disclosures.update(controls.disclosures())
        section_group_boxes.update(getattr(controls, "group_boxes", dict)())

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
    #: Directory of the last loaded or saved config, for relative-path rebasing.
    loaded_config_dir: list[Path | None] = [None]
    #: Last path the Load/Save dialogues used, so they reopen in the same place.
    last_config_path: list[str] = ["config.yaml"]
    #: Last path the Save run / Load run dialogues used.
    last_run_path: list[str] = [RUN_SNAPSHOT_FILENAME]

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
            name: fields[name].to_setting_value(_safe_widget_value(widget))
            for name, widget in rows.items()
        }
        for name, original in loaded_paths.items():
            current = values.get(name)
            if current is None:
                continue
            try:
                unchanged = Path(current) == _resolved_path(str(original))
            except (TypeError, ValueError, OSError):
                continue
            if unchanged:
                values[name] = original
        return values

    def use_layer(layer) -> None:
        """Point the run at *layer*: its own file, or its array written out."""
        axis_order = str(current_values().get("image_axis_order") or "zyx")
        try:
            chosen = input_for_layer(layer, _export_dir(current_values()), axis_order)
        except ValueError as error:
            report.value = str(error)
            return
        if chosen.needs_export:
            import tifffile

            target = Path(chosen.settings["input_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(target, data_for_pipeline(layer))
        for name, value in chosen.settings.items():
            if name in rows:
                rows[name].value = value
        # Kept so that opening a config afterwards does not silently point the
        # run back at whatever image that file was written for.
        adopted.settings = dict(chosen.settings)
        adopted.name = getattr(layer, "name", None)
        apply_prerequisites()
        note = chosen.note
        applied = _scale_layer_from_its_file(
            layer, chosen.settings.get("input_path"), axis_order
        )
        if applied is not None:
            note += (
                f" Scaled the layer to {tuple(round(v, 4) for v in applied)} "
                f"({', '.join(axis_order)}) microns, from the file, so it sits where "
                "the results will."
            )
        report.value = note

    def place_shared_ilastik() -> None:
        """Host the shared ilastik block on Input or Boundaries (same widgets).

        The block -- the shared rows and their own Advanced button -- is
        deliberately left out of the initial tab containers and moved later.
        A magicgui widget with no Qt parent that is set visible becomes a
        top-level window beside napari — the same failure ``place_shared``
        already guards against for boundary-method rows. Hide before detach,
        show only after a successful append, and never record a host that did
        not actually receive the block.
        """
        values = current_values()
        host = shared_ilastik_host(values)
        boundaries_holder = None
        if boundaries is not None:
            getter = getattr(boundaries, "shared_ilastik_holder", None)
            boundaries_holder = getter() if callable(getter) else getter

        if host == shared_ilastik_placement["host"]:
            # Already placed (or correctly unhosted). Do not poke ``visible``
            # here: setting True on an unparented block opens a floating window.
            return

        # Hide before remove: a visible widget with no parent is a window of
        # its own (see Boundaries ``place_shared``).
        shared_ilastik_block.visible = False
        for holder in (input_shared_ilastik_holder, boundaries_holder):
            if holder is not None and any(w is shared_ilastik_block for w in holder):
                holder.remove(shared_ilastik_block)

        attached: str | None = None
        if host == "input" and input_shared_ilastik_holder is not None:
            input_shared_ilastik_holder.append(shared_ilastik_block)
            attached = "input"
        elif host == "boundaries" and boundaries_holder is not None:
            boundaries_holder.append(shared_ilastik_block)
            attached = "boundaries"
        if attached is not None:
            shared_ilastik_block.visible = True

        shared_ilastik_placement["host"] = attached
        # Moving rows into a container re-unifies its label widths.
        _wrap_row_labels(tab_widget)

    #: Every gate above each row (see haemolynx.gui.form.prerequisite_chain).
    chains = {name: prerequisite_chain(schema, name) for name in fields}

    def chain_met(name: str, values: Mapping[str, Any]) -> bool:
        return all(is_prerequisite_met(rule, values) for rule in chains.get(name, ()))

    def row_shows(name: str, values: Mapping[str, Any]) -> bool:
        """Whether *name*'s row would show: hidden rows need their whole chain met."""
        field = fields[name]
        if not field.is_visible(values):
            return False
        return chain_met(name, values) if field.hide_when_unmet else True

    def apply_prerequisites(*_args) -> None:
        """Apply schema prerequisites: hide nested rows, grey others."""
        # Pinned before the values snapshot below, not inside the rows loop:
        # a dependent row reading a forced value to compute its own `enabled`
        # would otherwise use the stale, not-yet-forced value.
        values = current_values()
        for name in FORCED_HIDDEN_SETTINGS:
            forced = forced_hidden_value(name, values)
            if name in rows and rows[name].value != forced:
                rows[name].value = forced
        values = current_values()
        place_shared_ilastik()
        for name, widget in rows.items():
            if name in SHARED_ILASTIK_SETTING_SET:
                # Visibility and parent belong to place_shared_ilastik.
                widget.enabled = True
                widget.tooltip = fields[name].help
                continue
            if name in orphaned_perturbation_rows or name in DIAMETER_SOURCE_SETTINGS:
                # Never parented as flat tab rows; typed editors clone their
                # own widgets, and the Diameter measurement choice stands for
                # the two diameter checkboxes. Do not reveal — visible + no
                # parent is a window.
                widget.visible = False
                widget.enabled = True
                widget.tooltip = fields[name].help
                continue
            if name in FORCED_HIDDEN_SETTINGS:
                # Value already pinned above -- just keep it hidden.
                widget.visible = False
                widget.enabled = True
                widget.tooltip = fields[name].help
                continue
            field = fields[name]
            # The whole chain of gates above the row, not only its own.
            enabled = chain_met(name, values)
            if field.hide_when_unmet:
                # Input / Diameters / FWHM / Boundaries vessel / Graph
                # centreline options: only relevant nested knobs appear.
                widget.visible = field.is_visible(values) and enabled
                widget.enabled = True
                widget.tooltip = field.help
            else:
                widget.enabled = enabled
                widget.tooltip = (
                    field.help
                    if enabled
                    else replace(field, enabled_by=chains[name]).why_disabled(values)
                )

        for group, section_names in section_group_boxes.values():
            # Not `rows[name].visible`: magicgui reads that back through the
            # widget's own composite Qt visibility, which is masked by the
            # group's *current* (possibly still-hidden, from the previous
            # call) state -- reading it here would latch the group hidden
            # forever once hidden once. `field.is_visible` is the pure,
            # ancestor-independent answer the per-row loop above just used.
            group.setVisible(any(row_shows(name, values) for name in section_names))

        # Each Advanced button hides when none of its rows would show, and
        # counts the rows holding something other than where the panel started.
        for disclosure in advanced_disclosures.values():
            disclosure.refresh(
                lambda name: row_shows(name, values), values, schema, ADVANCED_BASELINE
            )
        if perturbations is not None:
            perturbations.refresh_entries()

        # The Diameter measurement choice shows while its settings apply.
        diameter_source.visible = chain_met(DIAMETER_SOURCE_SETTINGS[0], values)

        # Large-vessel-network mode relabels the thick-vessel checkbox: once
        # both are on, "thick vessel skeletonisation" is no longer the
        # user-facing action -- turning large vessels into network members
        # (Large_Art/Large_Ven) is. use_large_vessel_masks alone is not the
        # trigger: that flag also drives the long-standing cut-away
        # workflow, unrelated to this feature, so relabeling on it alone
        # would mislabel that ordinary case. Checking
        # use_thick_vessel_skeletonisation's own current value too (not
        # just trusting assign_large_vessel_branch_orders to stay
        # consistent, since schema `requires` does not enforce that at the
        # raw-value level) avoids showing the relabeled text for an
        # inconsistent stored config where the checkbox itself reads
        # unchecked.
        # "Optimise FWHM settings" (below the master checkbox and input file
        # line for FWHM): visible only while FWHM measurement is turned on --
        # a deliberate difference from "Optimise settings" on the Input tab,
        # which stays visible and only ever toggles `.enabled`. There is
        # nothing for this one to search once use_fwhm_edge_diameters is off,
        # so hiding it entirely (rather than greying it out) matches every
        # other `hide_when_unmet` FWHM row right next to it.
        if optimise_fwhm_button is not None:
            optimise_fwhm_button.visible = bool(values.get("use_fwhm_edge_diameters"))
        if fwhm_optimise_box is not None:
            fwhm_optimise_box.visible = bool(values.get("use_fwhm_edge_diameters"))

        thick_vessel_row = rows.get("use_thick_vessel_skeletonisation")
        if thick_vessel_row is not None:
            large_vessel_network_mode = bool(
                values.get("use_thick_vessel_skeletonisation")
            ) and bool(values.get("assign_large_vessel_branch_orders"))
            thick_vessel_row.label = (
                "Also segment large vessels"
                if large_vessel_network_mode
                else fields["use_thick_vessel_skeletonisation"].label
            )

    for name, value in DISPLAY_SETTINGS_OFF_IN_NAPARI.items():
        if name in rows:
            rows[name].value = value

    for widget in rows.values():
        widget.changed.connect(apply_prerequisites)

    #: True while one side of the Diameter measurement choice is writing the
    #: other, so the write does not echo back.
    diameter_source_syncing = {"active": False}

    def on_diameter_source_chosen(*_args) -> None:
        """The choice -> its two settings."""
        if diameter_source_syncing["active"]:
            return
        wanted = settings_for(str(diameter_source.value))
        diameter_source_syncing["active"] = True
        try:
            # Off before on, so the two are never both on, even for a moment.
            for name in sorted(wanted, key=lambda n: wanted[n]):
                if rows[name].value != wanted[name]:
                    rows[name].value = wanted[name]
        finally:
            diameter_source_syncing["active"] = False

    def diameter_settings_now() -> dict[str, Any]:
        return {name: rows[name].value for name in DIAMETER_SOURCE_SETTINGS}

    def show_diameter_source() -> None:
        chosen = source_from(diameter_settings_now())
        if diameter_source.value != chosen:
            diameter_source_syncing["active"] = True
            try:
                diameter_source.value = chosen
            finally:
                diameter_source_syncing["active"] = False

    def settle_both_on() -> None:
        """Both still on once the load is done: keep FWHM, and say so.

        The one combination the choice cannot show, and a run refuses.
        Checked once the event loop is back, not as each row is set: a
        config switching from the endothelium to FWHM is loaded one setting
        at a time, and passes through both on without meaning it.
        """
        if not both_on(diameter_settings_now()):
            return
        diameter_source_syncing["active"] = True
        try:
            rows[DIAMETER_SOURCE_SETTINGS[1]].value = False
        finally:
            diameter_source_syncing["active"] = False
        report.value = BOTH_ON_NOTE
        logger.warning(BOTH_ON_NOTE)
        show_diameter_source()

    def on_diameter_setting_changed(*_args) -> None:
        """Its two settings -> the choice: a loaded config, or a test, set them."""
        if diameter_source_syncing["active"]:
            return
        if both_on(diameter_settings_now()):
            from qtpy.QtCore import QTimer

            QTimer.singleShot(0, settle_both_on)
        show_diameter_source()

    diameter_source.changed.connect(on_diameter_source_chosen)
    for name in DIAMETER_SOURCE_SETTINGS:
        rows[name].changed.connect(on_diameter_setting_changed)

    #: User-facing values of the resume skip toggles before Revert turns them
    #: off. Restored by "Clear layers and state".
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
        _give_bottom_docks_the_corners(viewer)

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
            """A dropped image becomes the input, panel already open or not.

            Not one of our own: a run that shows results adds Image-kind
            layers for its own output (the large-vessel mask volumes, in
            particular -- see `vessel_mask_volume_layers`). Without this
            check, showing those masks silently repointed the *next* run's
            input_path at whichever one was added last, since every added
            Image layer was treated as something the user just dropped in.
            """
            layer_picker.reset_choices()
            layer = getattr(event, "value", None)
            if _is_ours(layer):
                return
            adopt(layer)

        def on_layer_removed(_event) -> None:
            layer_picker.reset_choices()

        # Both orders have to work: open the panel with an image already there,
        # or drop one in while it is open.
        layer_picker.changed.connect(lambda *_: adopt(layer_picker.value))
        use_button.changed.connect(lambda *_: use_layer(layer_picker.value))
        viewer.layers.events.inserted.connect(on_layer_added)
        viewer.layers.events.removed.connect(on_layer_removed)

        images = [
            layer
            for layer in viewer.layers
            if isinstance(layer, napari.layers.Image) and not _is_ours(layer)
        ]
        active = viewer.layers.selection.active
        if isinstance(active, napari.layers.Image) and not _is_ours(active):
            adopt(active)
        elif images:
            adopt(images[-1])

    #: The Input tab's "Check and optimise" box, and the Advanced button its
    #: own settings sit behind: the check and optimise buttons go in the box,
    #: before that button, and their less-used options behind it.
    check_box = input_boxes.get("check_and_optimise")
    check_advanced = advanced_disclosures.get("segment:box:check_and_optimise")

    def into_check_box(widget, position: int) -> None:
        if check_box is not None:
            check_box.insert(position, widget)

    def behind_check_advanced(widget) -> None:
        if check_advanced is not None:
            check_advanced.add(widget)
        elif check_box is not None:
            check_box.append(widget)

    #: "Optimise settings": empirically choose Skeletonise/Graph tab values
    #: from the segmented input image. The button lives on the Input tab
    #: itself, in its "Check and optimise" box; its progress bars are
    #: plain Qt (`OptimiseProgressBars.native`), so they join the shared panel
    #: chrome beside the pipeline's own `bars.native` rather than being forced
    #: into a magicgui Container, which only accepts magicgui widgets.
    optimise_button = PushButton(text="Optimise settings")
    from haemolynx.gui.chrome_tooltips import OPTIMISE_SETTINGS_TOOLTIP

    optimise_button.tooltip = OPTIMISE_SETTINGS_TOOLTIP
    into_check_box(optimise_button, 0)
    optimise_bars = OptimiseProgressBars()
    #: What a run would change, under the button, until Apply or Discard.
    optimise_review = _OptimiseReview(current_values, report, what="optimised settings")
    into_check_box(optimise_review.container, 1)

    #: "Optimisation downsampling": how coarse a copy of the image the search
    #: runs on, independent of the resolution a real pipeline run always uses
    #: -- see haemolynx.optimisation.search.estimate_downsample_factor_for_time_budget
    #: for what "Auto" picks. Kept as a display-label -> factor mapping, since
    #: "None" (auto-detect) is not a value a ComboBox choice can hold directly.
    _DOWNSAMPLE_CHOICES: dict[str, Any] = {
        "Auto": None,
        "Off": 1,
        "2x": 2,
        "4x": 4,
        "8x": 8,
        "16x": 16,
    }
    downsample_dropdown = ComboBox(
        label="Optimisation downsampling",
        choices=list(_DOWNSAMPLE_CHOICES),
        value="Auto",
    )
    downsample_dropdown.tooltip = (
        "How coarse a copy of the image to search on for speed -- applies "
        "only to Optimise settings, never to the pipeline run itself. Auto "
        "never goes so coarse that a typical vessel is under three voxels "
        "across, and goes finer only while its timing of this image says the "
        "search stays under about five minutes"
    )
    behind_check_advanced(downsample_dropdown)

    from haemolynx.gui.chrome_tooltips import OPTIMISE_TOOLTIPS

    #: "Optimisation passes": repeat the whole search from where the last pass
    #: ended (`max_passes`), stopping early once a pass changes nothing.
    passes_spinbox = SpinBox(label="Optimisation passes", value=1, min=1, max=5)
    passes_spinbox.tooltip = OPTIMISE_TOOLTIPS["passes"]
    behind_check_advanced(passes_spinbox)

    #: "Choose optimisation types": restricts Optimise settings to a subset of
    #: its eleven groups. The per-group checkboxes stay hidden until asked for --
    #: most runs want every group, so the list only appears once asked for.
    from haemolynx.optimisation import GROUP_LABELS, GROUP_NAMES

    choose_groups_checkbox = CheckBox(text="Choose optimisation types", value=False)
    choose_groups_checkbox.tooltip = (
        "Restrict Optimise settings to only the ticked group(s) below, "
        "instead of every Skeletonise/Graph setting"
    )
    behind_check_advanced(choose_groups_checkbox)

    group_checkboxes: dict[str, Any] = {
        name: CheckBox(text=GROUP_LABELS[name], value=True) for name in GROUP_NAMES
    }
    group_checkboxes_container = Container(widgets=list(group_checkboxes.values()), labels=False)
    group_checkboxes_container.visible = False
    behind_check_advanced(group_checkboxes_container)

    #: "Check segmented image": scores the raw segmented input mask, 0-10,
    #: before any pipeline stage runs on it -- see
    #: haemolynx.preprocessing.segmentation_quality for the five 0-2
    #: sub-scores. A read-only diagnostic: never writes to `rows`, unlike
    #: "Optimise settings" beside it.
    check_image_button = PushButton(text="Check segmented image")
    from haemolynx.gui.chrome_tooltips import CHECK_SEGMENTED_IMAGE_TOOLTIP

    check_image_button.tooltip = CHECK_SEGMENTED_IMAGE_TOOLTIP
    into_check_box(check_image_button, 0)

    #: A mirror of the FWHM raw-image path (`fwhm_raw_tiff_path`), shown here
    #: too so "Check segmented image" can cross-check the segmentation
    #: against its own raw data without needing use_fwhm_edge_diameters on.
    #: The Diameters-tab row for the same setting stays gated behind that
    #: toggle (its own `requires`) -- a real pipeline run only needs the
    #: file to exist when FWHM measurement itself is on -- but this mirror
    #: is a separate widget, so it stays enabled regardless. Two-way synced
    #: with rows["fwhm_raw_tiff_path"]: one underlying setting, editable
    #: from either place, so a path entered here also feeds FWHM
    #: measurement and vice versa.
    raw_data_row = FileEdit(
        mode="r",
        value=rows["fwhm_raw_tiff_path"].value,
        label="Raw data file (optional)",
    )
    raw_data_row.tooltip = (
        "Optional raw (unsegmented) image of the same dataset. When set, "
        "'Check segmented image' also cross-checks the segmentation against "
        "it for added voxels, removed voxels, and whole missed structures. "
        "Shares its value with fwhm_raw_tiff_path on the Diameters tab "
        "(also used there for FWHM diameter measurement)."
    )
    into_check_box(raw_data_row, 0)

    _raw_data_row_syncing = {"active": False}

    def _sync_raw_data_row_from_primary(*_args) -> None:
        if _raw_data_row_syncing["active"]:
            return
        _raw_data_row_syncing["active"] = True
        try:
            raw_data_row.value = rows["fwhm_raw_tiff_path"].value
        finally:
            _raw_data_row_syncing["active"] = False

    def _sync_primary_from_raw_data_row(*_args) -> None:
        if _raw_data_row_syncing["active"]:
            return
        _raw_data_row_syncing["active"] = True
        try:
            rows["fwhm_raw_tiff_path"].value = raw_data_row.value
        finally:
            _raw_data_row_syncing["active"] = False

    raw_data_row.changed.connect(_sync_primary_from_raw_data_row)
    rows["fwhm_raw_tiff_path"].changed.connect(_sync_raw_data_row_from_primary)

    #: The raw file's channel, mirrored the same way: "Check segmented image"
    #: reads fwhm_raw_channel with the file, and the Diameters-tab row for it
    #: is hidden while FWHM is off.
    raw_channel_row = _channel_combo_box_class()(
        choices=channel_choices([], rows["fwhm_raw_channel"].value),
        value=rows["fwhm_raw_channel"].value,
        label="Raw data channel",
    )
    raw_channel_row.tooltip = (
        "Which channel of a multi-channel raw data file to read, as Fiji names them. "
        "Shares its value with fwhm_raw_channel on the Diameters tab."
    )
    into_check_box(raw_channel_row, 1)
    _follow_image_channels(raw_channel_row, raw_data_row)
    _raw_channel_syncing = {"active": False}

    def _mirror_channel(source, target) -> None:
        if _raw_channel_syncing["active"]:
            return
        _raw_channel_syncing["active"] = True
        try:
            target.value = source.value
        finally:
            _raw_channel_syncing["active"] = False

    raw_channel_row.changed.connect(
        lambda *_a: _mirror_channel(raw_channel_row, rows["fwhm_raw_channel"])
    )
    rows["fwhm_raw_channel"].changed.connect(
        lambda *_a: _mirror_channel(rows["fwhm_raw_channel"], raw_channel_row)
    )

    #: "Optimise FWHM settings": empirically choose the FWHM diameter-
    #: measurement settings from the current in-memory graph and its raw
    #: image -- the Diameters tab's own analogue of "Optimise settings" on
    #: the Input tab. Placed right after the fwhm_raw_tiff_path row and
    #: before the rest of the FWHM settings, per its own visibility rule
    #: (`apply_prerequisites`, above) rather than always shown: there is
    #: nothing to search until FWHM measurement itself is turned on.
    optimise_fwhm_button = PushButton(text="Optimise FWHM settings")
    optimise_fwhm_button.tooltip = (
        "Empirically choose the FWHM profile model and sampling, exclusion, "
        "extent, clipping, same-edge-geometry, baseline and rejection-gate "
        "settings by running the real measurement on a sample of the current "
        "graph's own edges against its raw image -- judged, once the pipeline "
        "has segmented the image, on vessels of known width planted beside "
        "them. Proposes the winners for this tab; the current network's "
        "diameters are not changed"
    )
    #: Its options -- which groups, how many vessels, how many passes -- behind
    #: a toggle, and its review box: one box under the button, shown with it.
    from haemolynx.optimisation import FWHM_GROUP_LABELS

    fwhm_options_toggle = CheckBox(text="FWHM optimiser options", value=False)
    fwhm_options_toggle.tooltip = OPTIMISE_TOOLTIPS["fwhm_options"]
    fwhm_choose_groups = CheckBox(text="Choose FWHM optimisation types", value=False)
    fwhm_choose_groups.tooltip = OPTIMISE_TOOLTIPS["fwhm_choose_groups"]
    fwhm_group_checkboxes: dict[str, Any] = {
        name: CheckBox(text=FWHM_GROUP_LABELS[name], value=True) for name in FWHM_GROUP_NAMES
    }
    fwhm_group_container = Container(widgets=list(fwhm_group_checkboxes.values()), labels=False)
    fwhm_group_container.visible = False
    #: Display label -> sample size; ``None`` is Auto (see
    #: `optimise_fwhm_settings`'s own `sample_edge_count`).
    _FWHM_SAMPLE_CHOICES: dict[str, Any] = {
        "Auto": None,
        "25 vessels": 25,
        "50 vessels": 50,
        "100 vessels": 100,
        "200 vessels": 200,
        "Every vessel": 2**31 - 1,
    }
    fwhm_sample_dropdown = ComboBox(
        label="FWHM optimisation sample", choices=list(_FWHM_SAMPLE_CHOICES), value="Auto"
    )
    fwhm_sample_dropdown.tooltip = OPTIMISE_TOOLTIPS["fwhm_sample"]
    fwhm_passes_spinbox = SpinBox(label="Optimisation passes", value=1, min=1, max=5)
    fwhm_passes_spinbox.tooltip = OPTIMISE_TOOLTIPS["passes"]
    fwhm_options_container = Container(
        widgets=[fwhm_choose_groups, fwhm_group_container, fwhm_sample_dropdown, fwhm_passes_spinbox]
    )
    fwhm_options_container.visible = False
    fwhm_review = _OptimiseReview(current_values, report, what="optimised FWHM settings")
    fwhm_optimise_box = Container(
        widgets=[fwhm_options_toggle, fwhm_options_container, fwhm_review.container], labels=False
    )
    fwhm_options_toggle.changed.connect(
        lambda *_args: setattr(fwhm_options_container, "visible", bool(fwhm_options_toggle.value))
    )
    fwhm_choose_groups.changed.connect(
        lambda *_args: setattr(fwhm_group_container, "visible", bool(fwhm_choose_groups.value))
    )
    if diameters_settings is not None:
        raw_tiff_row_index = list(diameters_settings).index(rows["fwhm_raw_tiff_path"])
        diameters_settings.insert(raw_tiff_row_index + 1, optimise_fwhm_button)
        diameters_settings.insert(raw_tiff_row_index + 2, fwhm_optimise_box)
    # OptimiseProgressBars.native is a plain Qt widget, not a magicgui one --
    # like the Input tab's own `optimise_bars`, it joins the shared bottom
    # chrome (below the tabs, added further down) rather than the magicgui
    # Container above, which only accepts magicgui widgets.
    optimise_fwhm_bars = OptimiseProgressBars()

    def on_optimise_fwhm_settings() -> None:
        if run_state.running:
            report.value = ALREADY_RUNNING
            return
        if view.results is None or getattr(view.results, "_graph", None) is None:
            report.value = (
                "Nothing to optimise yet: run the pipeline through at least Graph first."
            )
            return
        values = current_values()
        raw_path = values.get("fwhm_raw_tiff_path")
        if not raw_path or not Path(raw_path).is_file():
            report.value = (
                "Choose a raw FWHM image first, then press Optimise FWHM settings."
            )
            return
        results = view.results
        graph = copy_graph(results._graph)
        voxel_size_zyx = tuple(
            float(v) for v in getattr(results, "_voxel_size_zyx", (1.0, 1.0, 1.0))
        )
        skeletonised = checkpoints.get("skeletonise")
        groups = None
        if fwhm_choose_groups.value:
            groups = tuple(name for name, box in fwhm_group_checkboxes.items() if box.value)
        fwhm_review.clear()
        _run_fwhm_optimisation_in_background(
            graph,
            voxel_size_zyx,
            _settings(),
            schema,
            rows,
            report,
            optimise_fwhm_button,
            optimise_fwhm_bars,
            apply_prerequisites=apply_prerequisites,
            run_state=run_state,
            groups=groups,
            vessel_mask=getattr(getattr(skeletonised, "output", None), "image", None),
            sample_edge_count=_FWHM_SAMPLE_CHOICES.get(fwhm_sample_dropdown.value),
            max_passes=int(fwhm_passes_spinbox.value),
            propose=fwhm_review.propose,
        )

    optimise_fwhm_button.changed.connect(lambda *_args: on_optimise_fwhm_settings())
    apply_prerequisites()

    def _toggle_group_checkboxes(*_args) -> None:
        group_checkboxes_container.visible = bool(choose_groups_checkbox.value)

    choose_groups_checkbox.changed.connect(_toggle_group_checkboxes)

    load_button = PushButton(text="Load config...")
    view_button = PushButton(text="View")
    edit_button = PushButton(text="Edit")
    save_button = PushButton(text="Save config...")
    check_button = PushButton(text="Run checks")
    run_button = PushButton(text="Run pipeline")
    clear_button = PushButton(text="Clear layers and state")
    save_run_button = PushButton(text="Save run...")
    load_run_button = PushButton(text="Load run...")

    from haemolynx.gui.chrome_tooltips import (
        CLEAR_LAYERS_TOOLTIP,
        EDIT_GRAPH_TOOLTIP,
        LOAD_CONFIG_TOOLTIP,
        LOAD_RUN_TOOLTIP,
        REOPEN_VIEW_TOOLTIP,
        RUN_CHECKS_TOOLTIP,
        RUN_PIPELINE_TOOLTIP,
        SAVE_CONFIG_TOOLTIP,
        SAVE_RUN_TOOLTIP,
        SCALE_BAR_TOOLTIP,
        SHOW_RESULTS_TOOLTIP,
        SHOW_STEPS_TOOLTIP,
        SNAPSHOT_TOOLTIP,
        SWEEP_TOOLTIP,
        LAYER_SET_TOOLTIP,
        VESSEL_DRAW_TOOLTIP,
        Z_DEPTH_TOOLTIP,
    )

    load_button.tooltip = LOAD_CONFIG_TOOLTIP
    view_button.tooltip = REOPEN_VIEW_TOOLTIP
    view_button.enabled = viewer is not None
    edit_button.tooltip = EDIT_GRAPH_TOOLTIP
    save_button.tooltip = SAVE_CONFIG_TOOLTIP
    check_button.tooltip = RUN_CHECKS_TOOLTIP
    run_button.tooltip = RUN_PIPELINE_TOOLTIP
    clear_button.tooltip = CLEAR_LAYERS_TOOLTIP
    save_run_button.tooltip = SAVE_RUN_TOOLTIP
    load_run_button.tooltip = LOAD_RUN_TOOLTIP
    save_run_button.native.setObjectName("haemolynx_save_run")
    load_run_button.native.setObjectName("haemolynx_load_run")

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
    from qtpy.QtWidgets import (
        QButtonGroup, QCheckBox, QFormLayout, QGroupBox, QHBoxLayout,
        QLabel, QMenu, QPushButton, QRadioButton, QToolButton, QWidget,
    )
    from superqt import QDoubleSlider

    z_depth_label = QLabel("Z-depth filter (µm)")
    z_depth_slider = _double_range_pair("haemolynx_z_depth_slider")
    z_depth_slider.setEnabled(False)
    z_depth_label.setToolTip(Z_DEPTH_TOOLTIP)
    z_depth_slider.setToolTip(Z_DEPTH_TOOLTIP)

    vessel_draw_label = QLabel("Vessels")
    vessel_draw_label.setObjectName("haemolynx_vessel_draw_label")
    # Radios, not a combo: the view dock floats over the vispy canvas, and a
    # QComboBox popup there often never receives the click that should swap
    # Tubes/Lines.
    vessel_draw = QWidget()
    vessel_draw.setObjectName("haemolynx_vessel_draw")
    tubes_radio = QRadioButton("Tubes")
    lines_radio = QRadioButton("Lines")
    tubes_radio.setObjectName("haemolynx_vessel_draw_tubes")
    lines_radio.setObjectName("haemolynx_vessel_draw_lines")
    draw_group = QButtonGroup(vessel_draw)
    draw_group.setExclusive(True)
    draw_group.addButton(tubes_radio)
    draw_group.addButton(lines_radio)
    draw_row_layout = QHBoxLayout(vessel_draw)
    draw_row_layout.setContentsMargins(0, 0, 0, 0)
    draw_row_layout.addWidget(tubes_radio)
    draw_row_layout.addWidget(lines_radio)
    vessel_draw.tubes = tubes_radio
    vessel_draw.lines = lines_radio

    def _vessel_draw_text() -> str:
        return "Lines" if lines_radio.isChecked() else "Tubes"

    def _set_vessel_draw_text(text: str) -> None:
        if text == "Lines":
            lines_radio.setChecked(True)
        else:
            tubes_radio.setChecked(True)

    vessel_draw.currentText = _vessel_draw_text
    vessel_draw.setCurrentText = _set_vessel_draw_text
    vessel_draw.setToolTip(VESSEL_DRAW_TOOLTIP)
    tubes_radio.setToolTip(VESSEL_DRAW_TOOLTIP)
    lines_radio.setToolTip(VESSEL_DRAW_TOOLTIP)
    vessel_draw_label.setToolTip(VESSEL_DRAW_TOOLTIP)
    if _vessel_draw_mode == VESSEL_DRAW_LINES:
        lines_radio.setChecked(True)
    else:
        tubes_radio.setChecked(True)

    # Which network is drawn: the baseline, or one perturbation. A menu on a
    # button rather than a QComboBox, whose popup over the floating dock
    # missed clicks (see the Tubes/Lines radios); it only appears once a
    # perturbation has layers to swap to.
    layer_set_label = QLabel("Showing")
    layer_set_label.setObjectName("haemolynx_layer_set_label")
    layer_set_button = QToolButton()
    layer_set_button.setObjectName("haemolynx_layer_set")
    layer_set_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
    layer_set_menu = QMenu(layer_set_button)
    layer_set_menu.setObjectName("haemolynx_layer_set_menu")
    layer_set_button.setMenu(layer_set_menu)
    layer_set_label.setToolTip(LAYER_SET_TOOLTIP)
    layer_set_button.setToolTip(LAYER_SET_TOOLTIP)

    def choose_layer_set(key: str | None) -> None:
        if viewer is None:
            return
        show_layer_set(viewer, key)
        refresh_layer_sets()

    def refresh_layer_sets(*_args) -> None:
        """List the networks in the viewer; hide the row with only the baseline.

        The row stays while a perturbation is shown even after its layers
        went, so there is always a way back to the baseline.
        """
        from haemolynx.gui.layer_sets import layer_set_choices

        try:
            perturbations = _perturbation_sets_in(viewer)
            shown = _layer_set_shown(viewer) if viewer is not None else None
            choices = layer_set_choices(perturbations)
            layer_set_menu.clear()
            for label, key in choices:
                action = layer_set_menu.addAction(label)
                action.setCheckable(True)
                action.setChecked(key == shown)
                action.setData(key)
                action.triggered.connect(
                    lambda _checked=False, key=key: choose_layer_set(key)
                )
            label = next((label for label, key in choices if key == shown), str(shown))
            layer_set_button.setText(label)
            layer_set_row.setVisible(bool(perturbations) or shown is not None)
            refresh_sweep_controls()
            colour_by.refresh()
        except RuntimeError:
            # The dock's Qt widgets outlive the panel on teardown.
            logger.debug("layer-set menu is gone", exc_info=True)

    scale_bar_box = QCheckBox("Scale bar")
    scale_bar_box.setObjectName("haemolynx_scale_bar")
    scale_bar_box.setChecked(False)
    scale_bar_box.setToolTip(SCALE_BAR_TOOLTIP)

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
    _view_z_apply = {"busy": False, "key": None}

    def reparent_arrow_length_slider() -> None:
        if viewer is None:
            return
        _reparent_arrow_length_slider(
            viewer,
            slider=arrow_length_slider,
            host=arrow_length_host,
            parking=arrow_length_parking,
            controls_holder=arrow_length_mount,
        )

    if viewer is not None:
        viewer._haemolynx_reparent_arrow_length_slider = reparent_arrow_length_slider
        viewer.layers.selection.events.active.connect(
            lambda *_args: reparent_arrow_length_slider()
        )
        viewer.layers.selection.events.active.connect(
            lambda *_args: _focus_image_layer_rendering(viewer)
        )

    def apply_view_z(*, force: bool = False) -> None:
        """Re-apply the left-panel Z-depth clip to every displayed layer.

        Full-range is the identity. Does not write into settings or stage
        inputs; ``data_for_pipeline`` still reads the unclipped
        ``z_window_full`` cache.

        Slider drags emit ``valueChanged`` on every tick; the expensive
        volume/graph rebuild runs on release (or immediately for
        programmatic ``setValue``). A matching window is a no-op unless
        *force* (new layers).
        """
        if viewer is None or _view_z_apply["busy"]:
            return
        extent, _step = _z_extent_and_step(view.results, viewer)
        if extent is None or extent <= 0.0:
            return
        try:
            if z_depth_slider.isEnabled() or getattr(
                z_depth_slider, "_haemolynx_extent_ready", False
            ):
                vol_lo, vol_hi = z_depth_slider.value()
            else:
                vol_lo, vol_hi = 0.0, extent
        except RuntimeError:
            # See _sync_z_depth_slider: the slider's Qt widget has, on real
            # production-scale data, been found already deleted. Fall back
            # to the full range rather than let this raise inside a Qt slot.
            logger.debug("Z-depth slider is gone; using the full range", exc_info=True)
            vol_lo, vol_hi = 0.0, extent
        key = (float(vol_lo), float(vol_hi))
        if not force and key == _view_z_apply["key"]:
            return
        _view_z_apply["busy"] = True
        try:
            viewer._haemolynx_z_depth_window = (float(vol_lo), float(vol_hi), float(extent))
            _apply_volume_z_display(viewer, vol_lo, vol_hi, z_extent=extent)
            _resync_thick_thin_skeleton_after_z_change(viewer)
            _resync_segmentation_cleanup_after_z_change(viewer)
            _apply_z_filter(viewer, vol_lo, vol_hi, z_extent=extent)
            for layer in viewer.layers:
                if _is_branch_hover_layer(layer):
                    _apply_branch_hover_selection(
                        layer, _branch_hover_selected_for(layer)
                    )
            if scale_bar_box.isChecked():
                _apply_scale_bar(viewer, True)
            _view_z_apply["key"] = key
        finally:
            _view_z_apply["busy"] = False

    def _sync_view_z_sliders() -> None:
        _sync_z_depth_slider(z_depth_slider, view.results, viewer)

    def _after_layers_applied() -> None:
        """Refresh Z sliders and re-apply the current view-only filters."""
        if viewer is None:
            return
        _sync_view_z_sliders()
        reparent_arrow_length_slider()
        _focus_image_layer_rendering(viewer)
        apply_view_z(force=True)
        _reapply_layer_set(viewer)
        refresh_layer_sets()

    if viewer is not None:
        viewer._haemolynx_after_layers_applied = _after_layers_applied
        # Only the menu: a layer the Z filter replaces is removed and re-added,
        # and switching networks in between would lose the choice.
        viewer.layers.events.inserted.connect(refresh_layer_sets)
        viewer.layers.events.removed.connect(refresh_layer_sets)

    def _z_slider_should_apply(slider) -> bool:
        return not bool(slider.isSliderDown())

    def on_z_depth_changed(*_args) -> None:
        if _z_slider_should_apply(z_depth_slider):
            apply_view_z()

    def on_z_slider_released() -> None:
        apply_view_z()

    def on_scale_bar_toggled(checked: bool) -> None:
        if viewer is None:
            return
        _apply_scale_bar(viewer, bool(checked))

    def on_vessel_draw_changed(checked: bool = True) -> None:
        if not checked:
            return
        global _vessel_draw_mode
        mode = (
            VESSEL_DRAW_LINES if lines_radio.isChecked() else VESSEL_DRAW_TUBES
        )
        _vessel_draw_mode = mode
        if viewer is not None:
            _sync_vessel_tubes(viewer)

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
    z_depth_slider.sliderReleased.connect(on_z_slider_released)
    scale_bar_box.toggled.connect(on_scale_bar_toggled)
    tubes_radio.toggled.connect(on_vessel_draw_changed)
    lines_radio.toggled.connect(on_vessel_draw_changed)
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
        loaded_config_dir[0] = Path(path).parent
        last_config_path[0] = str(path)
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
            _dialog_parent(viewer),
            "Open a HaemoLynx config",
            last_config_path[0],
            "YAML (*.yaml *.yml)",
        )
        if not path:
            return
        last_config_path[0] = path
        load_config_file(path)

    def _values_for_save(dest: Path) -> dict[str, Any]:
        """Form values with relative loaded paths rebased if *dest* left cwd."""
        values = current_values()
        try:
            dest_dir = dest.parent.resolve()
            keep_relative = dest_dir == Path.cwd().resolve()
            if loaded_config_dir[0] is not None:
                keep_relative = dest_dir == loaded_config_dir[0].resolve()
        except OSError:
            keep_relative = False
        if keep_relative:
            return values
        for name, original in loaded_paths.items():
            if original is None or name not in values:
                continue
            try:
                if Path(original).is_absolute():
                    continue
                values[name] = Path(original).expanduser().resolve()
            except (TypeError, ValueError, OSError):
                continue
        return values

    def save_config_file(path: Path | str) -> bool:
        """Write the panel's current settings to *path*.

        Returns True when the file was written. Failures (schema validation,
        I/O, still-broken YAML edge cases) are reported in the panel rather
        than raised through the Qt/psygnal signal — same contract as load.
        """
        dest = ensure_yaml_suffix(path)
        try:
            dump_config(dest, schema, values=_values_for_save(dest))
        except Exception as error:
            report.value = f"Could not save {dest}:\n{error}"
            return False
        last_config_path[0] = str(dest)
        loaded_config_dir[0] = dest.parent
        report.value = f"Wrote {dest}"
        return True

    def on_save() -> None:
        from qtpy.QtWidgets import QFileDialog

        path, _filter = QFileDialog.getSaveFileName(
            _dialog_parent(viewer),
            "Save these settings",
            last_config_path[0] or "config.yaml",
            "YAML (*.yaml *.yml)",
        )
        if not path:
            return
        save_config_file(path)

    def on_optimise_settings() -> None:
        if run_state.running:
            report.value = ALREADY_RUNNING
            return
        values = current_values()
        input_path = values.get("input_path")
        if not input_path or not Path(input_path).is_file():
            report.value = "Choose a segmented input image first, then press Optimise settings."
            return
        downsample_factor = _DOWNSAMPLE_CHOICES.get(downsample_dropdown.value)
        groups = None
        if choose_groups_checkbox.value:
            groups = tuple(name for name, box in group_checkboxes.items() if box.value)
        optimise_review.clear()
        _run_optimisation_in_background(
            _settings(),
            schema,
            rows,
            report,
            optimise_button,
            optimise_bars,
            apply_prerequisites=apply_prerequisites,
            run_state=run_state,
            downsample_factor=downsample_factor,
            groups=groups,
            max_passes=int(passes_spinbox.value),
            propose=optimise_review.propose,
        )

    def on_check_segmented_image() -> None:
        if run_state.running:
            report.value = ALREADY_RUNNING
            return
        values = current_values()
        input_path = values.get("input_path")
        if not input_path or not Path(input_path).is_file():
            report.value = "Choose a segmented input image first, then press Check segmented image."
            return
        _run_segmentation_quality_check_in_background(
            _settings(),
            report,
            check_image_button,
            run_state=run_state,
        )

    def on_check() -> None:
        result = preflight(_settings(), schema)
        lines = [f"FAILED: {message}" for message in result.errors]
        lines += [f"warning: {message}" for message in result.warnings]
        report.value = "\n".join(lines) if lines else "All checks passed."

    def refresh_revert_buttons() -> None:
        """Enable each tab's run-from button only when its previous stage is saved."""
        for title, button in revert_buttons.items():
            button.enabled = (
                not run_state.running and can_revert_from(title, checkpoints)
            )

    def _restore_skip_toggles() -> bool:
        """Put skeletonize / graph-building / FWHM back to the user's values."""
        restored = False
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
                        restored = True
        finally:
            for name in disconnected:
                rows[name].changed.connect(snapshot_skip_toggles)
        if restored:
            apply_prerequisites()
        return restored

    def on_run(
        *_args,
        start_from=None,
        resume=None,
        replace_checkpoints: bool = True,
        restored_stages=(),
        stop_after="auto",
    ) -> None:
        """Start a run. *stop_after* is where it stops: ``"auto"`` pauses it
        before Post processing when Mid-run postprocessing is on (see
        :func:`~haemolynx.gui.run_state.mid_run_stop_after`), else a stage
        call, or None to run to the end."""
        if run_state.running:
            # The button is disabled while a run is going, so this is only
            # reached from a script or a keyboard -- but it is also the one
            # place that says how to get out of a run, so it says it.
            report.value = ALREADY_RUNNING
            return
        if start_from is None:
            # Run pipeline always uses the user's skip toggles, not leftover
            # flags from "Run from this stage".
            _restore_skip_toggles()
        try:
            settings = _settings()
        except Exception as error:
            report.value = f"Could not read settings:\n{error}"
            return
        if not preflight(settings, schema).ok:
            report.value = "Checks failed; nothing was run. Press 'Run checks' for detail."
            return
        if start_from is None:
            # A full run starts from a clean slate: old layers, checkpoints
            # and cached resume pickles from a previous run must not leak
            # into this one. "Run from this stage" resumes deliberately keep
            # them, so this only fires for a genuine fresh run.
            on_clear(ask=False)
        results = None
        if show_results.value and viewer is not None:
            if replace_checkpoints or view.results is None:
                results = ResultLayers(
                    show_steps=bool(show_steps.value),
                    settings=settings,
                )
                view.results = results
                if boundaries is not None:
                    boundaries.state.results = results
            else:
                results = view.results
            # A full run that will show layers replaces the previous run's
            # checkpoints. "Run from this stage" has already dropped later
            # ones and must keep the previous tab's snapshot.
            if replace_checkpoints:
                checkpoints.clear()
        # How much of the run to show follows the setting that already means
        # "tell me everything"; the window itself is a panel control, like
        # `show_results`, because a config file is read by CLI runs too.
        log_view.set_level(
            VERBOSE_LEVEL if settings.get("verbose_logging") else DEFAULT_LEVEL
        )
        show_log()
        refresh_revert_buttons()
        if stop_after == "auto":
            # A pause needs the run on screen: the tab edits the viewer's graph.
            stop_after = mid_run_stop_after(settings, start_from) if results is not None else None
        run_state.paused_after = None
        worker = _run_in_background(
            settings, schema, report, run_button, bars,
            viewer=viewer if show_results.value else None,
            results=results,
            state=run_state,
            log=log_view,
            checkpoints=checkpoints if results is not None else None,
            after_layers=_after_layers_applied if show_results.value else None,
            start_from=start_from,
            resume=resume,
            restored_stages=restored_stages,
            stop_after=stop_after,
            on_paused=on_paused,
        )
        # After start(): running is True, so run-from buttons grey out for
        # the length of the run. They come back when the worker stops.
        refresh_revert_buttons()
        post_processing.refresh()
        if worker is not None:
            for signal in (worker.returned, worker.errored, worker.finished):
                signal.connect(lambda *_: refresh_revert_buttons())
                signal.connect(lambda *_: post_processing.refresh())

    def on_paused() -> None:
        """A run stopped before Post processing (or after a Regenerate graph
        there): bring up that tab with the network scanned, for Continue."""
        titles = [tab_widget.tabText(i) for i in range(tab_widget.count())]
        if POST_PROCESSING_TAB in titles:
            tab_widget.setCurrentIndex(titles.index(POST_PROCESSING_TAB))
        # Scanning reports its junction count; the pause message says more.
        message = report.value
        try:
            post_processing.scan()
        except Exception:  # noqa: BLE001 - the run is paused either way
            logger.exception("could not scan the paused network")
        report.value = message
        post_processing.refresh()

    def on_clear(*, ask: bool = False) -> None:
        """Take our layers out of the viewer, stop the run, and forget state.

        Cancel first so in-flight layer groups cannot put work back. Then
        free the Run button immediately: the dying worker keeps its own
        cancel flag and cannot grey the panel out again.
        """
        stopping = run_state.cancel()
        if stopping:
            checkpoints.freeze()
            run_state.supersede()
            run_button.enabled = True
            # Whichever of the three was the one running: cancel() does not
            # say which, and re-enabling/resetting the others is a no-op.
            optimise_button.enabled = True
            optimise_bars.reset()
            optimise_fwhm_button.enabled = True
            optimise_fwhm_bars.reset()
            log_view.cancelled()
        optimise_review.clear()
        fwhm_review.clear()
        removed = 0
        if viewer is not None:
            removed = _clear_our_layers(viewer)
            viewer._haemolynx_layer_set = None
            refresh_layer_sets()
            reparent_arrow_length_slider()
            z_depth_slider._haemolynx_extent_ready = False
            z_depth_slider.setEnabled(False)
            arrow_length_slider.setEnabled(False)
            arrow_length_slider.setVisible(False)
            apply_view_z(force=True)
        view.results = None
        if boundaries is not None:
            boundaries.state.results = None
        discarded_artefacts = False
        extra = checkpoints.session_artefact_paths
        try:
            values = current_values()
        except Exception:  # noqa: BLE001 - still reset the panel
            values = None
            logger.debug("could not read settings while clearing", exc_info=True)
        pending = list(extra)
        if values:
            vtk_prefix = values.get("vtk_output_prefix")
            output_dir = output_dir_from_prefix(vtk_prefix)
            if output_dir is not None:
                from haemolynx.gui.stage_checkpoints import stems_for_cached_artefacts

                for stem in stems_for_cached_artefacts(values):
                    pending.extend(output_dir.glob(f"{stem}_checkpoint_*.pkl"))
        existing = tuple(path for path in dict.fromkeys(pending) if Path(path).is_file())
        do_discard = True
        if ask and existing:
            from qtpy.QtWidgets import QMessageBox

            listed = "\n".join(str(path) for path in existing[:8])
            more = "" if len(existing) <= 8 else f"\n... and {len(existing) - 8} more"
            answer = QMessageBox.question(
                _dialog_parent(viewer),
                "Discard cached pickles?",
                "Clear will delete these resume and checkpoint files:\n"
                + listed
                + more,
            )
            do_discard = answer == QMessageBox.Yes
        if do_discard:
            try:
                removed_paths = discard_cached_artefacts_for_settings(
                    values, extra_paths=extra
                )
            except Exception:  # noqa: BLE001 - clearing the panel still matters
                logger.exception("could not discard cached artefacts")
                removed_paths = ()
            discarded_artefacts = bool(removed_paths)
        restored_skips = _restore_skip_toggles()
        checkpoints.clear()
        run_state.paused_after = None
        refresh_revert_buttons()
        post_processing.forget()
        report.value = clear_message(
            removed,
            stopping,
            discarded_artefacts=discarded_artefacts,
            restored_skips=restored_skips,
        )

    def _apply_resume_skips(skip_settings) -> None:
        """Turn *skip_settings* off for a resumed run, without them becoming
        the user's own toggles (which Run pipeline puts back)."""
        nonlocal revert_setting_skips
        saved_skip_snapshot = dict(skip_toggle_snapshot)
        disconnected: list[str] = []
        for name in SKIP_FOR_RESUME:
            if name in rows:
                try:
                    rows[name].changed.disconnect(snapshot_skip_toggles)
                    disconnected.append(name)
                except (TypeError, RuntimeError):
                    pass
        revert_setting_skips = True
        try:
            for name in skip_settings:
                if name in rows:
                    rows[name].value = False
        finally:
            revert_setting_skips = False
            skip_toggle_snapshot.update(saved_skip_snapshot)
            for name in disconnected:
                rows[name].changed.connect(snapshot_skip_toggles)
        apply_prerequisites()

    def _resumed_run_passes_checks(settings, plan) -> bool:
        """Preflight the run *plan* would start, before anything is dropped.

        `on_run` checks too, but only after this tab's and later tabs'
        checkpoints and layers are gone and the skip toggles are off: a
        failing check there threw the finished work away for a run that then
        never started.
        """
        planned = dict(settings)
        planned.update({name: False for name in plan.skip_settings})
        if preflight(planned, schema).ok:
            return True
        report.value = "Checks failed; nothing was run. Press 'Run checks' for detail."
        refresh_revert_buttons()
        return False

    def prepare_run_from(tab_title: str, *, check: bool = False):
        """Drop this tab and later work; return the plan, or None.

        With *check*, nothing is dropped unless the run the plan starts
        passes preflight (see :func:`_resumed_run_passes_checks`).
        """
        if run_state.running:
            report.value = ALREADY_RUNNING
            return None
        if viewer is None:
            report.value = "No viewer to restore layers into."
            return None
        try:
            settings = _settings()
        except Exception as error:
            report.value = f"Could not read settings:\n{error}"
            return None
        # Another run-from may have left its skip toggles off; each one starts
        # from the user's own, or a run from Graph after one from Haemodynamics
        # would load the old graph instead of building a new one.
        if _restore_skip_toggles():
            try:
                settings = _settings()
            except Exception as error:
                report.value = f"Could not read settings:\n{error}"
                return None
        plan = checkpoints.plan_run_from(tab_title, settings=settings, drop=False)
        if plan is None:
            report.value = (
                "Nothing ready to run from: run the pipeline with 'Show each "
                "stage in the viewer' through the previous tab first."
            )
            refresh_revert_buttons()
            return None
        if check and not _resumed_run_passes_checks(settings, plan):
            return None
        checkpoints.drop_from(plan.start_from)
        results = view.results
        if results is None:
            results = ResultLayers()
            view.results = results
            if boundaries is not None:
                boundaries.state.results = results
        checkpoints.apply_to_results(results, plan.checkpoint)
        # Only what the replay will not put back: the replay updates the rest
        # in place. Removing every layer -- the image, masks and FWHM volumes,
        # the tube mesh -- only to re-add each one tore down and rebuilt all
        # of their GL resources at once, the teardown that crashes the process
        # (see _clear_our_layers), and did it most on the latest tabs, which
        # replay the most stages.
        _remove_our_layers(viewer, keep=_layer_names_redrawn_by(plan.groups))
        _apply_layer_groups(viewer, plan.groups)
        _after_layers_applied()
        _apply_resume_skips(plan.skip_settings)
        if plan.tab_title:
            titles = [tab_widget.tabText(i) for i in range(tab_widget.count())]
            if plan.tab_title in titles:
                tab_widget.setCurrentIndex(titles.index(plan.tab_title))
        refresh_revert_buttons()
        report.value = restore_message(plan)
        return plan

    def on_revert(tab_title: str) -> None:
        """Prepare the previous tab's work, then rerun from this stage."""
        from haemolynx.gui.post_processing import edits_lost_by_running_from
        from haemolynx.gui.stage_checkpoints import tab_start_stage

        post_processed = checkpoints.get(POST_PROCESS)
        if edits_lost_by_running_from(
            tab_start_stage(tab_title),
            post_processing.state.graph,
            post_processed.graph if post_processed is not None else None,
        ) and not _confirm_discard_hand_edits(_dialog_parent(viewer), tab_title):
            report.value = f"Kept the hand edits on {POST_PROCESSING_TAB}: nothing was run."
            return
        plan = prepare_run_from(tab_title, check=True)
        if plan is None:
            return
        on_run(
            start_from=plan.start_from or None,
            resume=plan.resume,
            replace_checkpoints=False,
            restored_stages=stages_before(plan.start_from),
        )

    # --- "Edit": add or delete a vessel by hand, then Regenerate to catch
    # haemodynamics/perturbations/measurements/export up to the edit. The
    # editing rules live in gui.graph_editor.GraphEditorState (pure, tested
    # without a viewer); everything here is just wiring a click on the
    # vessels/nodes layers to it and pushing the layers it changes back.
    graph_editor: dict[str, Any] = {"state": None, "window": None}

    def _graph_editor_layer(name: str):
        return viewer.layers[name] if viewer is not None and name in viewer.layers else None

    def _attach_graph_editor_click_callbacks() -> None:
        """(Re-)attach the click handler to the current VESSELS/NODES layer
        objects. `_add_or_update` replaces a Vectors/Points layer outright
        rather than shrinking it in place whenever an edit (e.g. Delete
        edge) reduces its count -- idempotent, so safe to call after every
        refresh, which is what makes that swap harmless instead of silently
        detaching the handler from the layer now on screen."""
        for name in (VESSELS, NODES):
            layer = _graph_editor_layer(name)
            if layer is not None and _graph_editor_click not in layer.mouse_drag_callbacks:
                layer.mouse_drag_callbacks.append(_graph_editor_click)

    def _refresh_graph_editor_layers() -> None:
        state = graph_editor["state"]
        if state is None or view.results is None:
            return
        draft_points = state.draft.points_um if state.draft is not None else None
        _apply_layers(
            viewer, view.results.layers_for_graph(state.graph, draft_points_um=draft_points)
        )
        # `_apply_layers` only ever adds or updates layers it was given --
        # once a draft is finished/committed there is nothing to draw, but
        # the last drawn segment would otherwise stay on screen forever.
        if draft_points is None and EDIT_DRAFT in viewer.layers:
            viewer.layers.remove(viewer.layers[EDIT_DRAFT])
        _attach_graph_editor_click_callbacks()

    def _graph_editor_click_impl(_layer, event) -> None:
        window = graph_editor["window"]
        state = graph_editor["state"]
        if window is None or state is None or not window.is_open() or state.mode == "idle":
            return
        from haemolynx.gui.graph_click import hit_test_nodes, hit_test_vessels

        position = event.position
        dims = list(getattr(event, "dims_displayed", ()) or ())
        view_direction = getattr(event, "view_direction", None)

        edge_hit = None
        vessels = _graph_editor_layer(VESSELS)
        if vessels is not None:
            edge_hit = hit_test_vessels(
                vessels.data, _layer_features(vessels), position,
                view_direction=view_direction, dims=dims,
            )
        node_hit = None
        if edge_hit is None:
            nodes_layer = _graph_editor_layer(NODES)
            if nodes_layer is not None:
                try:
                    index = nodes_layer.get_value(
                        position, view_direction=view_direction,
                        dims_displayed=dims, world=True,
                    )
                except TypeError:
                    index = nodes_layer.get_value(position, world=True)
                node_hit = hit_test_nodes(index, _layer_features(nodes_layer))
        hit = edge_hit if edge_hit is not None else node_hit

        if state.mode == "delete":
            if edge_hit is None:
                return
            state.click_delete(edge_hit)
            _refresh_graph_editor_layers()
            window.report("delete-armed")
            return

        raw_point = tuple(float(c) for c in position[-3:])
        result = state.click_add(raw_point, hit)
        if result != "rejected":
            _refresh_graph_editor_layers()
        window.report(result)

    def _graph_editor_click(_layer, event) -> None:
        # A napari mouse_drag_callback runs on the GUI thread with no
        # surrounding try/except of its own -- an uncaught exception here
        # previously surfaced as the editor just "crashing" mid-click rather
        # than a message the user could act on.
        try:
            _graph_editor_click_impl(_layer, event)
        except Exception as exc:  # noqa: BLE001 - report it, don't crash the viewer
            logger.exception("graph editor click failed")
            window = graph_editor["window"]
            message = f"error: {exc}"
            if window is not None:
                window.report(message)
            report.value = f"Edit: {message}"

    def _arm_graph_editor(mode: str) -> None:
        state = graph_editor["state"]
        if state is None:
            return
        if mode == "add":
            state.start_add()
        else:
            state.start_delete()

    def _finish_graph_editor_branch() -> None:
        state = graph_editor["state"]
        if state is None:
            return
        state.finish_add()
        _refresh_graph_editor_layers()

    def on_regenerate_from_edit() -> None:
        state = graph_editor["state"]
        if state is None:
            return
        regenerate_from_graph(state.graph)

    def regenerate_from_graph(
        graph, *, from_post_processing: bool = False, stop_after=None
    ) -> bool:
        """Run the later stages on the hand-edited *graph*. Returns whether a
        run started (a failed check says why in the report box).

        From the Post processing tab (*from_post_processing*) the run starts
        at its ``post_process`` stage, which brings the edits in line --
        boundary lists cut to the nodes left (the tab's Prune drops inlets on
        purpose), bridges into thick vessels, lengths, branch orders, and the
        edited vessels measured by the run's own diameter methods -- then
        solves the haemodynamics again on the edited network, and goes on to
        *stop_after*, or to the end. Haemodynamics and everything before it
        keep what they recorded: the tab comes after them.

        From the Edit window the run starts at Diameters, as it always has.
        """
        if run_state.running:
            report.value = ALREADY_RUNNING
            return False
        # Like a run from a tab: the user's own toggles first, then the skeleton
        # and the edited graph written where the run loads them, so it neither
        # re-skeletonises the image nor rebuilds a graph it would discard.
        _restore_skip_toggles()
        try:
            settings = _settings()
        except Exception as error:
            report.value = f"Could not read settings:\n{error}"
            return False
        start_from = POST_PROCESS if from_post_processing else "assign_diameters"
        plan = checkpoints.plan_regenerate(
            graph, settings=settings, start_from=start_from, drop=False
        )
        if plan is None:
            report.value = (
                "Nothing to regenerate from: run the pipeline through at "
                "least Boundaries first."
            )
            return False
        if not _resumed_run_passes_checks(settings, plan):
            return False
        checkpoints.drop_from(plan.start_from)
        _apply_resume_skips(plan.skip_settings)
        on_run(
            start_from=plan.start_from,
            resume=plan.resume,
            replace_checkpoints=False,
            # From the Edit window, Boundaries is recorded again, from the
            # edited graph, so a later "Run from this stage" on Diameters keeps
            # the edit. From Post processing the edit comes after
            # Haemodynamics: every stage before it keeps its own record.
            restored_stages=stages_before(
                start_from if from_post_processing else "assign_boundaries"
            ),
            stop_after=stop_after,
        )
        return bool(run_state.running)

    def on_open_graph_editor() -> None:
        if view.results is None or getattr(view.results, "_graph", None) is None:
            report.value = "Nothing to edit yet: run the pipeline through at least Graph first."
            return
        from haemolynx.gui.graph_editor import GraphEditorState

        results = view.results
        state = GraphEditorState(
            graph=copy_graph(results._graph),
            voxel_size_zyx=tuple(
                float(v) for v in getattr(results, "_voxel_size_zyx", (1.0, 1.0, 1.0))
            ),
        )
        image_layer = _graph_editor_layer(IMAGE)
        if image_layer is not None:
            state.cost_field_from_mask(
                np.asanyarray(image_layer.data),
                use_memmap=bool(_settings().get("use_memmap_loading")),
            )
        graph_editor["state"] = state

        window = _GraphEditorWindow(
            on_add=lambda: _arm_graph_editor("add"),
            on_finish=_finish_graph_editor_branch,
            on_delete=lambda: _arm_graph_editor("delete"),
            on_regenerate=on_regenerate_from_edit,
        )
        graph_editor["window"] = window
        _attach_graph_editor_click_callbacks()
        window.show()

    def save_run_file(path: Path | str) -> bool:
        """Write the current viewer run to *path*. Returns whether it wrote."""
        if run_state.running:
            report.value = ALREADY_RUNNING
            return False
        try:
            snapshot = capture_run(
                checkpoints=checkpoints,
                results=view.results,
                settings=current_values(),
                skip_toggle_snapshot=skip_toggle_snapshot,
                show_results=bool(show_results.value),
                show_steps=bool(show_steps.value),
                report=str(report.value or ""),
                paused_after=run_state.paused_after,
            )
        except RunSnapshotError as error:
            report.value = str(error)
            return False
        dest = ensure_run_suffix(path)
        try:
            write_run_snapshot(dest, snapshot)
        except Exception as error:  # noqa: BLE001 - report in the panel
            report.value = f"Could not save run {dest}:\n{error}"
            return False
        last_run_path[0] = str(dest)
        report.value = f"Wrote run {dest}"
        return True

    def _strip_session_for_load() -> None:
        """Forget the live run so a loaded snapshot owns the panel."""
        stopping = run_state.cancel()
        if stopping:
            checkpoints.freeze()
            run_state.supersede()
            run_button.enabled = True
            # Whichever of the three was the one running: cancel() does not
            # say which, and re-enabling/resetting the others is a no-op.
            optimise_button.enabled = True
            optimise_bars.reset()
            optimise_fwhm_button.enabled = True
            optimise_fwhm_bars.reset()
            log_view.cancelled()
        optimise_review.clear()
        fwhm_review.clear()
        if viewer is not None:
            _clear_our_layers(viewer)
            reparent_arrow_length_slider()
            z_depth_slider._haemolynx_extent_ready = False
            z_depth_slider.setEnabled(False)
            arrow_length_slider.setEnabled(False)
            arrow_length_slider.setVisible(False)
            apply_view_z(force=True)
        view.results = None
        if boundaries is not None:
            boundaries.state.results = None
        try:
            discard_cached_artefacts_for_settings(
                current_values(), extra_paths=checkpoints.session_artefact_paths
            )
        except Exception:  # noqa: BLE001 - loading still proceeds
            logger.exception("could not discard cached artefacts before loading a run")
        checkpoints.clear()
        run_state.paused_after = None
        post_processing.forget()
        bars.reset()

    def load_run_file(path: Path | str) -> bool:
        """Replace the current session with the run stored at *path*."""
        try:
            snapshot = read_run_snapshot(path)
        except Exception as error:
            report.value = f"Could not load run {path}:\n{error}"
            return False
        _strip_session_for_load()
        loaded_paths.clear()
        loaded_config_dir[0] = Path(path).parent
        last_run_path[0] = str(path)
        # A run saved by an older version may name a retired setting or spell
        # a value the way it used to; the rows only take today's.
        for name, value in schema.upgrade(snapshot.settings).items():
            if name not in rows:
                continue
            if schema[name].kind == "path" and value is not None:
                loaded_paths[name] = value
            rows[name].value = display_value_for(schema[name], value)
        show_results.value = snapshot.show_results
        show_steps.value = snapshot.show_steps
        skip_toggle_snapshot.clear()
        skip_toggle_snapshot.update(snapshot.skip_toggle_snapshot)
        if not skip_toggle_snapshot:
            snapshot_skip_toggles()
        apply_prerequisites()
        apply_snapshot_to_checkpoints(checkpoints, snapshot)
        results = ResultLayers(
            show_steps=snapshot.show_steps,
            settings=snapshot.settings,
        )
        apply_snapshot_to_results(results, snapshot)
        view.results = results
        if boundaries is not None:
            boundaries.state.results = results
        if viewer is not None:
            _apply_layer_groups(viewer, replay_groups(snapshot))
            _after_layers_applied()
        try:
            resolved = _settings()
        except Exception:  # noqa: BLE001 - resume pickles are optional
            resolved = current_values()
        write_resume_artefacts(snapshot, resolved, checkpoints)
        if snapshot.last_tab_title:
            titles = [tab_widget.tabText(i) for i in range(tab_widget.count())]
            if snapshot.last_tab_title in titles:
                tab_widget.setCurrentIndex(titles.index(snapshot.last_tab_title))
        bars.start()
        bars.finish(f"Loaded run ({len(snapshot.stages)} stages)")
        refresh_revert_buttons()
        run_button.enabled = True
        report.value = (
            f"Loaded run from {path}: {len(snapshot.stages)} stages restored."
        )
        if snapshot.paused_after is not None:
            # Saved while paused: pick it up where it stopped, Continue ready.
            run_state.paused_after = snapshot.paused_after
            report.value += f" It was paused; Continue on {POST_PROCESSING_TAB} runs the rest."
            on_paused()
        post_processing.refresh()
        return True

    def on_save_run() -> None:
        from qtpy.QtWidgets import QFileDialog

        if run_state.running:
            report.value = ALREADY_RUNNING
            return
        suggested = last_run_path[0]
        if suggested == RUN_SNAPSHOT_FILENAME:
            suggested = default_run_path(current_values())
        path, _filter = QFileDialog.getSaveFileName(
            _dialog_parent(viewer),
            "Save this pipeline run",
            suggested,
            RUN_SNAPSHOT_FILTER,
        )
        if not path:
            return
        save_run_file(path)

    def on_load_run() -> None:
        from qtpy.QtWidgets import QFileDialog

        path, _filter = QFileDialog.getOpenFileName(
            _dialog_parent(viewer),
            "Open a HaemoLynx pipeline run",
            last_run_path[0],
            RUN_SNAPSHOT_FILTER,
        )
        if not path:
            return
        load_run_file(path)

    for title, button in revert_buttons.items():
        button.changed.connect(
            lambda *_args, tab=title: on_revert(tab)
        )

    load_button.changed.connect(on_load)
    edit_button.changed.connect(lambda *_args: on_open_graph_editor())
    save_button.changed.connect(on_save)
    optimise_button.changed.connect(lambda *_args: on_optimise_settings())
    check_image_button.changed.connect(lambda *_args: on_check_segmented_image())
    check_button.changed.connect(on_check)
    run_button.changed.connect(on_run)
    clear_button.changed.connect(lambda *_args: on_clear(ask=True))
    save_run_button.changed.connect(on_save_run)
    load_run_button.changed.connect(on_load_run)
    buttons = Container(
        widgets=[load_button, save_button, check_button, run_button, clear_button],
        layout="horizontal",
        labels=False,
    )
    run_file_row = QWidget()
    run_file_row.setObjectName("haemolynx_run_file_row")
    run_file_layout = QHBoxLayout(run_file_row)
    run_file_layout.setContentsMargins(0, 0, 0, 0)
    run_file_layout.addStretch(1)
    run_file_layout.addWidget(view_button.native)
    run_file_layout.addWidget(edit_button.native)
    # Editing lives in the "7. Post processing" tab now: the button keeps
    # its place in the row (and the floating Edit window its code) but is
    # hidden.
    edit_button.visible = False
    run_file_layout.addWidget(save_run_button.native)
    run_file_layout.addWidget(load_run_button.native)
    view_controls = Container(
        widgets=[show_results, show_steps],
        labels=True,
    )
    view_controls.native.setObjectName("haemolynx_view_controls")

    view_panel = QWidget()
    view_panel.setObjectName("haemolynx_view_panel")
    display_group = QGroupBox("Display")
    display_group.setObjectName("haemolynx_display_group")
    display_form = QFormLayout(display_group)
    z_depth_row = QWidget()
    z_depth_row.setObjectName("haemolynx_z_depth_row")
    z_depth_form = QFormLayout(z_depth_row)
    z_depth_form.setContentsMargins(0, 0, 0, 0)
    z_depth_form.addRow(z_depth_label, z_depth_slider)
    display_form.addRow(z_depth_row)
    vessel_draw_row = QWidget()
    vessel_draw_row.setObjectName("haemolynx_vessel_draw_row")
    vessel_draw_form = QFormLayout(vessel_draw_row)
    vessel_draw_form.setContentsMargins(0, 0, 0, 0)
    vessel_draw_form.addRow(vessel_draw_label, vessel_draw)
    display_form.addRow(vessel_draw_row)
    layer_set_row = QWidget()
    layer_set_row.setObjectName("haemolynx_layer_set_row")
    layer_set_form = QFormLayout(layer_set_row)
    layer_set_form.setContentsMargins(0, 0, 0, 0)
    layer_set_form.addRow(layer_set_label, layer_set_button)
    display_form.addRow(layer_set_row)
    # What the shown network's vessels are coloured by -- flow, log10 flow,
    # ... -- whether drawn as tubes or lines, without selecting the layer.
    colour_by = _VesselColourMenu(viewer, on_resize=lambda: fit_view_dock())
    display_form.addRow(colour_by.row)
    if viewer is not None:
        viewer._haemolynx_refresh_colour_by = colour_by.refresh

    # A sweep perturbation's grid sliders, one set per sweep, of which only
    # the one chosen under "Showing" is shown -- a slider for a network not on
    # screen moved flows nobody could see.
    sweep_group = QGroupBox("Sweep")
    sweep_group.setObjectName("haemolynx_sweep_group")
    sweep_group.setToolTip(SWEEP_TOOLTIP)
    sweep_layout = QVBoxLayout(sweep_group)
    sweep_group.setVisible(False)
    #: layer name -> (the perturbation it belongs to, its magicgui Container).
    sweep_controls: dict[str, tuple[str | None, Any]] = {}
    view_dock_holder: dict[str, Any] = {"dock": None}

    def fit_view_dock() -> None:
        """Grow or shrink the floating view panel to its contents."""
        dock = view_dock_holder["dock"]
        try:
            if dock is None or not dock.isFloating():
                return
            view_panel.adjustSize()
            hint = view_panel.sizeHint()
            dock.resize(max(dock.width(), int(hint.width())),
                        max(int(hint.height()) + 24, 220))
        except RuntimeError:
            logger.debug("view dock is gone", exc_info=True)

    def refresh_sweep_controls() -> None:
        shown = _layer_set_shown(viewer) if viewer is not None else None
        any_shown = False
        for key, container in sweep_controls.values():
            on = key == shown
            container.native.setVisible(on)
            any_shown = any_shown or on
        was_shown = not sweep_group.isHidden()
        sweep_group.setTitle(f"Sweep: {shown}" if any_shown and shown else "Sweep")
        sweep_group.setVisible(any_shown)
        if any_shown != was_shown:
            fit_view_dock()

    def add_sweep_controls(layer_name: str, key: str | None, container) -> None:
        remove_sweep_controls(layer_name)
        sweep_controls[layer_name] = (key, container)
        sweep_layout.addWidget(container.native)
        refresh_sweep_controls()

    def remove_sweep_controls(layer_name: str) -> None:
        entry = sweep_controls.pop(layer_name, None)
        if entry is not None:
            native = entry[1].native
            sweep_layout.removeWidget(native)
            native.setParent(None)
            native.deleteLater()
        refresh_sweep_controls()

    if viewer is not None:
        viewer._haemolynx_sweep_host = SimpleNamespace(
            add=add_sweep_controls, remove=remove_sweep_controls
        )

    refresh_layer_sets()
    display_form.addRow(scale_bar_box)

    def on_save_snapshot() -> None:
        folder = output_folder_from_settings(current_values())
        if folder is None:
            logger.warning(SNAPSHOT_NO_OUTPUT_FOLDER)
            report.value = SNAPSHOT_NO_OUTPUT_FOLDER
            return
        if viewer is None:
            report.value = SNAPSHOT_NO_VIEWER
            return
        try:
            path = write_viewer_snapshot(viewer, folder)
        except Exception as error:  # noqa: BLE001 - must not crash Qt
            logger.exception("could not write snapshot")
            report.value = f"Could not write snapshot:\n{error}"
            return
        report.value = f"Wrote snapshot {path}"

    snapshot_group = QGroupBox("Snapshot")
    snapshot_group.setObjectName("haemolynx_snapshot_group")
    snapshot_layout = QVBoxLayout(snapshot_group)
    snapshot_button = QPushButton("Save snapshot")
    snapshot_button.setObjectName("haemolynx_snapshot_button")
    snapshot_button.setToolTip(SNAPSHOT_TOOLTIP)
    snapshot_button.clicked.connect(on_save_snapshot)
    snapshot_layout.addWidget(snapshot_button)
    snapshot_group.setMinimumHeight(48)
    view_layout = QVBoxLayout(view_panel)
    view_layout.addWidget(display_group)
    view_layout.addWidget(sweep_group)
    view_layout.addWidget(snapshot_group)

    view_dock = None
    if viewer is not None:
        dock_kwargs = {
            "name": VIEW_DOCK_NAME,
            "area": "left",
            "allowed_areas": ["left", "right"],
        }
        try:
            view_dock = viewer.window.add_dock_widget(
                view_panel, add_vertical_stretch=False, **dock_kwargs
            )
        except TypeError:
            view_dock = viewer.window.add_dock_widget(view_panel, **dock_kwargs)
        try:
            view_dock.setObjectName("haemolynx_view_dock")
        except Exception:  # noqa: BLE001
            pass
        view_dock_holder["dock"] = view_dock
        _give_bottom_docks_the_corners(viewer)
        _float_dock_over_canvas(viewer, view_dock)
        try:
            _install_view_snap_buttons(viewer)
        except Exception:  # noqa: BLE001 - a missing overlay must not stop the panel
            logger.debug("could not add the view-snap buttons", exc_info=True)
    else:
        view_panel.setVisible(False)

    def on_reopen_view() -> None:
        """Bring the view panel back after its own close button hid it.

        A no-op while it is already open, so the button never steals focus
        or re-floats a dock the user has deliberately moved or docked.
        """
        if view_dock is None or view_dock.isVisible():
            return
        view_dock.show()
        _float_dock_over_canvas(viewer, view_dock)

    view_button.changed.connect(lambda *_args: on_reopen_view())

    panel = QWidget()
    # What the panel would send to a run, and what a run would report back,
    # for a test that cannot press buttons and wait.
    panel._haemolynx_values = current_values
    panel._haemolynx_progress = bars
    panel._haemolynx_log = log_view
    panel._haemolynx_log_dock = log_dock
    panel._haemolynx_run = on_run
    panel._haemolynx_clear = on_clear
    panel._haemolynx_revert = prepare_run_from
    panel._haemolynx_run_from = on_revert
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
    panel._haemolynx_z_depth_row = z_depth_row
    panel._haemolynx_vessel_draw = vessel_draw
    panel._haemolynx_vessel_draw_row = vessel_draw_row
    panel._haemolynx_layer_set_row = layer_set_row
    panel._haemolynx_layer_set_button = layer_set_button
    panel._haemolynx_layer_set_menu = layer_set_menu
    panel._haemolynx_choose_layer_set = choose_layer_set
    panel._haemolynx_colour_by = colour_by
    panel._haemolynx_view_panel = view_panel
    panel._haemolynx_view_dock = view_dock
    panel._haemolynx_view_button = view_button
    panel._haemolynx_reopen_view = on_reopen_view
    panel._haemolynx_scale_bar = scale_bar_box
    panel._haemolynx_display_group = display_group
    panel._haemolynx_snapshot_group = snapshot_group
    panel._haemolynx_sweep_group = sweep_group
    panel._haemolynx_sweep_controls = sweep_controls
    panel._haemolynx_snapshot_button = snapshot_button
    panel._haemolynx_apply_view_z = apply_view_z
    panel._haemolynx_data_for_pipeline = data_for_pipeline
    panel._haemolynx_arrow_length_slider = arrow_length_slider
    panel._haemolynx_arrow_length_host = arrow_length_host
    panel._haemolynx_reparent_arrow_length = reparent_arrow_length_slider
    panel._haemolynx_apply_z_filter = _apply_z_filter
    panel._haemolynx_after_layers_applied = _after_layers_applied
    panel._haemolynx_load_config = load_config_file
    panel._haemolynx_save_config = save_config_file
    panel._haemolynx_save_run = save_run_file
    panel._haemolynx_load_run = load_run_file
    panel._haemolynx_save_run_button = save_run_button
    panel._haemolynx_load_run_button = load_run_button
    panel._haemolynx_run_file_row = run_file_row
    panel._haemolynx_report = lambda: report.value
    panel._haemolynx_rows = lambda: rows
    panel._haemolynx_boundaries = boundaries
    panel._haemolynx_perturbations = perturbations
    panel._haemolynx_tabs = tab_widget
    panel._haemolynx_optimise_button = optimise_button
    panel._haemolynx_optimise_bars = optimise_bars
    panel._haemolynx_optimise_settings = on_optimise_settings
    panel._haemolynx_optimise_fwhm_button = optimise_fwhm_button
    panel._haemolynx_optimise_fwhm_bars = optimise_fwhm_bars
    panel._haemolynx_optimise_fwhm_settings = on_optimise_fwhm_settings
    panel._haemolynx_diameters_settings = diameters_settings
    panel._haemolynx_optimise_downsample = downsample_dropdown
    panel._haemolynx_optimise_choose_groups = choose_groups_checkbox
    panel._haemolynx_optimise_group_checkboxes = group_checkboxes
    panel._haemolynx_optimise_group_checkboxes_container = group_checkboxes_container
    panel._haemolynx_optimise_review = optimise_review
    panel._haemolynx_optimise_passes = passes_spinbox
    panel._haemolynx_optimise_fwhm_review = fwhm_review
    panel._haemolynx_optimise_fwhm_box = fwhm_optimise_box
    panel._haemolynx_optimise_fwhm_options = fwhm_options_toggle
    panel._haemolynx_optimise_fwhm_options_container = fwhm_options_container
    panel._haemolynx_optimise_fwhm_choose_groups = fwhm_choose_groups
    panel._haemolynx_optimise_fwhm_group_checkboxes = fwhm_group_checkboxes
    panel._haemolynx_optimise_fwhm_sample = fwhm_sample_dropdown
    panel._haemolynx_optimise_fwhm_passes = fwhm_passes_spinbox
    panel._haemolynx_check_image_button = check_image_button
    panel._haemolynx_raw_data_row = raw_data_row
    panel._haemolynx_raw_channel_row = raw_channel_row

    def expand_advanced(expanded: bool = True) -> None:
        """Open (or close) every Advanced button on the panel."""
        for disclosure in advanced_disclosures.values():
            disclosure.toggle(expanded)
        if perturbations is not None:
            for editor in perturbations.editors():
                editor.advanced.toggle(expanded)

    panel._haemolynx_advanced = advanced_disclosures
    panel._haemolynx_diameter_source = diameter_source
    panel._haemolynx_expand_advanced = expand_advanced
    panel._haemolynx_shared_ilastik_block = shared_ilastik_block
    panel._haemolynx_input_boxes = input_boxes
    panel._haemolynx_check_segmented_image = on_check_segmented_image
    panel._haemolynx_edit_button = edit_button
    panel._haemolynx_graph_editor = graph_editor
    panel._haemolynx_open_graph_editor = on_open_graph_editor
    panel._haemolynx_regenerate_from_edit = on_regenerate_from_edit
    panel._haemolynx_finish_graph_editor_branch = _finish_graph_editor_branch
    panel._haemolynx_post_processing = post_processing
    layout = QVBoxLayout(panel)
    if layer_row is not None:
        layout.addWidget(layer_row.native)
    layout.addWidget(tab_widget, 1)
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
    layout.addWidget(run_file_row)
    layout.addWidget(bars.native)
    # Optimise settings' own bars, beside the pipeline run's: the button
    # itself is on the Input tab, but its progress is chrome shared across
    # every tab, same as the pipeline run's bars above.
    layout.addWidget(optimise_bars.native)
    # "Optimise FWHM settings" is a plain Qt widget too (see its own
    # construction above), so its bars join the same shared chrome rather
    # than the magicgui Diameters-tab Container, which the button itself
    # (a real magicgui PushButton) was inserted into directly.
    layout.addWidget(optimise_fwhm_bars.native)
    layout.addWidget(report.native)
    if viewer is None:
        view_panel.setParent(panel)
        view_panel.setVisible(False)
    else:
        _after_layers_applied()
    refresh_revert_buttons()
    post_processing.refresh()
    # Applied once, panel-wide, after every row -- including the buttons and
    # dropdowns appended onto the Input tab above -- is in place.
    _wrap_row_labels(panel)
    return panel
