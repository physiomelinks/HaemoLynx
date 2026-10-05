"""How the canvas lights every surface, for the view window.

The lamp is vispy's Phong light on each surface mesh: a white ambient fill,
a white diffuse lamp, a white specular highlight, and how tight that
highlight is. Shading says whether that light is off, flat, or smooth.
Light direction stays with the camera, which already keeps the lamp over
the viewer's shoulder.

Nothing here is applied until the user moves a control, so a surface that
was given its own shading (the tissue mesh is flat, the tubes are smooth)
keeps it until then.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

logger = logging.getLogger(__name__)

#: What a surface layer accepts as ``shading``.
SHADING_MODES = ("none", "flat", "smooth")

#: Vispy's own ``ShadingFilter`` defaults, so an untouched box matches the
#: canvas a surface already has.
DEFAULT_AMBIENT = 0.25
DEFAULT_DIFFUSE = 0.7
DEFAULT_SPECULAR = 0.25
DEFAULT_SHININESS = 100.0


@dataclass(frozen=True)
class CanvasIllumination:
    """The lamp the view window is currently asking for."""

    shading: str = "smooth"
    ambient: float = DEFAULT_AMBIENT
    diffuse: float = DEFAULT_DIFFUSE
    specular: float = DEFAULT_SPECULAR
    shininess: float = DEFAULT_SHININESS


def white_light(intensity: float) -> tuple[float, float, float, float]:
    """A white light whose alpha is *intensity*, clamped to 0–1."""
    level = min(1.0, max(0.0, float(intensity)))
    return (1.0, 1.0, 1.0, level)


def apply_shading_filter(
    filt,
    illumination: CanvasIllumination,
) -> None:
    """Write *illumination*'s lamp onto one vispy shading filter."""
    filt.ambient_light = white_light(illumination.ambient)
    filt.diffuse_light = white_light(illumination.diffuse)
    filt.specular_light = white_light(illumination.specular)
    filt.shininess = max(0.0, float(illumination.shininess))


def surface_layers(layers: Sequence[Any]) -> list[Any]:
    """Layers the canvas lights as meshes."""
    return [layer for layer in layers if type(layer).__name__ == "Surface"]


def _visual_for(viewer, layer):
    window = getattr(viewer, "window", None)
    qt_viewer = getattr(window, "_qt_viewer", None) if window is not None else None
    mapping = getattr(qt_viewer, "layer_to_visual", None) if qt_viewer is not None else None
    if mapping is None:
        return None
    return mapping.get(layer)


def apply_canvas_illumination(viewer, illumination: CanvasIllumination) -> None:
    """Light every surface on *viewer* with *illumination*.

    Shading ``none`` turns the lamp off. The other three modes turn it on
    and set its strength. A layer that cannot take a shading is skipped.
    """
    if viewer is None or illumination.shading not in SHADING_MODES:
        return
    for layer in surface_layers(viewer.layers):
        try:
            layer.shading = illumination.shading
        except Exception:
            logger.debug("could not shade %s", getattr(layer, "name", layer), exc_info=True)
            continue
        visual = _visual_for(viewer, layer)
        node = getattr(visual, "node", None)
        filt = getattr(node, "shading_filter", None)
        if filt is None:
            continue
        apply_shading_filter(filt, illumination)
    window = getattr(viewer, "window", None)
    qt_viewer = getattr(window, "_qt_viewer", None) if window is not None else None
    canvas = getattr(qt_viewer, "canvas", None)
    native = getattr(canvas, "native", None)
    update = getattr(native, "update", None)
    if callable(update):
        update()


def illumination_box(viewer):
    """The view window's Illumination group, wired to *viewer*'s surfaces.

    Moving a control lights every surface already open, and every surface
    added afterwards. Until then the box only displays the defaults.
    """
    from types import SimpleNamespace

    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QComboBox, QFormLayout, QGroupBox, QLabel
    from superqt import QDoubleSlider

    from haemolynx.gui.chrome_tooltips import ILLUMINATION_TOOLTIPS

    group = QGroupBox("Illumination")
    group.setObjectName("haemolynx_illumination_group")
    form = QFormLayout(group)

    shading = QComboBox()
    shading.setObjectName("haemolynx_illumination_shading")
    for label, mode in (("None", "none"), ("Flat", "flat"), ("Smooth", "smooth")):
        shading.addItem(label, mode)
    shading.setCurrentIndex(shading.findData("smooth"))
    shading.setToolTip(ILLUMINATION_TOOLTIPS["shading"])

    def slider(name: str, low: float, high: float, value: float, step: float):
        widget = QDoubleSlider(Qt.Orientation.Horizontal)
        widget.setObjectName(f"haemolynx_illumination_{name}")
        widget.setRange(low, high)
        widget.setSingleStep(step)
        widget.setValue(value)
        widget.setToolTip(ILLUMINATION_TOOLTIPS[name])
        return widget

    ambient = slider("ambient", 0.0, 1.0, DEFAULT_AMBIENT, 0.05)
    diffuse = slider("diffuse", 0.0, 1.0, DEFAULT_DIFFUSE, 0.05)
    specular = slider("specular", 0.0, 1.0, DEFAULT_SPECULAR, 0.05)
    shininess = slider("shininess", 1.0, 256.0, DEFAULT_SHININESS, 1.0)

    def add(label: str, widget) -> None:
        caption = QLabel(label)
        caption.setToolTip(widget.toolTip())
        form.addRow(caption, widget)

    add("Shading", shading)
    add("Ambient", ambient)
    add("Diffuse", diffuse)
    add("Specular", specular)
    add("Shininess", shininess)

    box = SimpleNamespace(
        group=group,
        shading=shading,
        ambient=ambient,
        diffuse=diffuse,
        specular=specular,
        shininess=shininess,
        adjusted=False,
    )

    def state() -> CanvasIllumination:
        mode = shading.currentData()
        return CanvasIllumination(
            shading=mode if mode in SHADING_MODES else "smooth",
            ambient=float(ambient.value()),
            diffuse=float(diffuse.value()),
            specular=float(specular.value()),
            shininess=float(shininess.value()),
        )

    def sync_enabled() -> None:
        lit = shading.currentData() != "none"
        for widget in (ambient, diffuse, specular, shininess):
            widget.setEnabled(lit)

    def push(*_args) -> None:
        box.adjusted = True
        sync_enabled()
        apply_canvas_illumination(viewer, state())

    box.state = state
    sync_enabled()
    shading.currentIndexChanged.connect(push)
    for widget in (ambient, diffuse, specular, shininess):
        widget.valueChanged.connect(push)

    if viewer is not None:
        from qtpy.QtCore import QTimer

        def on_inserted(_event=None) -> None:
            # After add_surface finishes applying its own shading keyword,
            # so a surface opened once the lamp has been moved takes the lamp.
            if box.adjusted:
                QTimer.singleShot(0, lambda: apply_canvas_illumination(viewer, state()))

        viewer.layers.events.inserted.connect(on_inserted)
    return box
