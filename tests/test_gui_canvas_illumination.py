"""The lamp written onto a surface, without a viewer."""
from __future__ import annotations

import pytest

from haemolynx.gui.canvas_illumination import (
    CanvasIllumination,
    apply_shading_filter,
    white_light,
)


class _Filter:
    pass


def test_white_light_is_white_and_clamped():
    assert white_light(0.25) == (1.0, 1.0, 1.0, 0.25)
    assert white_light(-1) == (1.0, 1.0, 1.0, 0.0)
    assert white_light(4) == (1.0, 1.0, 1.0, 1.0)


def test_apply_shading_filter_writes_the_lamp():
    filt = _Filter()
    apply_shading_filter(
        filt,
        CanvasIllumination(
            shading="smooth", ambient=0.1, diffuse=0.4, specular=0.8, shininess=40
        ),
    )
    assert filt.ambient_light == (1.0, 1.0, 1.0, 0.1)
    assert filt.diffuse_light == (1.0, 1.0, 1.0, 0.4)
    assert filt.specular_light == (1.0, 1.0, 1.0, 0.8)
    assert filt.shininess == 40
