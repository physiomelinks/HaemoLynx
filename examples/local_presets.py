#!/usr/bin/env python3
"""User-local preset extensions for resistance pipeline.

Define `LOCAL_PRESET_DEFINITIONS` to add or override presets without editing
`examples/presets.py`.

Example:

LOCAL_PRESET_DEFINITIONS = {
    "my_lab_default": {
        "extends": "automated_assignment",
        "description": "Lab-default automated assignment profile.",
        "overrides": {
            "VERBOSE_LOGGING": True,
            "SMALL_VESSEL_MASK_MIN_OVERLAP_FRACTION": 0.65,
        },
    },
}
"""

LOCAL_PRESET_DEFINITIONS: dict[str, dict[str, object]] = {}
