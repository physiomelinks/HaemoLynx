"""Settings for ``simple_network_haemodynamics.py``.

Every setting in ``simple_network_config.yaml`` has an entry here, and nothing
else is configurable. Regenerate the config file after changing this schema::

    python -m examples.regenerate_configs
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from haemolynx.parsers import Schema, Setting

SCHEMA = Schema(
    [
        # --- Output ---------------------------------------------------------
        Setting(
            "output_dir",
            "path",
            "examples/outputs/simple_network",
            "Directory for the exported VTK files",
            "Output",
        ),
        # --- Vessel geometry ------------------------------------------------
        Setting(
            "diameter_by_branch_order",
            "mapping",
            {"Art1": 20.0, "B01": 5.0, "Ven1": 30.0},
            "Vessel diameter for each branch-order label",
            "Vessel geometry",
            unit="um",
        ),
        # --- Boundary conditions --------------------------------------------
        Setting(
            "inlet_coordinate_zyx",
            "float_list",
            (0.0, 0.0, 0.0),
            "Point whose nearest terminal node becomes the inlet",
            "Boundary conditions",
            unit="um",
        ),
        Setting(
            "outlet_coordinate_zyx",
            "float_list",
            (0.0, 0.0, 600.0),
            "Point whose nearest terminal node becomes the outlet",
            "Boundary conditions",
            unit="um",
        ),
        Setting(
            "inlet_pressure_pa",
            "float",
            6000.0,
            "Pressure imposed at the inlet node",
            "Boundary conditions",
            unit="Pa",
            minimum=0.0,
        ),
        Setting(
            "outlet_pressure_pa",
            "float",
            1000.0,
            "Pressure imposed at the outlet node",
            "Boundary conditions",
            unit="Pa",
            minimum=0.0,
        ),
    ],
    title="Minimal network haemodynamics",
    description=(
        "Settings for examples/simple_network_haemodynamics.py.\n"
        "Edit values here; the script reads them and never hard-codes its own."
    ),
)
