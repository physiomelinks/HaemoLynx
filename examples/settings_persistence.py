"""Helpers for persisting selected nodes to the example settings file."""
from __future__ import annotations

import re
from pathlib import Path


def _replace_settings_assignment(
    file_text: str,
    variable_name: str,
    replacement_value_literal: str,
) -> tuple[str, bool]:
    """Replace one top-level settings assignment while keeping annotation/prefix."""
    pattern = re.compile(
        rf"^(?P<prefix>\s*{re.escape(variable_name)}\s*(?::[^\n=]+)?=\s*)(?P<value>[^\n]*)$",
        flags=re.MULTILINE,
    )
    match = pattern.search(file_text)
    if match is None:
        return file_text, False
    updated_text = (
        file_text[: match.start()]
        + f"{match.group('prefix')}{replacement_value_literal}"
        + file_text[match.end() :]
    )
    return updated_text, True


def persist_automated_io_assignment_to_settings_file(
    *,
    settings_file_path: Path,
    assigned_start_nodes: list[int],
    assigned_output_nodes: list[int],
) -> bool:
    """Persist automated node IDs and disable automation for subsequent runs."""
    if not settings_file_path.exists():
        print(
            "Could not persist automated I/O assignment: settings file not found at "
            f"{settings_file_path}."
        )
        return False

    start_nodes_literal = repr([int(node_id) for node_id in assigned_start_nodes])
    output_nodes_literal = repr([int(node_id) for node_id in assigned_output_nodes])
    updated_text = settings_file_path.read_text(encoding="utf-8")

    updated_text, ok_start = _replace_settings_assignment(
        updated_text,
        "STARTING_NODES",
        start_nodes_literal,
    )
    updated_text, ok_output = _replace_settings_assignment(
        updated_text,
        "OUTPUT_NODES",
        output_nodes_literal,
    )
    updated_text, ok_automated_flag = _replace_settings_assignment(
        updated_text,
        "AUTOMATED_VESSEL_ASSIGNMENT",
        "False",
    )

    if not (ok_start and ok_output and ok_automated_flag):
        print(
            "Could not persist automated I/O assignment because one or more "
            "settings keys were not found "
            "(required: STARTING_NODES, OUTPUT_NODES, AUTOMATED_VESSEL_ASSIGNMENT)."
        )
        return False

    settings_file_path.write_text(updated_text, encoding="utf-8")
    print(
        "Persisted automated I/O assignment to settings file and set "
        "AUTOMATED_VESSEL_ASSIGNMENT=False for next run."
    )
    print(
        f"Persisted {len(assigned_start_nodes)} STARTING_NODES and "
        f"{len(assigned_output_nodes)} OUTPUT_NODES to {settings_file_path}."
    )
    return True


def persist_small_vessel_boundary_assignment_to_settings_file(
    *,
    settings_file_path: Path,
    assigned_arteriole_boundary_nodes: list[int],
    assigned_venule_boundary_nodes: list[int],
) -> bool:
    """Persist small-vessel boundary nodes and disable auto boundary assignment."""
    if not settings_file_path.exists():
        print(
            "Could not persist small-vessel boundary assignment: settings file not found at "
            f"{settings_file_path}."
        )
        return False

    arteriole_nodes_literal = repr(
        [int(node_id) for node_id in assigned_arteriole_boundary_nodes]
    )
    venule_nodes_literal = repr(
        [int(node_id) for node_id in assigned_venule_boundary_nodes]
    )
    updated_text = settings_file_path.read_text(encoding="utf-8")

    updated_text, ok_arteriole = _replace_settings_assignment(
        updated_text,
        "ARTERIOLE_BOUNDARY_NODES",
        arteriole_nodes_literal,
    )
    updated_text, ok_venule = _replace_settings_assignment(
        updated_text,
        "VENULE_BOUNDARY_NODES",
        venule_nodes_literal,
    )
    updated_text, ok_small_mask_flag = _replace_settings_assignment(
        updated_text,
        "USE_SMALL_VESSEL_MASKS_FOR_BOUNDARY_ASSIGNMENT",
        "False",
    )

    if not (ok_arteriole and ok_venule and ok_small_mask_flag):
        print(
            "Could not persist small-vessel boundary assignment because one or more "
            "settings keys were not found "
            "(required: ARTERIOLE_BOUNDARY_NODES, VENULE_BOUNDARY_NODES, "
            "USE_SMALL_VESSEL_MASKS_FOR_BOUNDARY_ASSIGNMENT)."
        )
        return False

    settings_file_path.write_text(updated_text, encoding="utf-8")
    print(
        "Persisted small-vessel boundary assignment to settings file and set "
        "USE_SMALL_VESSEL_MASKS_FOR_BOUNDARY_ASSIGNMENT=False for next run."
    )
    print(
        f"Persisted {len(assigned_arteriole_boundary_nodes)} "
        f"ARTERIOLE_BOUNDARY_NODES and {len(assigned_venule_boundary_nodes)} "
        f"VENULE_BOUNDARY_NODES to {settings_file_path}."
    )
    return True
