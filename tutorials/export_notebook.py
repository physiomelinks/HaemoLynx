"""Export pipeline_tutorial.ipynb to a runnable Python script."""
from __future__ import annotations

from pathlib import Path

GENERATED_HEADER = """#!/usr/bin/env python3
# AUTO-GENERATED from tutorials/pipeline_tutorial.ipynb — do not edit manually.
# Regenerate: pytest tests/integration/test_pipeline_tutorial.py

"""


def export_pipeline_tutorial_script(
    notebook_path: Path,
    output_path: Path,
) -> Path:
    """nbconvert the tutorial notebook to ``output_path``."""
    import nbformat
    from nbconvert import PythonExporter

    notebook = nbformat.read(notebook_path, as_version=nbformat.NO_CONVERT)
    body, _resources = PythonExporter().from_notebook_node(notebook)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = body.splitlines()
    while lines and lines[0].startswith("#!"):
        lines.pop(0)
    body = GENERATED_HEADER + "\n".join(lines)
    if body and not body.endswith("\n"):
        body += "\n"
    output_path.write_text(body, encoding="utf-8")
    return output_path
