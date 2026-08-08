"""The napari plugin: a settings form built from the schema.

Importing this package pulls in nothing GUI-related. :mod:`haemolynx.gui.form`
is the pure mapping from schema to form rows, and the widget itself is imported
by napari through ``napari.yaml`` when the panel is opened.

Install the GUI with ``pip install "HaemoLynx[napari]"``, which needs Python
3.11 or newer -- napari's own floor, above this library's 3.9.
"""
from .form import Field, field_for, fields_for, label_for, sections_for, values_from

__all__ = [
    "Field",
    "field_for",
    "fields_for",
    "label_for",
    "sections_for",
    "values_from",
]
