"""Reading a settings row's value must never crash the whole panel.

`LiteralEvalLineEdit` (the widget behind the ``int_list``/``float_list``/
``mapping``/``any`` schema kinds) evaluates its text as a Python literal on
every read of ``.value``, including this dict comprehension's own read of
*every other* row whenever any one row changes (see
``settings_widget.current_values`` and ``_boundary_controls.current_values``
in ``haemolynx.gui._widget``). Text that is not a complete literal -- a box
cleared by hand, or simply mid-edit while typing one out -- raises
``SyntaxError`` or ``ValueError`` there instead of returning, which used to
propagate out of the panel's own ``changed`` callbacks and disable
prerequisite handling for every row until the offending box was fixed.

These tests do not need napari, Qt or a display: `_safe_widget_value` takes
any object with a `.value` property, so a plain stand-in is enough.
"""
from __future__ import annotations

from haemolynx.gui._widget import _safe_widget_value


class _FakeWidget:
    def __init__(self, value=None, raises=None):
        self._value = value
        self._raises = raises

    @property
    def value(self):
        if self._raises is not None:
            raise self._raises
        return self._value


def test_it_returns_the_widgets_value_when_reading_it_does_not_raise():
    assert _safe_widget_value(_FakeWidget(value=[1, 2])) == [1, 2]
    assert _safe_widget_value(_FakeWidget(value=0.97)) == 0.97
    assert _safe_widget_value(_FakeWidget(value=False)) is False


def test_it_treats_an_empty_literal_eval_line_edit_as_unset():
    """``ast.literal_eval("")`` raises `SyntaxError`, e.g. a cleared box."""
    widget = _FakeWidget(raises=SyntaxError("invalid syntax (<unknown>, line 0)"))
    assert _safe_widget_value(widget) == ""


def test_it_treats_a_mid_edit_literal_eval_line_edit_as_unset():
    """Every keystroke of typing a list/dict out fires `changed` too, and a
    partial literal like ``"[1, 2,"`` is as invalid as an empty box until the
    user finishes it."""
    widget = _FakeWidget(raises=SyntaxError("unexpected EOF while parsing"))
    assert _safe_widget_value(widget) == ""


def test_it_treats_a_non_literal_expression_as_unset():
    """``ast.literal_eval`` raises `ValueError`, not `SyntaxError`, for
    syntactically valid but non-literal text such as ``"1+1"`` or ``"nan"``
    (the latter a real value this GUI has produced before -- see
    `tests/test_gui_widget.py`'s own docstring)."""
    widget = _FakeWidget(raises=ValueError("malformed node or string"))
    assert _safe_widget_value(widget) == ""
