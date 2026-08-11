"""What the About panel says, checked without a display.

It exists for two reasons. It answers the questions the settings panel cannot
-- where the colour controls are, and what a config file is for -- and it is
the plugin's second widget, which is what makes napari title the Plugins menu
"HaemoLynx" instead of "Pipeline settings (HaemoLynx)". Both are load-bearing,
so both are pinned: `tests/test_napari_manifest.py` covers the menu, this
covers the text.
"""
from __future__ import annotations

import haemolynx
from haemolynx.gui._widget import about_text
from haemolynx.pipeline import default_schema


def test_it_names_the_installed_version():
    assert haemolynx.__version__ in about_text()


def test_it_counts_the_settings_this_build_actually_has():
    assert str(len(list(default_schema()))) in about_text()


def test_it_says_where_the_colour_controls_are():
    """The panel is for the pipeline; colouring is napari's layer controls.

    That split is deliberate and was asked for, and it is the thing people
    look for in the wrong place, so the About text has to say it.
    """
    text = about_text().lower()
    assert "layer controls" in text
    assert "left" in text


def test_it_says_how_a_config_repeats_on_the_command_line():
    assert "--config" in about_text()


def test_it_is_plain_text_a_read_only_box_can_show():
    text = about_text()
    assert text.strip()
    assert "{" not in text and "}" not in text, "an unfilled format placeholder"


def test_it_credits_the_authors():
    """The credit line is the reason several people will open this panel."""
    text = about_text()
    for name in ("Finbar Argus", "Harvey Davis", "Animus Laboratory"):
        assert name in text, f"{name} is missing from the About text"
    assert text.rstrip().endswith(
        "Created by Finbar Argus, Harvey Davis, and the Animus Laboratory."
    ), "the credit should be the last thing the panel says"


def test_it_describes_the_graph_as_solver_input_rather_than_solved_output():
    """HaemoLynx builds the network; a solver is what turns it into flows.

    Saying it "gives haemodynamic edge weights" on its own reads as though the
    haemodynamics fall out of the image, which overstates what this does.
    """
    text = about_text()
    assert "fed" in text and "haemodynamic solver" in text
