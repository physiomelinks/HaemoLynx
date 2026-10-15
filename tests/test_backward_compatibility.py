import pytest

def test_hemodynamics_backward_compatibility_alias():
    """Ensure legacy 'hemodynamics' imports route to 'haemodynamics' and warn."""
    with pytest.warns(DeprecationWarning, match="The 'ImageLynx.hemodynamics' module has been renamed to 'ImageLynx.haemodynamics'"):
        # Attempt to import the old spelling
        from ImageLynx.hemodynamics import PoiseuilleModel
        
        # Verify it successfully loaded the class from the new spelling
        assert PoiseuilleModel is not None
