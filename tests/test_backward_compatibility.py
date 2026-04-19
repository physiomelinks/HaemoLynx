import pytest
import warnings
import networkx as nx
import numpy as np

def test_hemodynamics_backward_compatibility_alias():
    """Ensure legacy 'hemodynamics' imports route to 'haemodynamics' and warn."""
    with pytest.warns(DeprecationWarning, match="The 'ImageLynx.hemodynamics' module has been renamed to 'ImageLynx.haemodynamics'"):
        # Attempt to import the old spelling
        from ImageLynx.hemodynamics import PoiseuilleModel
        
        # Verify it successfully loaded the class from the new spelling
        assert PoiseuilleModel is not None

def test_legacy_graph_weight_attribute_fallback():
    """Ensure statistics and visualization modules do not crash when fed a legacy graph containing only 'weight'."""
    from ImageLynx.statistics import compute_basic_statistics
    
    legacy_graph = nx.MultiGraph()
    legacy_graph.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    legacy_graph.add_node(1, pos=np.array([1.0, 1.0, 1.0]))
    
    # Intentionally only adding 'weight' (the old standard, no 'resistance')
    legacy_graph.add_edge(0, 1, key=0, weight=150.0, length=10.0, branch_order="capillary")
    
    # This should succeed via the fallback logic, rather than throwing a KeyError
    stats = compute_basic_statistics(legacy_graph, is_multigraph=True)
    assert stats is not None
    assert stats["Total Edges"] == 1
