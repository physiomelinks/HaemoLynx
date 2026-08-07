"""Guards on the advertised public API of every subpackage.

`ImageLynx.graph.__all__` listed two names that never existed
(`create_merged_edge_attributes_simple` / `_full`, leftovers from a reverted
split), so `from ImageLynx.graph import *` raised AttributeError. Nothing
caught it because no test performed a star-import or checked `__all__`.
"""
import importlib

import pytest

SUBPACKAGES = [
    "ImageLynx",
    "ImageLynx.graph",
    "ImageLynx.io",
    "ImageLynx.haemodynamics",
    "ImageLynx.parsers",
    "ImageLynx.pipeline",
    "ImageLynx.preprocessing",
    "ImageLynx.statistics",
    "ImageLynx.visualization",
]


@pytest.mark.parametrize("module_name", SUBPACKAGES)
def test_every_name_in_all_is_importable(module_name):
    """`__all__` must not advertise names the package does not define."""
    module = importlib.import_module(module_name)
    missing = [name for name in getattr(module, "__all__", []) if not hasattr(module, name)]
    assert not missing, f"{module_name}.__all__ names undefined attributes: {missing}"


@pytest.mark.parametrize("module_name", SUBPACKAGES)
def test_star_import_succeeds(module_name):
    """A star-import must not raise — this is what the phantom names broke."""
    namespace: dict = {}
    exec(f"from {module_name} import *", namespace)  # noqa: S102
    module = importlib.import_module(module_name)
    for name in getattr(module, "__all__", []):
        assert name in namespace, f"{name} missing after star-import of {module_name}"


@pytest.mark.parametrize("module_name", SUBPACKAGES)
def test_all_has_no_duplicates(module_name):
    module = importlib.import_module(module_name)
    names = list(getattr(module, "__all__", []))
    duplicates = {name for name in names if names.count(name) > 1}
    assert not duplicates, f"{module_name}.__all__ lists duplicates: {sorted(duplicates)}"


def test_path_length_has_exactly_one_public_name():
    """`calculate_voxel_path_length` was a second public name for the same behaviour."""
    import ImageLynx.graph as graph

    assert hasattr(graph, "calculate_path_length")
    assert not hasattr(graph, "calculate_voxel_path_length")


def test_calculate_path_length_tolerates_empty_and_none():
    """The removed twin's only real difference was its falsy-input guard."""
    from ImageLynx.graph import calculate_path_length

    assert calculate_path_length([]) == 0.0
    assert calculate_path_length(None) == 0.0
    assert calculate_path_length([(0.0, 0.0, 0.0)]) == 0.0
    assert calculate_path_length([(0.0, 0.0, 0.0), (0.0, 0.0, 2.5)]) == pytest.approx(2.5)
