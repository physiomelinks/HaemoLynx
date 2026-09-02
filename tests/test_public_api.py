"""Guards on the advertised public API of every subpackage.

`haemolynx.graph.__all__` listed two names that never existed
(`create_merged_edge_attributes_simple` / `_full`, leftovers from a reverted
split), so `from haemolynx.graph import *` raised AttributeError. Nothing
caught it because no test performed a star-import or checked `__all__`.

The same class of bug runs the other way too: `statistics/__init__.py` did not
re-export `compute_weighted_betweenness_summary` or
`compute_weighted_communities_summary`, which `pipeline/stages.py` calls as
`statistics.compute_weighted_betweenness_summary(...)`, so the export stage
raised AttributeError on the one configuration that reaches them. Both
directions are checked here.
"""
import ast
import importlib
from pathlib import Path

import pytest

SUBPACKAGES = [
    "haemolynx",
    "haemolynx.graph",
    "haemolynx.io",
    "haemolynx.haemodynamics",
    "haemolynx.parsers",
    "haemolynx.pipeline",
    "haemolynx.preprocessing",
    "haemolynx.statistics",
    "haemolynx.visualization",
]

#: The subpackage names a module can import as a whole, `from haemolynx import io`.
SUBPACKAGE_NAMES = {name.partition(".")[2] for name in SUBPACKAGES if "." in name}


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


def _subpackage_attributes_used_by(source_path: Path) -> dict[str, set[str]]:
    """Which subpackage attributes *source_path* reaches through the namespace.

    Maps `haemolynx.statistics` -> {"export_statistics_to_csv", ...} for every
    `statistics.name` in a module that imported the subpackage as a whole. A
    local name that is assigned anywhere in the module is dropped, because
    `name.attr` no longer necessarily means the subpackage.
    """
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    local_to_subpackage: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "haemolynx" and node.level == 0:
            for alias in node.names:
                if alias.name in SUBPACKAGE_NAMES:
                    local_to_subpackage[alias.asname or alias.name] = f"haemolynx.{alias.name}"
    # A class attribute called `statistics` -- `Solution` has one -- is not a
    # rebinding of the imported name, so it must not disqualify the module.
    class_attributes = {
        target.id
        for class_def in ast.walk(tree)
        if isinstance(class_def, ast.ClassDef)
        for node in class_def.body
        for target in (
            [node.target] if isinstance(node, ast.AnnAssign)
            else node.targets if isinstance(node, ast.Assign)
            else []
        )
        if isinstance(target, ast.Name)
    }
    shadowed = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
    } - class_attributes
    used: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or not isinstance(node.value, ast.Name):
            continue
        local = node.value.id
        if local in shadowed or local not in local_to_subpackage:
            continue
        used.setdefault(local_to_subpackage[local], set()).add(node.attr)
    return used


def test_every_attribute_a_module_reaches_through_a_subpackage_exists():
    """The other direction: a name a caller uses must actually be re-exported.

    `pipeline/stages.py` calls `statistics.compute_weighted_betweenness_summary`
    and `statistics.compute_weighted_communities_summary`; neither was in
    `statistics/__init__.py`, so the export stage raised AttributeError
    whenever it took that branch. `__all__` checks cannot see this — the names
    were absent from the package altogether.
    """
    package_root = Path(importlib.import_module("haemolynx").__file__).parent
    offenders = []
    for source_path in sorted(package_root.rglob("*.py")):
        for subpackage, names in _subpackage_attributes_used_by(source_path).items():
            module = importlib.import_module(subpackage)
            offenders += [
                f"{source_path.relative_to(package_root).as_posix()} -> {subpackage}.{name}"
                for name in sorted(names)
                if not hasattr(module, name)
            ]
    assert offenders == [], (
        "these modules reach a subpackage attribute that does not exist; "
        f"re-export it from the subpackage's __init__: {offenders}"
    )


def test_path_length_has_exactly_one_public_name():
    """`calculate_voxel_path_length` was a second public name for the same behaviour."""
    import haemolynx.graph as graph

    assert hasattr(graph, "calculate_path_length")
    assert not hasattr(graph, "calculate_voxel_path_length")


def test_calculate_path_length_tolerates_empty_and_none():
    """The removed twin's only real difference was its falsy-input guard."""
    from haemolynx.graph import calculate_path_length

    assert calculate_path_length([]) == 0.0
    assert calculate_path_length(None) == 0.0
    assert calculate_path_length([(0.0, 0.0, 0.0)]) == 0.0
    assert calculate_path_length([(0.0, 0.0, 0.0), (0.0, 0.0, 2.5)]) == pytest.approx(2.5)


def test_every_module_can_be_imported_by_normal_syntax():
    """A module whose name is not an identifier is unreachable by `import`.

    `statistics/3D_distances.py` started with a digit, so the package had to
    reach it through `importlib.import_module` and no user could write
    `from haemolynx.statistics import 3D_distances` at all. It is now
    `three_dim_distances`; this catches the next one.
    """
    import pkgutil

    import haemolynx

    offenders = [
        info.name
        for info in pkgutil.walk_packages(haemolynx.__path__, prefix="haemolynx.")
        if not info.name.rpartition(".")[2].isidentifier()
    ]
    assert offenders == [], (
        f"module names that cannot be imported normally: {offenders}. Rename them; "
        "a digit or a dash makes the module reachable only through importlib."
    )


def test_the_cell_distance_module_is_a_normal_import():
    from haemolynx.statistics import three_dim_distances

    assert callable(three_dim_distances.run_3d_measurement_to_cell_mask)
