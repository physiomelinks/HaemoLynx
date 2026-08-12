"""A run through HaemoLynx's public API, for checking an installed package.

Run directly (``python installed_package_smoke.py``) from any directory that is
not the repository: everything it needs must come from the installed package.
It prints one ``RESULT {json}`` line that
``tests/test_installed_package.py`` reads back, and the CI job runs it against
a wheel installed into an empty environment.
"""
import json
from pathlib import Path

import networkx as nx
import numpy as np

import haemolynx
# Every subpackage, imported for its own sake: one missing from the wheel
# fails here rather than in whichever call happens to reach it first.
from haemolynx import graph, haemodynamics, io, preprocessing, statistics, visualization
from haemolynx.pipeline import default_schema, write_default_config

assert graph and io and preprocessing  # imported above for the check, used below via haemodynamics

here = Path.cwd()

# The settings schema ships with the package, so a config can be written
# without cloning the repository.
config_path = write_default_config(here / "config.yaml")
assert config_path.exists() and config_path.stat().st_size > 0
schema = default_schema()
assert len(schema.names) > 100

# A network, solved and exported.
G = nx.MultiGraph()
positions = {0: (0.0, 0.0, 0.0), 1: (0.0, 0.0, 100.0), 2: (0.0, 0.0, 200.0)}
for node_id, pos in positions.items():
    G.add_node(node_id, pos=np.asarray(pos, dtype=float))
for u, v in ((0, 1), (1, 2)):
    start, end = G.nodes[u]["pos"], G.nodes[v]["pos"]
    G.add_edge(
        u, v,
        length=float(np.linalg.norm(end - start)),
        branch_order="B01",
        voxels=[start.tolist(), end.tolist()],
    )

G, _ = haemodynamics.apply_poiseuille_haemodynamics(
    G, diameter_by_branch_order={"B01": 6.0}
)
conductance, node_list = haemodynamics.build_conductance_matrix_from_graph(G)
flow = haemodynamics.solve_flow_from_conductance_matrix(
    conductance, node_list,
    inlet_p_bc=1000.0, outlet_p_bc=500.0,
    inlet_nodes=[0], outlet_nodes=[2],
)
haemodynamics.set_edge_flows(G, node_list, flow["pressure"])

written = visualization.graph_to_vtk(G, str(here / "out"))
stats = statistics.compute_comprehensive_vessel_statistics(G)

result = {
    "version": haemolynx.__version__,
    "package_dir": str(Path(haemolynx.__file__).parent),
    "edges": G.number_of_edges(),
    "resistances": [float(d["resistance"]) for _u, _v, d in G.edges(data=True)],
    "vtk_files": sorted(
        Path(v).name for v in written.values()
        if isinstance(v, str) and v.endswith(".vtp")
    ),
    "total_length": float(stats["Total Edge Length (microns)"]),
}
print("RESULT " + json.dumps(result))
