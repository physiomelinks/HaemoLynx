"""Utilities for removing terminal-to-terminal edges."""
from typing import Union

import networkx as nx


def remove_terminal_terminal_edges(
    G: Union[nx.Graph, nx.MultiGraph],
    debug: bool = False,
    return_removed_count: bool = False,
) -> Union[nx.Graph, nx.MultiGraph] | tuple[Union[nx.Graph, nx.MultiGraph], int]:
    """Remove edges whose endpoints are both degree-1 terminal nodes.

    This targets tiny disconnected edge fragments (u--v) where both endpoints
    are terminal nodes at the time of evaluation.
    """
    G_clean = G.copy()
    if G_clean.number_of_edges() == 0:
        if return_removed_count:
            return G_clean, 0
        return G_clean

    degree_snapshot = dict(G_clean.degree())

    if isinstance(G_clean, nx.MultiGraph):
        edges_to_remove = [
            (u, v, key)
            for u, v, key in G_clean.edges(keys=True)
            if degree_snapshot.get(u, 0) == 1 and degree_snapshot.get(v, 0) == 1
        ]
        if edges_to_remove:
            G_clean.remove_edges_from(edges_to_remove)
    else:
        edges_to_remove = [
            (u, v)
            for u, v in G_clean.edges()
            if degree_snapshot.get(u, 0) == 1 and degree_snapshot.get(v, 0) == 1
        ]
        if edges_to_remove:
            G_clean.remove_edges_from(edges_to_remove)

    removed_count = len(edges_to_remove)

    if debug:
        print(
            "Removed "
            f"{removed_count} edge(s) where both endpoints were degree-1 terminal nodes."
        )

    if return_removed_count:
        return G_clean, removed_count
    return G_clean
