"""Pure state and rules for the "Edit" window: Add branch, Delete edge.

No Qt, no napari import here -- :mod:`haemolynx.gui._widget` wires the
floating window's buttons and mouse clicks to :class:`GraphEditorState`,
and this is what runs when a click comes in, so the editing rules
themselves are testable without a live viewer (this sandbox's
``make_napari_viewer`` is known to crash -- see project memory).

Reuses :mod:`haemolynx.graph.edit`'s primitives for every actual graph
mutation and :mod:`haemolynx.gui.graph_click`'s hit results for what a click
resolved to; this module only sequences them into "Add branch" (first click
restricted to existing structure, every later click extends the draft by an
A* segment toward wherever was clicked, landing on existing structure
finishes it) and "Delete edge" (click routes straight to
:func:`haemolynx.graph.edit.delete_edge_and_collapse`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import networkx as nx
import numpy as np

from haemolynx.graph import (
    EdgeDraft,
    astar_path,
    commit_new_edge,
    delete_edge_and_collapse,
    insert_node_on_edge,
    mask_cost_field,
    voxel_path_to_microns,
)

from .graph_click import EdgeHit, NodeHit

__all__ = ["GraphEditorState"]

#: What one call to click_add did, for the caller to decide what to redraw
#: and report: a first click that missed existing structure does nothing.
AddClickResult = Literal["rejected", "started", "extended", "finished"]


def _hit_point(graph: nx.MultiGraph, hit: NodeHit | EdgeHit) -> tuple[float, float, float]:
    if isinstance(hit, EdgeHit):
        return hit.point_um
    return tuple(float(c) for c in graph.nodes[hit.node_id]["pos"])


@dataclass
class GraphEditorState:
    """Everything the "Edit" window needs between clicks.

    ``graph`` is the working copy every edit mutates directly -- the caller
    gives it a fresh deep copy on open (so cancelling the window discards
    edits) and pushes an updated vessels/nodes layer after every call here
    that returns a non-empty ``changed`` set (see
    :meth:`haemolynx.gui.results.ResultLayers.layers_for_graph`).

    ``cost_field`` is built once, from the segmented mask (see
    :func:`haemolynx.graph.edit.mask_cost_field`), by the caller before
    arming "add" mode -- not by this class, which has no I/O of its own.
    """

    graph: nx.MultiGraph
    voxel_size_zyx: tuple[float, float, float]
    cost_field: np.ndarray | None = None
    mode: Literal["idle", "add", "delete"] = "idle"
    draft: EdgeDraft | None = None
    reserved_ids: set[Any] = field(default_factory=set)

    def start_add(self) -> None:
        """Arm "Add branch": the next click must land on a node or edge."""
        self.mode = "add"
        self.draft = None

    def start_delete(self) -> None:
        """Arm "Delete edge": the next click removes whatever edge it hits."""
        self.mode = "delete"
        self.draft = None

    def stop(self) -> None:
        """Disarm both modes and drop any in-progress draft."""
        self.mode = "idle"
        self.draft = None

    def _anchor_node(self, hit: NodeHit | EdgeHit) -> Any:
        if isinstance(hit, NodeHit):
            return hit.node_id
        return insert_node_on_edge(
            self.graph, hit.u, hit.v, hit.key, hit.point_um, reserved_ids=self.reserved_ids
        )

    def _extend_draft_to(self, point_um: tuple[float, float, float]) -> None:
        if self.draft is None:
            raise RuntimeError("no draft to extend")
        if self.cost_field is None:
            # No segmented mask was available to route through (e.g. the
            # image layer was missing when the editor opened) -- draw a
            # straight line rather than refuse the click outright, matching
            # astar_path's own "always draw something" fallback below.
            self.draft.extend([self.draft.last_point, point_um])
            return
        scale = self.voxel_size_zyx
        start_vox = tuple(c / s for c, s in zip(self.draft.last_point, scale))
        end_vox = tuple(c / s for c, s in zip(point_um, scale))
        path_vox = astar_path(self.cost_field, start_vox, end_vox)
        self.draft.extend(voxel_path_to_microns(path_vox, scale))

    def click_add(
        self,
        raw_point_um: tuple[float, float, float],
        hit: NodeHit | EdgeHit | None,
    ) -> AddClickResult:
        """One click while :attr:`mode` is ``"add"``.

        The first click must resolve to *hit* (an existing node or edge) --
        a miss is rejected outright, nothing started. Every later click
        extends the draft by an A* segment toward *raw_point_um* regardless
        of whether it also hit something; when it did, the draft finishes
        there instead of just extending.
        """
        if self.draft is None:
            if hit is None:
                return "rejected"
            node = self._anchor_node(hit)
            self.draft = EdgeDraft(
                start_node=node, points_um=[tuple(float(c) for c in self.graph.nodes[node]["pos"])]
            )
            return "started"

        if hit is None:
            self._extend_draft_to(raw_point_um)
            return "extended"

        self._extend_draft_to(_hit_point(self.graph, hit))
        end_node = self._anchor_node(hit)
        if end_node == self.draft.start_node:
            # Clicking back on the draft's own start is not a real target --
            # keep drawing rather than commit a zero-length self-loop.
            return "extended"
        commit_new_edge(self.graph, self.draft, end_node, reserved_ids=self.reserved_ids)
        self.draft = None
        return "finished"

    def finish_add(self) -> Any | None:
        """The Finish button: commit the draft as a dangling terminal.

        Returns the new terminal node id, or ``None`` if there was no draft.
        """
        if self.draft is None:
            return None
        _start, end_node, _key = commit_new_edge(
            self.graph, self.draft, end_node=None, reserved_ids=self.reserved_ids
        )
        self.draft = None
        return end_node

    def click_delete(self, hit: EdgeHit | None) -> set[Any]:
        """One click while :attr:`mode` is ``"delete"``.

        Returns the node ids the caller's drawn layers need to refresh
        (empty when the click missed every edge).
        """
        if hit is None:
            return set()
        return delete_edge_and_collapse(self.graph, hit.u, hit.v, hit.key)

    def cost_field_from_mask(self, mask: np.ndarray, *, use_memmap: bool = False) -> None:
        """Convenience: build and store :attr:`cost_field` from a segmented mask.

        *use_memmap* (the low-RAM option) stores a
        :class:`haemolynx.graph.edit.WindowedMaskCostField`, which gives the
        same values one routing window at a time.
        """
        self.cost_field = mask_cost_field(mask, use_memmap=use_memmap)
