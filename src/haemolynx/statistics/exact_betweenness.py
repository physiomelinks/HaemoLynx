"""Exact betweenness centrality in worker processes, digit for digit
``nx.betweenness_centrality``'s.

networkx sums each node's betweenness over the source nodes in the graph's
own node order: one breadth-first (or Dijkstra) search from a source, then
one addition onto every node that search reached. Each source's additions
depend on nothing but the graph, so the searches can run anywhere; only the
sum has to be taken in networkx's order, since floating-point addition is
not associative. Here the workers run networkx's own per-source functions
and hand back each source's increments, and this process adds them source
by source in node order, then rescales with networkx's own ``_rescale``.
The answer is the same bits, not merely close.

On a 34,000-node batch graph one exact pass took 20-40 minutes in one
process. Nothing is sampled (``k``) and no other library is used: those
change digits.

The workers are ``python -c`` processes that import networkx and nothing
else -- never this package, and never the caller's script (which
multiprocessing's spawn would re-run without an ``if __name__ ==
"__main__"`` guard) -- with one OpenMP/BLAS thread each, as graph
building's routing workers are started (``graph.reconnect``).
"""
from __future__ import annotations

import logging
import pickle
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Hashable, Optional

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "PARALLEL_BETWEENNESS_MIN_NODES",
    "betweenness_worker_count",
    "exact_betweenness_centrality",
]

#: Below this many nodes ``nx.betweenness_centrality`` runs in this process:
#: an exact pass there takes seconds, about what starting the workers costs.
PARALLEL_BETWEENNESS_MIN_NODES = 2000

#: What one worker holds besides its graph: the interpreter and networkx.
WORKER_BASE_BYTES = 80 << 20
#: A worker's share per node and per directed edge of the graph it is sent,
#: networkx's dicts and one search's working dicts (measured on batch graphs
#: at ~1.5 kB a node, ~0.5 kB an edge; rounded up).
WORKER_BYTES_PER_NODE = 2048
WORKER_BYTES_PER_EDGE = 1024

#: About how many bytes of increments one task sends back, which sets how
#: many sources a task carries; at most this many tasks wait unsummed a
#: worker, so what waits is bounded.
TASK_RESULT_BYTES = 4 << 20
TASKS_AHEAD_PER_WORKER = 4

_WORKER_CODE = """
import pickle, sys
from collections import defaultdict
import numpy as np
from networkx.algorithms.centrality.betweenness import (
    _accumulate_basic, _accumulate_endpoints,
    _single_source_dijkstra_path_basic, _single_source_shortest_path_basic,
)
read, write = sys.stdin.buffer, sys.stdout.buffer
H, weighted, endpoints = pickle.load(read)
n = H.number_of_nodes()
accumulate = _accumulate_endpoints if endpoints else _accumulate_basic
while True:
    try:
        sources = pickle.load(read)
    except EOFError:
        break
    try:
        rows = []
        for s in sources:
            if weighted:
                S, P, sigma = _single_source_dijkstra_path_basic(H, s, "w")[:3]
            else:
                S, P, sigma = _single_source_shortest_path_basic(H, s)[:3]
            # Each node is added onto once a source, so its increment is what
            # networkx's own accumulation adds onto a zero.
            increments = accumulate(defaultdict(float), S, P, sigma, s)
            if isinstance(increments, tuple):
                increments = increments[0]
            m = len(increments)
            nodes = np.fromiter(increments.keys(), dtype=np.int64, count=m)
            values = np.fromiter(increments.values(), dtype=np.float64, count=m)
            if 3 * m > 2 * n:
                row = np.zeros(n, dtype=np.float64)
                row[nodes] = values
                rows.append((None, row))
            else:
                rows.append((nodes.astype(np.int32), values))
        reply = (True, rows)
    except BaseException as exc:
        reply = (False, exc)
    pickle.dump(reply, write, protocol=pickle.HIGHEST_PROTOCOL)
    write.flush()
"""


class _WorkerFailed(RuntimeError):
    """A worker stopped or raised; the pass is redone in this process."""


def betweenness_worker_count(n_nodes: int, n_edges: int) -> int:
    """How many workers an exact pass over a graph this size gets:
    :func:`~haemolynx.preprocessing.shared_volume_tasks.default_worker_count`,
    or as many as half the memory free now holds, if fewer."""
    from haemolynx.haemodynamics.resistance import available_memory_bytes
    from haemolynx.preprocessing.shared_volume_tasks import default_worker_count

    workers = default_worker_count()
    available = available_memory_bytes()
    if available is not None:
        per_worker = (
            WORKER_BASE_BYTES
            + WORKER_BYTES_PER_NODE * int(n_nodes)
            + WORKER_BYTES_PER_EDGE * 2 * int(n_edges)
        )
        workers = min(workers, int(0.5 * available // per_worker))
    return max(1, workers)


def _indexed_graph(G: nx.Graph, weight) -> nx.DiGraph:
    """*G* as a digraph on 0..n-1 in *G*'s node order, every node's
    out-neighbours in the order ``G[v]`` gives them (what a search's
    tie-breaking and its sums follow), and, when *weight* is given, each
    edge's ``"w"`` the very value networkx's weight function returns for it
    (for a multigraph, the least of the parallel edges')."""
    index = {v: i for i, v in enumerate(G)}
    H = nx.DiGraph()
    H.add_nodes_from(range(len(index)))
    if weight is None:
        for v, i in index.items():
            H.add_edges_from((i, index[w]) for w in G[v])
    else:
        from networkx.algorithms.shortest_paths.weighted import _weight_function

        weight_of = _weight_function(G, weight)
        for v, i in index.items():
            H.add_edges_from(
                (i, index[w], {"w": weight_of(v, w, data)}) for w, data in G[v].items()
            )
    return H


def _start_workers(count: int, payload: bytes) -> list:
    from haemolynx.graph.reconnect import _routing_environment

    environment = _routing_environment()
    workers = []
    try:
        for _ in range(count):
            worker = subprocess.Popen(
                [sys.executable, "-c", _WORKER_CODE],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                env=environment,
            )
            workers.append(worker)
            worker.stdin.write(payload)
            worker.stdin.flush()
    except OSError:
        _stop_workers(workers)
        raise
    return workers


def _stop_workers(workers: list) -> None:
    for worker in workers:
        try:
            worker.stdin.close()
        except OSError:
            pass
    for worker in workers:
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait()
        try:
            worker.stdout.close()
        except OSError:
            pass


def _parallel_sums(H: nx.DiGraph, weighted: bool, endpoints: bool, workers: int) -> np.ndarray:
    """Every node's unscaled betweenness, the sources' increments added in
    node order (the order ``nx.betweenness_centrality`` adds them in)."""
    n = H.number_of_nodes()
    per_task = int(max(1, min(64, TASK_RESULT_BYTES // (8 * n), -(-n // (8 * workers)))))
    tasks = [list(range(start, min(n, start + per_task))) for start in range(0, n, per_task)]
    ahead = TASKS_AHEAD_PER_WORKER * workers

    payload = pickle.dumps((H, weighted, endpoints), protocol=pickle.HIGHEST_PROTOCOL)
    processes = _start_workers(workers, payload)
    done: Dict[int, Any] = {}
    state = {"next_task": 0, "summed": 0, "failed": None}
    condition = threading.Condition()

    def serve(worker) -> None:
        while True:
            with condition:
                while (
                    state["failed"] is None
                    and state["next_task"] < len(tasks)
                    and state["next_task"] >= state["summed"] + ahead
                ):
                    condition.wait()
                if state["failed"] is not None or state["next_task"] >= len(tasks):
                    return
                task = state["next_task"]
                state["next_task"] += 1
            try:
                pickle.dump(tasks[task], worker.stdin, protocol=pickle.HIGHEST_PROTOCOL)
                worker.stdin.flush()
                ok, value = pickle.load(worker.stdout)
                if not ok:
                    raise _WorkerFailed(f"a betweenness worker raised {value!r}")
            except BaseException as exc:  # a dead pipe, EOF, or the worker's own error
                with condition:
                    if state["failed"] is None:
                        state["failed"] = exc
                    condition.notify_all()
                return
            with condition:
                done[task] = value
                condition.notify_all()

    threads = [threading.Thread(target=serve, args=(w,), daemon=True) for w in processes]
    for thread in threads:
        thread.start()
    sums = np.zeros(n, dtype=np.float64)
    try:
        for task in range(len(tasks)):
            with condition:
                while task not in done and state["failed"] is None:
                    condition.wait()
                if state["failed"] is not None:
                    raise _WorkerFailed("a betweenness worker failed") from state["failed"]
                rows = done.pop(task)
            for nodes, values in rows:
                # One source's increments onto every node at once: each node
                # is added onto once a source, so this is networkx's sum.
                if nodes is None:
                    sums += values
                else:
                    sums[nodes] += values
            with condition:
                state["summed"] = task + 1
                condition.notify_all()
    finally:
        with condition:
            if state["failed"] is None and state["summed"] < len(tasks):
                state["failed"] = _WorkerFailed("stopped")
            condition.notify_all()
        _stop_workers(processes)
        for thread in threads:
            thread.join()
    return sums


def exact_betweenness_centrality(
    G: nx.Graph,
    *,
    normalized: bool = True,
    weight: Optional[str] = None,
    endpoints: bool = False,
    workers: Optional[int] = None,
    min_nodes: int = PARALLEL_BETWEENNESS_MIN_NODES,
) -> Dict[Hashable, float]:
    """``nx.betweenness_centrality(G, normalized=..., weight=...,
    endpoints=...)`` (exact: no ``k``), the same dict to the last bit, with
    the per-source searches spread over *workers* processes
    (:func:`betweenness_worker_count` when ``None``).

    Below *min_nodes* nodes, with one worker, or when the workers cannot
    start or one fails, networkx runs it here instead.
    """
    n = G.number_of_nodes()
    if workers is None:
        workers = betweenness_worker_count(n, G.number_of_edges()) if n >= min_nodes else 1
    workers = int(min(workers, n))
    if n < min_nodes or workers <= 1:
        return nx.betweenness_centrality(
            G, normalized=normalized, weight=weight, endpoints=endpoints
        )

    from networkx.algorithms.centrality.betweenness import _rescale

    started = time.perf_counter()
    logger.info(
        "[betweenness] exact %s pass over %d nodes in %d worker processes",
        "weighted" if weight is not None else "unweighted", n, workers,
    )
    try:
        sums = _parallel_sums(_indexed_graph(G, weight), weight is not None, endpoints, workers)
    except (OSError, _WorkerFailed):
        logger.warning(
            "[betweenness] the worker processes failed; computing it in this process",
            exc_info=True,
        )
        return nx.betweenness_centrality(
            G, normalized=normalized, weight=weight, endpoints=endpoints
        )
    betweenness = dict(zip(G, sums.tolist()))
    betweenness = _rescale(
        betweenness, n, normalized=normalized, directed=G.is_directed(), endpoints=endpoints
    )
    logger.info("[betweenness] done in %.1f s", time.perf_counter() - started)
    return betweenness
