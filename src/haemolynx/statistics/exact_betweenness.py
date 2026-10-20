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

The unweighted pass the statistics make (no endpoints, an undirected graph)
is not networkx's loop but the same algorithm for a batch of sources at once
(:data:`_BATCHED_PASS`): level by level with numpy, each source on its own
copy of its connected component, every addition networkx makes made in the
same order -- a node's path count and dependency take their terms in
networkx's sequence (``np.add.at`` adds in index order, the sorts are
stable) -- so each source's increments are networkx's bits, summed here in
node order as before. It runs in the workers (numpy only) and, below
:data:`PARALLEL_BETWEENNESS_MIN_NODES` or with one worker, here. On 29_2_f
contra's statistics graph (15,843 nodes, 2026-10-20, one machine state):
networkx's loop in 8 workers 164 s, the batched pass here 125 s, in 8
workers 19 s; 29_1_f contra (21,944): 303, 213 and 26 s; networkx alone
~1,040 s on the first.
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


#: The batched pass (see the module's docstring), as source: the workers run
#: it, importing numpy and nothing of this package, and so does this process
#: (:func:`_brandes_dependencies`), the very same code.
_BATCHED_PASS = r'''
import numpy as np


def brandes_dependencies(sources, indptr, indices, n):
    """networkx's _single_source_shortest_path_basic and _accumulate_basic for
    each of *sources* at once, on the graph whose CSR adjacency (in networkx's
    neighbour order) is *indptr*/*indices*: a (len(sources), n) array, row k
    the increment each node takes from sources[k] -- 0 at the source itself
    and wherever it does not reach. Source k searches nodes k*n .. k*n+n-1. A
    level is found in networkx's queue order (each frontier node's neighbours
    in turn, a node where it is first met), its path counts summed over its
    predecessors in that order, and the dependencies passed back from the
    deepest level up, each level's nodes in reverse queue order."""
    k = len(sources)
    dist = np.full(k * n, -1, dtype=np.int64)
    sigma = np.zeros(k * n)
    delta = np.zeros(k * n)
    position = np.empty(k * n, dtype=np.int64)
    roots = np.arange(k, dtype=np.int64) * n + sources
    dist[roots] = 0
    sigma[roots] = 1.0
    frontier, levels, links = roots, [roots], []
    depth = 0
    while True:
        offset = (frontier // n) * n
        starts = indptr[frontier - offset]
        counts = indptr[frontier - offset + 1] - starts
        reached = int(counts.sum())
        if reached == 0:
            break
        ends = np.cumsum(counts)
        neighbour = indices[np.arange(reached) + np.repeat(starts - (ends - counts), counts)]
        neighbour += np.repeat(offset, counts)
        found = neighbour[dist[neighbour] == -1]
        if not len(found):
            break
        _values, first = np.unique(found, return_index=True)
        following = found[np.sort(first)]
        dist[following] = depth + 1
        on_path = dist[neighbour] == depth + 1
        before, after = np.repeat(frontier, counts)[on_path], neighbour[on_path]
        np.add.at(sigma, after, sigma[before])
        links.append((before, after))
        levels.append(following)
        frontier = following
        depth += 1
    for level in range(len(levels) - 1, 0, -1):
        position[levels[level]] = np.arange(len(levels[level]))
        before, after = links[level - 1]
        order = np.argsort(-position[after], kind="stable")
        before, after = before[order], after[order]
        coefficient = (1.0 + delta[after]) / sigma[after]
        np.add.at(delta, before, sigma[before] * coefficient)
    delta[roots] = 0.0
    return delta.reshape(k, n)
'''

_BATCHED_NAMESPACE: Dict[str, Any] = {}
exec(_BATCHED_PASS, _BATCHED_NAMESPACE)
_brandes_dependencies = _BATCHED_NAMESPACE["brandes_dependencies"]

_BATCHED_WORKER_CODE = _BATCHED_PASS + """
import pickle, sys
read, write = sys.stdin.buffer, sys.stdout.buffer
csrs = pickle.load(read)
while True:
    try:
        task = pickle.load(read)
    except EOFError:
        break
    try:
        blocks = []
        for c, start, stop in task:
            indptr, indices = csrs[c]
            blocks.append(brandes_dependencies(
                np.arange(start, stop, dtype=np.int64), indptr, indices, len(indptr) - 1
            ))
        reply = (True, blocks)
    except BaseException as exc:
        reply = (False, exc)
    pickle.dump(reply, write, protocol=pickle.HIGHEST_PROTOCOL)
    write.flush()
"""

#: Node slots (sources x component nodes) one batch of the batched pass
#: searches at once: about 32 MB of working arrays.
BATCH_SLOTS = 1 << 20
#: Most sources a batch holds; past this one gets no faster.
MAX_BATCH_SOURCES = 128


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


def _start_workers(count: int, payload: bytes, code: Optional[str] = None) -> list:
    from haemolynx.graph.reconnect import _routing_environment

    environment = _routing_environment()
    workers = []
    try:
        for _ in range(count):
            worker = subprocess.Popen(
                [sys.executable, "-c", _WORKER_CODE if code is None else code],
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
    sums = np.zeros(n, dtype=np.float64)

    def add(_task: int, rows) -> None:
        for nodes, values in rows:
            # One source's increments onto every node at once: each node
            # is added onto once a source, so this is networkx's sum.
            if nodes is None:
                np.add(sums, values, out=sums)
            else:
                sums[nodes] += values

    payload = pickle.dumps((H, weighted, endpoints), protocol=pickle.HIGHEST_PROTOCOL)
    _in_task_order(_WORKER_CODE, payload, tasks, workers, add)
    return sums


def _in_task_order(code: str, payload: bytes, tasks: list, workers: int, add) -> None:
    """Hand *tasks* out to *workers* ``python -c`` *code* processes started
    with *payload*, and call ``add(task, value)`` with each task's reply in
    task order -- the order the sums need -- at most
    :data:`TASKS_AHEAD_PER_WORKER` tasks a worker ahead of it."""
    ahead = TASKS_AHEAD_PER_WORKER * workers
    processes = _start_workers(workers, payload, code)
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
    try:
        for task in range(len(tasks)):
            with condition:
                while task not in done and state["failed"] is None:
                    condition.wait()
                if state["failed"] is not None:
                    raise _WorkerFailed("a betweenness worker failed") from state["failed"]
                value = done.pop(task)
            add(task, value)
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


def _components_in_node_order(adjacency: list) -> list:
    """The connected components of the graph *adjacency* (lists of node
    indices), each as its node indices in ascending order."""
    label = [-1] * len(adjacency)
    components = []
    for seed in range(len(adjacency)):
        if label[seed] != -1:
            continue
        label[seed] = seed
        stack, members = [seed], []
        while stack:
            v = stack.pop()
            members.append(v)
            for w in adjacency[v]:
                if label[w] == -1:
                    label[w] = seed
                    stack.append(w)
        components.append(sorted(members))
    return components


def _batched_tasks(sizes: list, workers: int) -> list:
    """The batched pass's tasks: ``[(component, first source, past the last),
    ...]`` lists, every component's sources in order, a task holding about
    :data:`BATCH_SLOTS` node slots, and a large component split across at
    least four batches a worker."""
    tasks, task, slots = [], [], 0
    for component, size in enumerate(sizes):
        batch = max(1, min(MAX_BATCH_SOURCES, BATCH_SLOTS // size))
        if workers > 1:
            batch = max(1, min(batch, -(-size // (4 * workers))))
        for start in range(0, size, batch):
            stop = min(size, start + batch)
            if task and slots + (stop - start) * size > BATCH_SLOTS:
                tasks.append(task)
                task, slots = [], 0
            task.append((component, start, stop))
            slots += (stop - start) * size
    if task:
        tasks.append(task)
    return tasks


def _batched_sums(G: nx.Graph, workers: int) -> np.ndarray:
    """Every node's unscaled betweenness by the batched pass (undirected,
    unweighted, no endpoints), in *workers* processes when more than one:
    each component's sources' rows added in node order. A component of two
    nodes or fewer adds nothing."""
    nodes = list(G)
    at = {node: i for i, node in enumerate(nodes)}
    adjacency = [[at[w] for w in G[node]] for node in nodes]
    components = [members for members in _components_in_node_order(adjacency) if len(members) > 2]
    csrs = []
    for members in components:
        local = {node: i for i, node in enumerate(members)}
        neighbours = [[local[w] for w in adjacency[node]] for node in members]
        indptr = np.zeros(len(members) + 1, dtype=np.int64)
        indptr[1:] = np.cumsum([len(row) for row in neighbours])
        indices = np.fromiter((w for row in neighbours for w in row), dtype=np.int64, count=int(indptr[-1]))
        csrs.append((indptr, indices))
    tasks = _batched_tasks([len(members) for members in components], workers)
    totals = [np.zeros(len(members)) for members in components]

    def add(task: int, blocks) -> None:
        for (component, _start, _stop), block in zip(tasks[task], blocks):
            for row in block:
                totals[component] += row

    if workers > 1 and tasks:
        payload = pickle.dumps(csrs, protocol=pickle.HIGHEST_PROTOCOL)
        _in_task_order(_BATCHED_WORKER_CODE, payload, tasks, workers, add)
    else:
        for task, jobs in enumerate(tasks):
            add(task, [
                _brandes_dependencies(np.arange(start, stop, dtype=np.int64), *csrs[c], len(csrs[c][0]) - 1)
                for c, start, stop in jobs
            ])
    sums = np.zeros(len(nodes), dtype=np.float64)
    for members, total in zip(components, totals):
        sums[np.asarray(members, dtype=np.intp)] = total
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
    start or one fails, it runs here instead: the batched pass for an
    unweighted undirected graph without endpoints, else networkx.
    """
    from networkx.algorithms.centrality.betweenness import _rescale

    n = G.number_of_nodes()
    if workers is None:
        workers = betweenness_worker_count(n, G.number_of_edges()) if n >= min_nodes else 1
    workers = int(min(workers, n))
    if weight is None and not endpoints and not G.is_directed():
        started = time.perf_counter()
        sums = None
        if n >= min_nodes and workers > 1:
            logger.info("[betweenness] exact batched pass over %d nodes in %d worker processes", n, workers)
            try:
                sums = _batched_sums(G, workers)
            except (OSError, _WorkerFailed):
                logger.warning(
                    "[betweenness] the worker processes failed; computing it in this process",
                    exc_info=True,
                )
        if sums is None:
            sums = _batched_sums(G, 1)
        if n >= min_nodes:
            logger.info("[betweenness] done in %.1f s", time.perf_counter() - started)
        return _rescale(
            dict(zip(G, sums.tolist())), n, normalized=normalized, directed=False, endpoints=False
        )
    if n < min_nodes or workers <= 1:
        return nx.betweenness_centrality(
            G, normalized=normalized, weight=weight, endpoints=endpoints
        )

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
