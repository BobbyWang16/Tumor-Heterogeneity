from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
from skimage.morphology import skeletonize


Coord = Tuple[int, int, int]
Edge = Tuple[Coord, Coord]


@dataclass(slots=True)
class SkeletonResult:
    skeleton_mask: np.ndarray
    branches: list[np.ndarray]


_NEIGHBOR_OFFSETS = [
    offset
    for offset in product((-1, 0, 1), repeat=3)
    if offset != (0, 0, 0)
]


def skeletonize_volume(mask: np.ndarray) -> np.ndarray:
    """Return a one-voxel-thick 3D skeleton."""

    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape {mask.shape}")
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    return skeletonize(mask.astype(bool)).astype(bool)


def skeletonize_and_extract_branches(mask: np.ndarray) -> SkeletonResult:
    skel = skeletonize_volume(mask)
    branches = extract_skeleton_branches(skel)
    return SkeletonResult(skeleton_mask=skel, branches=branches)


def extract_skeleton_branches(skeleton_mask: np.ndarray) -> list[np.ndarray]:
    """Split a skeleton mask into graph branches.

    Endpoints and junctions terminate branches. Remaining degree-2 cycles are
    emitted as closed paths.
    """

    coords = [tuple(int(v) for v in c) for c in np.argwhere(skeleton_mask)]
    node_set: Set[Coord] = set(coords)
    if not coords:
        return []

    neighbors: Dict[Coord, list[Coord]] = {c: [] for c in coords}
    shape = np.asarray(skeleton_mask.shape)
    for c in coords:
        base = np.asarray(c)
        for off in _NEIGHBOR_OFFSETS:
            nb = tuple(int(v) for v in base + off)
            if all(0 <= nb[i] < shape[i] for i in range(3)) and nb in node_set:
                neighbors[c].append(nb)

    critical = {c for c, nbs in neighbors.items() if len(nbs) != 2}
    visited_edges: Set[Edge] = set()
    branches: list[np.ndarray] = []

    for start in sorted(critical):
        for nb in sorted(neighbors[start]):
            edge = _edge(start, nb)
            if edge in visited_edges:
                continue
            path = _walk_path(start, nb, neighbors, critical, visited_edges)
            if len(path) >= 2:
                branches.append(np.asarray(path, dtype=float))

    for start in sorted(node_set):
        for nb in sorted(neighbors[start]):
            edge = _edge(start, nb)
            if edge in visited_edges:
                continue
            path = _walk_cycle(start, nb, neighbors, visited_edges)
            if len(path) >= 2:
                branches.append(np.asarray(path, dtype=float))

    return branches


def _walk_path(
    start: Coord,
    next_node: Coord,
    neighbors: Dict[Coord, list[Coord]],
    critical: Set[Coord],
    visited_edges: Set[Edge],
) -> list[Coord]:
    path = [start]
    prev = start
    cur = next_node
    visited_edges.add(_edge(prev, cur))

    while True:
        path.append(cur)
        if cur in critical and cur != start:
            break
        candidates = [n for n in neighbors[cur] if n != prev]
        if not candidates:
            break
        nxt = sorted(candidates)[0]
        edge = _edge(cur, nxt)
        if edge in visited_edges:
            break
        visited_edges.add(edge)
        prev, cur = cur, nxt
    return path


def _walk_cycle(
    start: Coord,
    next_node: Coord,
    neighbors: Dict[Coord, list[Coord]],
    visited_edges: Set[Edge],
) -> list[Coord]:
    path = [start]
    prev = start
    cur = next_node
    visited_edges.add(_edge(prev, cur))

    while True:
        path.append(cur)
        candidates = [n for n in neighbors[cur] if n != prev]
        if not candidates:
            break
        nxt = sorted(candidates)[0]
        edge = _edge(cur, nxt)
        if edge in visited_edges:
            break
        visited_edges.add(edge)
        prev, cur = cur, nxt
        if cur == start:
            break
    return path


def _edge(a: Coord, b: Coord) -> Edge:
    return (a, b) if a <= b else (b, a)

