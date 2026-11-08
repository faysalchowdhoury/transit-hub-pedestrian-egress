"""Platform geometry on an occupancy grid, Dijkstra distance field and A* paths.

The walkable area is a boolean grid (``True`` = free) with cell size ``h``. Exits are sets of
free cells at the stair landings. Navigation uses an 8-connected **Dijkstra distance field**
(geodesic distance to the nearest exit around walls and obstacles), so every pedestrian's desired
direction is the negative gradient of one precomputed field: O(cells log cells) once, O(1) per agent
per step. **A*** with an octile heuristic answers single-route queries (API / wayfinding).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

SQ2 = math.sqrt(2.0)
_NB = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, SQ2), (-1, 1, SQ2), (1, -1, SQ2), (1, 1, SQ2)]


@dataclass(frozen=True)
class Exit:
    name: str
    x0: float
    x1: float
    y: float  # exit line (stair landing edge) at the platform boundary
    width: float


@dataclass
class Platform:
    """Island platform ``length x width`` with stair openings; obstacles = columns / stair enclosures."""

    length: float = 120.0
    width: float = 10.0
    h: float = 0.25
    exits: list[Exit] = field(default_factory=list)
    obstacles: list[tuple[float, float, float, float]] = field(default_factory=list)  # x0, y0, x1, y1

    def __post_init__(self) -> None:
        nx, ny = int(round(self.length / self.h)), int(round(self.width / self.h))
        self.free = np.ones((nx, ny), bool)
        for x0, y0, x1, y1 in self.obstacles:
            i0, i1 = int(x0 / self.h), int(math.ceil(x1 / self.h))
            j0, j1 = int(y0 / self.h), int(math.ceil(y1 / self.h))
            self.free[max(i0, 0):min(i1, nx), max(j0, 0):min(j1, ny)] = False
        self.dist = self._distance_field()

    def cell(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        i = np.clip((np.asarray(x) / self.h).astype(int), 0, self.free.shape[0] - 1)
        j = np.clip((np.asarray(y) / self.h).astype(int), 0, self.free.shape[1] - 1)
        return i, j

    def exit_cells(self, edge_margin: float = 0.3) -> list[tuple[int, int]]:
        """Target cells of each opening, kept ``edge_margin`` clear of the opening's jambs.

        Without the margin, the field points straight out at the jamb itself, where the wall's repulsion
        balances it, and agents freeze there (observed in testing).
        """
        out = []
        ny = self.free.shape[1]
        for e in self.exits:
            j = 0 if e.y <= 0.0 else ny - 1
            for i in range(int((e.x0 + edge_margin) / self.h), int(math.ceil((e.x1 - edge_margin) / self.h))):
                if self.free[i, j]:
                    out.append((i, j))
        return out

    def _distance_field(self) -> np.ndarray:
        d = np.full(self.free.shape, np.inf)
        pq: list[tuple[float, int, int]] = []
        for i, j in self.exit_cells():
            d[i, j] = 0.0
            pq.append((0.0, i, j))
        heapq.heapify(pq)
        nx, ny = self.free.shape
        while pq:
            dist, i, j = heapq.heappop(pq)
            if dist > d[i, j]:
                continue
            for di, dj, w in _NB:
                a, b = i + di, j + dj
                if 0 <= a < nx and 0 <= b < ny and self.free[a, b]:
                    if di and dj and not (self.free[i + di, j] and self.free[i, j + dj]):
                        continue  # no corner cutting
                    nd = dist + w * self.h
                    if nd < d[a, b]:
                        d[a, b] = nd
                        heapq.heappush(pq, (nd, a, b))
        return d

    def direction(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Unit desired direction = steepest descent of the distance field (8-neighbour argmin)."""
        i, j = self.cell(x, y)
        nx, ny = self.free.shape
        best = np.full(i.shape, np.inf)
        di_best = np.zeros(i.shape)
        dj_best = np.zeros(i.shape)
        for di, dj, w in _NB:
            a, b = np.clip(i + di, 0, nx - 1), np.clip(j + dj, 0, ny - 1)
            val = self.dist[a, b] + 0.0 * w
            better = val < best
            best = np.where(better, val, best)
            di_best = np.where(better, di, di_best)
            dj_best = np.where(better, dj, dj_best)
        # on a target cell nothing is "downhill": point outward through the opening instead
        at_exit = self.dist[i, j] <= 1e-9
        di_best = np.where(at_exit, 0.0, di_best)
        dj_best = np.where(at_exit, np.where(j == 0, -1.0, 1.0), dj_best)
        v = np.stack([di_best, dj_best], axis=1).astype(float)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return np.where(n > 0, v / np.maximum(n, 1e-9), 0.0)

    def astar(self, start: tuple[float, float], goal: tuple[float, float]) -> list[tuple[float, float]]:
        """Shortest 8-connected path (cell centres) between two points; empty list if unreachable."""
        s = tuple(int(v) for v in (start[0] / self.h, start[1] / self.h))
        g = tuple(int(v) for v in (goal[0] / self.h, goal[1] / self.h))
        nx, ny = self.free.shape
        if not (0 <= s[0] < nx and 0 <= s[1] < ny and 0 <= g[0] < nx and 0 <= g[1] < ny):
            return []
        if not (self.free[s] and self.free[g]):
            return []

        def hfun(a: tuple[int, int]) -> float:
            dx, dy = abs(a[0] - g[0]), abs(a[1] - g[1])
            return self.h * (max(dx, dy) + (SQ2 - 1) * min(dx, dy))

        open_: list[tuple[float, float, tuple[int, int]]] = [(hfun(s), 0.0, s)]
        came: dict[tuple[int, int], tuple[int, int]] = {}
        cost = {s: 0.0}
        while open_:
            _, c, cur = heapq.heappop(open_)
            if cur == g:
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                return [((i + 0.5) * self.h, (j + 0.5) * self.h) for i, j in reversed(path)]
            if c > cost.get(cur, math.inf):
                continue
            for di, dj, w in _NB:
                nb = (cur[0] + di, cur[1] + dj)
                if not (0 <= nb[0] < nx and 0 <= nb[1] < ny and self.free[nb]):
                    continue
                if di and dj and not (self.free[cur[0] + di, cur[1]] and self.free[cur[0], cur[1] + dj]):
                    continue
                nc = c + w * self.h
                if nc < cost.get(nb, math.inf):
                    cost[nb] = nc
                    came[nb] = cur
                    heapq.heappush(open_, (nc + hfun(nb), nc, nb))
        return []


def standard_platform(n_stairs: int = 3, stair_width: float = 2.4) -> Platform:
    """120 m x 10 m island platform. Stairs alternate between the two long edges, evenly spaced."""
    length, width = 120.0, 10.0
    xs = [length * (k + 0.5) / n_stairs for k in range(n_stairs)]
    exits = [Exit(f"S{k + 1}", x - stair_width / 2, x + stair_width / 2, 0.0 if k % 2 == 0 else width, stair_width)
             for k, x in enumerate(xs)]
    cols = [(x - 0.3, 4.7, x + 0.3, 5.3) for x in np.arange(10.0, length, 20.0)]  # 0.6 m columns on the centreline
    return Platform(length, width, 0.25, exits, cols)
