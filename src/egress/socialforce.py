"""Vectorised Helbing social-force model with spatial hashing.

For pedestrian ``i`` with mass ``m``, desired speed ``v0_i`` and direction ``e_i`` (from the distance field)::

    m dv_i/dt = m (v0_i e_i - v_i) / tau
              + sum_j [ A exp((r_ij - d_ij)/B) + k g(r_ij - d_ij) ] n_ij + kappa g(r_ij - d_ij) dv_ji^t t_ij
              + wall terms of the same form

(Helbing, Farkas & Vicsek 2000), with ``g(x) = max(x, 0)``. Defaults: m = 80 kg, tau = 0.5 s, A = 2000 N,
B = 0.08 m, radius 0.25-0.30 m. The contact stiffnesses are softened (k = 1.2e4 N/m, kappa = 2.4e4 kg/(m s))
from Helbing's 1.2e5 / 2.4e5, so that a 0.05-s explicit step stays stable; speeds are capped at 2.5 m/s.
Softened contacts alone let agents compress unrealistically in a bottleneck (specific flow of 3.4 p/m/s
in a first test, about twice empirical values), so the desired speed is also limited by the local density
through **Weidmann's fundamental diagram**: ``v0_eff = v0 [1 - exp(-1.913 (1/rho - 1/5.4))]``, with ``rho``
the density within 1 m. The resulting bottleneck flows are checked against empirical data in the study.

**Neighbour search**: agents are hashed into 1.5-m cells; pairs come from each cell and its 4 forward
neighbours (half-stencil), so pair generation is O(N) for bounded density instead of O(N^2).
Walls are the platform edges plus rectangular obstacles; exits are gaps in the edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import Platform

CELL = 1.5
_HALF = ((1, -1), (1, 0), (1, 1), (0, 1))


@dataclass(frozen=True)
class SFParams:
    mass: float = 80.0
    tau: float = 0.5
    a: float = 2000.0
    b: float = 0.08
    k: float = 1.2e4
    kappa: float = 2.4e4
    a_wall: float = 2000.0
    b_wall: float = 0.08
    v_max: float = 2.5
    dt: float = 0.05


def neighbour_pairs(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All pairs within the same or adjacent 1.5-m cells (each unordered pair once)."""
    n = pos.shape[0]
    if n < 2:
        return np.empty(0, int), np.empty(0, int)
    keys = np.floor(pos / CELL).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv, minlength=uniq.shape[0])
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    buckets = {(int(uniq[c, 0]), int(uniq[c, 1])): order[starts[c]:starts[c] + counts[c]] for c in range(uniq.shape[0])}
    ii, jj = [], []
    for (kx, ky), m in buckets.items():
        if m.size > 1:
            a, b = np.triu_indices(m.size, 1)
            ii.append(m[a])
            jj.append(m[b])
        for dx, dy in _HALF:
            o = buckets.get((kx + dx, ky + dy))
            if o is not None:
                ii.append(np.repeat(m, o.size))
                jj.append(np.tile(o, m.size))
    if not ii:
        return np.empty(0, int), np.empty(0, int)
    return np.concatenate(ii), np.concatenate(jj)


@dataclass
class Crowd:
    platform: Platform
    pos: np.ndarray
    vel: np.ndarray
    v0: np.ndarray
    radius: np.ndarray
    p: SFParams = field(default_factory=SFParams)
    t: float = 0.0
    exit_times: list[float] = field(default_factory=list)
    exit_names: list[str] = field(default_factory=list)
    max_density: float = 0.0

    def _walls(self) -> np.ndarray:
        """Force from the long edges (except inside stair openings) and from obstacles."""
        pf, p = self.platform, self.p
        x, y = self.pos[:, 0], self.pos[:, 1]
        f = np.zeros_like(self.pos)
        for edge_y, sign in ((0.0, 1.0), (pf.width, -1.0)):
            d = np.abs(y - edge_y)
            in_opening = np.zeros(x.shape, bool)
            for e in pf.exits:
                if abs(e.y - edge_y) < 1e-9:
                    in_opening |= (x > e.x0) & (x < e.x1)
            overlap = self.radius - d
            mag = p.a_wall * np.exp(overlap / p.b_wall) + p.k * np.maximum(overlap, 0.0)
            f[:, 1] += np.where(in_opening, 0.0, sign * mag)
        for x0, y0, x1, y1 in pf.obstacles:
            cx, cy = np.clip(x, x0, x1), np.clip(y, y0, y1)
            dx, dy = x - cx, y - cy
            d = np.hypot(dx, dy) + 1e-9
            overlap = self.radius - d
            mag = p.a_wall * np.exp(overlap / p.b_wall) + p.k * np.maximum(overlap, 0.0)
            f[:, 0] += mag * dx / d
            f[:, 1] += mag * dy / d
        for end_x, sign in ((0.0, 1.0), (pf.length, -1.0)):
            overlap = self.radius - np.abs(x - end_x)
            f[:, 0] += sign * (p.a_wall * np.exp(overlap / p.b_wall) + p.k * np.maximum(overlap, 0.0))
        return f

    def step(self) -> None:
        p, pf = self.p, self.platform
        n = self.pos.shape[0]
        if n == 0:
            self.t += p.dt
            return
        e = pf.direction(self.pos[:, 0], self.pos[:, 1])
        i, j = neighbour_pairs(self.pos)
        rho = np.full(n, 1.0 / np.pi)  # self only
        if i.size:
            d_vec = self.pos[i] - self.pos[j]
            d = np.linalg.norm(d_vec, axis=1) + 1e-9
            within1 = d < 1.0
            rho = (1.0 + np.bincount(np.r_[i[within1], j[within1]], minlength=n)) / np.pi
        w = np.where(rho < 5.4, 1.0 - np.exp(-1.913 * (1.0 / np.maximum(rho, 0.1) - 1.0 / 5.4)), 0.0)
        v0_eff = self.v0 * np.clip(w, 0.05, 1.0)
        f = p.mass * (v0_eff[:, None] * e - self.vel) / p.tau
        if i.size:
            near = d < 2.0
            i, j, d_vec, d = i[near], j[near], d_vec[near], d[near]
            nrm = d_vec / d[:, None]
            r = self.radius[i] + self.radius[j]
            overlap = r - d
            g = np.maximum(overlap, 0.0)
            tang = np.stack([-nrm[:, 1], nrm[:, 0]], axis=1)
            dvt = np.einsum("ij,ij->i", self.vel[j] - self.vel[i], tang)
            fij = (p.a * np.exp(overlap / p.b) + p.k * g)[:, None] * nrm + (p.kappa * g * dvt)[:, None] * tang
            np.add.at(f, i, fij)
            np.add.at(f, j, -fij)
            cnt = np.bincount(np.r_[i, j], minlength=n)
            self.max_density = max(self.max_density, float(cnt.max() if cnt.size else 0) / (np.pi * 2.0 ** 2))
        f += self._walls()
        self.vel = self.vel + f / p.mass * p.dt
        sp = np.linalg.norm(self.vel, axis=1)
        cap = np.minimum(p.v_max, 1.2 * v0_eff + 0.05)
        self.vel *= np.minimum(1.0, cap / np.maximum(sp, 1e-9))[:, None]
        self.pos = self.pos + self.vel * p.dt
        self.t += p.dt
        # leave through an exit: crossed the edge inside a stair opening
        gone = np.zeros(n, bool)
        for ex in pf.exits:
            inside_x = (self.pos[:, 0] >= ex.x0) & (self.pos[:, 0] <= ex.x1)
            crossed = (self.pos[:, 1] <= 0.0) if ex.y <= 0.0 else (self.pos[:, 1] >= pf.width)
            hit = inside_x & crossed & ~gone
            for _ in range(int(hit.sum())):
                self.exit_times.append(self.t)
                self.exit_names.append(ex.name)
            gone |= hit
        # anyone squeezed through a wall is put back on the platform (numerical safety); only stair
        # openings may be crossed
        self.pos[:, 0] = np.clip(self.pos[:, 0], 0.05, pf.length - 0.05)
        in_any_opening = np.zeros(n, bool)
        for ex in pf.exits:
            in_any_opening |= (self.pos[:, 0] >= ex.x0) & (self.pos[:, 0] <= ex.x1)
        clamp = ~gone & ~in_any_opening
        self.pos[clamp, 1] = np.clip(self.pos[clamp, 1], 0.05, pf.width - 0.05)
        keep = ~gone
        self.pos, self.vel, self.v0, self.radius = self.pos[keep], self.vel[keep], self.v0[keep], self.radius[keep]


def populate(platform: Platform, n: int, rng: np.random.Generator, v0_mean: float = 1.34, v0_sd: float = 0.26) -> Crowd:
    """Uniform random placement on free cells away from the edges, Weidmann-like speed spread."""
    free_i, free_j = np.nonzero(platform.free)
    ok = (free_j * platform.h > 0.6) & (free_j * platform.h < platform.width - 0.6)
    idx = rng.choice(np.flatnonzero(ok), size=n, replace=n > ok.sum())
    pos = np.stack([(free_i[idx] + rng.random(n)) * platform.h, (free_j[idx] + rng.random(n)) * platform.h], axis=1)
    v0 = np.clip(rng.normal(v0_mean, v0_sd, n), 0.5, 2.5)
    radius = rng.uniform(0.25, 0.30, n)
    return Crowd(platform, pos, np.zeros((n, 2)), v0, radius)


def simulate(platform: Platform, n: int, seed: int, *, t_max: float = 900.0, v0_mean: float = 1.34) -> dict[str, float | list[float]]:
    rng = np.random.default_rng(seed)
    c = populate(platform, n, rng, v0_mean=v0_mean)
    while c.pos.shape[0] and c.t < t_max:
        c.step()
    times = np.array(c.exit_times)
    names = np.array(c.exit_names)
    out: dict[str, float | list[float]] = {
        "n": n, "evacuated": int(times.size), "clearance_s": float(times.max()) if times.size else float("nan"),
        "t50_s": float(np.percentile(times, 50)) if times.size else float("nan"),
        "t90_s": float(np.percentile(times, 90)) if times.size else float("nan"),
        "stuck": int(c.pos.shape[0]), "max_local_density": c.max_density,
    }
    # specific flow at each exit during its saturated period (middle 80% of its departures)
    flows = []
    for ex in platform.exits:
        tt = np.sort(times[names == ex.name])
        if tt.size > 20:
            lo, hi = int(0.1 * tt.size), int(0.9 * tt.size)
            flows.append((hi - lo) / max(tt[hi - 1] - tt[lo], 1e-6) / ex.width)
    out["specific_flow_p_per_m_s"] = float(np.mean(flows)) if flows else float("nan")
    return out
