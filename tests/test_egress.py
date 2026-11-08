from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from egress.api.app import create_app
from egress.geometry import Exit, Platform, standard_platform
from egress.nfpa import hand_calc
from egress.socialforce import neighbour_pairs, simulate


def test_distance_field_and_directions() -> None:
    pf = Platform(20.0, 4.0, 0.25, [Exit("A", 9.0, 11.0, 0.0, 2.0)])
    i, j = pf.cell(np.array([10.0]), np.array([0.1]))
    assert pf.dist[i[0], j[0]] == pytest.approx(0.0)
    i, j = pf.cell(np.array([1.0]), np.array([2.0]))
    assert 8.0 < pf.dist[i[0], j[0]] < 12.0  # geodesic ~ sqrt(8^2 + 2^2) on an 8-connected grid
    d = pf.direction(np.array([1.0, 19.0]), np.array([2.0, 2.0]))
    assert d[0, 0] > 0 and d[1, 0] < 0  # both point towards the exit in the middle


def test_astar_goes_around_obstacle() -> None:
    pf = Platform(20.0, 6.0, 0.25, [Exit("A", 18.0, 19.5, 0.0, 1.5)], obstacles=[(9.0, 0.0, 10.0, 5.0)])
    path = pf.astar((2.0, 1.0), (17.0, 1.0))
    assert path and max(y for _, y in path) >= 5.0  # must go over the wall that leaves a 1 m gap at the top
    assert pf.astar((9.5, 1.0), (17.0, 1.0)) == []  # start inside the obstacle


def test_neighbour_pairs_complete() -> None:
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 20, (300, 2))
    i, j = neighbour_pairs(pos)
    got = {(min(a, b), max(a, b)) for a, b in zip(i, j)}
    d = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    want = {(a, b) for a, b in zip(*np.nonzero(np.triu(d < 1.5, 1)))}
    assert want <= got


def test_small_evacuation_is_complete_and_realistic() -> None:
    pf = standard_platform(3, 2.4)
    r = simulate(pf, 250, 0)
    assert r["evacuated"] == 250 and r["stuck"] == 0
    assert 0.8 < r["specific_flow_p_per_m_s"] < 2.0  # empirical bottleneck range
    assert r["clearance_s"] < 120


def test_nfpa_hand_calc() -> None:
    pf = standard_platform(3, 2.4)
    hc = hand_calc(pf, 900)
    # effective width 1.8 m x 3 x 55.5 p/m/min = 299.7 p/min -> 3.0 min
    assert hc["total_capacity_p_min"] == pytest.approx(299.7)
    assert hc["platform_clearance_min"] == pytest.approx(900 / 299.7, rel=1e-6)
    assert hc["meets_4_min"] and not hand_calc(pf, 1300)["meets_4_min"]


def test_api() -> None:
    c = TestClient(create_app())
    r = c.post("/v1/route", json={"start": [5.0, 5.0], "goal": [20.0, 0.2]})
    assert r.status_code == 200 and r.json()["length_m"] > 15
    assert c.post("/v1/route", json={"start": [-5, 5], "goal": [20, 0.2]}).status_code == 422
    assert c.post("/v1/nfpa", json={"occupants": 900}).json()["platform_clearance_min"] == pytest.approx(3.0, abs=0.01)
