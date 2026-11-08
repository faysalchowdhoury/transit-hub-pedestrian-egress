"""Wayfinding and screening API.

``POST /v1/route``     A* path between two points on the platform (cached distance grid)
``POST /v1/nfpa``      NFPA 130-style hand calculation for a layout and occupant load
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .. import __version__
from ..geometry import Platform, standard_platform
from ..nfpa import hand_calc


class RouteRequest(BaseModel):
    n_stairs: int = Field(3, ge=1, le=6)
    stair_width_m: float = Field(2.4, gt=0.8, le=6)
    start: tuple[float, float]
    goal: tuple[float, float]


class NFPARequest(BaseModel):
    n_stairs: int = Field(3, ge=1, le=6)
    stair_width_m: float = Field(2.4, gt=0.8, le=6)
    occupants: int = Field(ge=1, le=20000)


@lru_cache(maxsize=32)
def _platform(n: int, w: float) -> Platform:
    return standard_platform(n, w)


def create_app() -> FastAPI:
    app = FastAPI(title="Platform egress", version=__version__)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.post("/v1/route")
    def route(r: RouteRequest) -> dict[str, object]:
        path = _platform(r.n_stairs, r.stair_width_m).astar(r.start, r.goal)
        if not path:
            raise HTTPException(422, "start or goal is blocked / outside the platform, or unreachable")
        length = sum(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 for (x1, y1), (x2, y2) in zip(path, path[1:]))
        return {"length_m": round(length, 2), "points": len(path), "path": path}

    @app.post("/v1/nfpa")
    def nfpa(r: NFPARequest) -> dict[str, float]:
        return hand_calc(_platform(r.n_stairs, r.stair_width_m), r.occupants)

    return app
