"""NFPA 130-style hand calculation of platform evacuation time.

The NFPA 130 method checks that the platform occupant load can **leave the platform within 4 minutes**
and **reach a point of safety within 6 minutes**, using element capacities (persons per unit width per
minute) and walking speeds. The platform-clearance part is::

    T_platform = max over exits ( longest walk to the exit / walking speed ,
                                  persons assigned to the exit / (capacity x usable width) )

The capacity values are configurable because they differ between editions and between level egress
elements and stairs. The defaults below are commonly quoted NFPA 130 values (platforms/corridors about
81.9 p/m/min, stairs in the up direction about 55.5 p/m/min; walking speed 38 m/min on platforms).
**Verify them against the edition and AHJ interpretation for your project**; this module is a screening
calculation, not a code compliance submittal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Platform


@dataclass(frozen=True)
class NFPAParams:
    stair_capacity_p_per_m_min: float = 55.5
    platform_speed_m_min: float = 38.0
    edge_clearance_m: float = 0.3  # per side, deducted from stair width (boundary-layer allowance)


def hand_calc(platform: Platform, occupants: int, p: NFPAParams | None = None) -> dict[str, float]:
    p = p or NFPAParams()
    widths = np.array([max(e.width - 2 * p.edge_clearance_m, 0.1) for e in platform.exits])
    cap = widths * p.stair_capacity_p_per_m_min  # persons/min per exit
    share = cap / cap.sum()  # occupants split in proportion to exit capacity
    queue_min = (occupants * share) / cap
    xs = np.array([(e.x0 + e.x1) / 2 for e in platform.exits])
    longest_walk = max(np.max(np.abs(xs - 0.0)), np.max(np.abs(xs - platform.length)))
    # the farthest person walks to the nearest exit: half the largest gap between exits or ends
    pts = np.sort(np.r_[0.0, xs, platform.length])
    gaps = np.diff(pts)
    walk = max(gaps[0], gaps[-1], (gaps[1:-1].max() / 2) if gaps.size > 2 else 0.0) + platform.width / 2
    walk_min = walk / p.platform_speed_m_min
    t = float(max(queue_min.max(), walk_min))
    return {"occupants": occupants, "total_capacity_p_min": float(cap.sum()), "queue_time_min": float(queue_min.max()),
            "walk_time_min": float(walk_min), "platform_clearance_min": t, "meets_4_min": t <= 4.0,
            "longest_walk_m": float(walk), "unused_longest_end_walk_m": float(longest_walk)}
