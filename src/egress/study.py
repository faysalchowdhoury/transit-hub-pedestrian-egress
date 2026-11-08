"""Occupant load x stair layout study (+ faster-is-slower), compared with the NFPA 130 hand calculation."""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .geometry import standard_platform
from .nfpa import hand_calc
from .socialforce import simulate

LAYOUTS = {"2 stairs x 2.4 m": (2, 2.4), "3 stairs x 2.4 m": (3, 2.4), "3 stairs x 1.8 m": (3, 1.8)}
LOADS = (400, 1000, 1600)


def _one(args: tuple[str, int, int, float]) -> dict[str, Any]:
    layout, n, seed, v0 = args
    k, w = LAYOUTS[layout]
    pf = standard_platform(k, w)
    t0 = time.perf_counter()
    r = simulate(pf, n, seed, v0_mean=v0)
    hc = hand_calc(pf, n)
    return {"layout": layout, "occupants": n, "seed": seed, "v0_mean": v0, "runtime_s": time.perf_counter() - t0, **r,
            "nfpa_platform_min": hc["platform_clearance_min"], "nfpa_meets_4min": hc["meets_4_min"]}


def run(out_dir: str | Path, seeds: int = 5, workers: int = 2) -> dict[str, Any]:
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    jobs = [(lay, n, s, 1.34) for lay in LAYOUTS for n in LOADS for s in range(seeds)]
    jobs += [("3 stairs x 2.4 m", 1600, s, 2.0) for s in range(seeds)]  # faster-is-slower check
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(_one, jobs))
    df = pd.DataFrame(rows)
    df.to_csv(out / "runs.csv", index=False)
    g = df.groupby(["layout", "occupants", "v0_mean"]).agg(
        clearance_s=("clearance_s", "mean"), clearance_sd=("clearance_s", "std"), t90_s=("t90_s", "mean"),
        flow=("specific_flow_p_per_m_s", "mean"), density=("max_local_density", "mean"), stuck=("stuck", "sum"),
        nfpa_min=("nfpa_platform_min", "first"), runtime_s=("runtime_s", "mean")).reset_index()
    g["sim_min"] = g["clearance_s"] / 60.0
    g["sim_meets_4min"] = g["sim_min"] <= 4.0
    g["nfpa_meets_4min"] = g["nfpa_min"] <= 4.0
    g["nfpa_over_sim"] = g["nfpa_min"] / g["sim_min"]
    summary = {"runs": len(df), "wall_s": time.perf_counter() - t0, "table": g.to_dict("records")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    _plot(g, out / "figures")
    return summary


def _plot(g: pd.DataFrame, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = g[g["v0_mean"] == 1.34]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for (lay, sub), col in zip(base.groupby("layout"), ("#e03131", "#1c7ed6", "#f59f00")):
        sub = sub.sort_values("occupants")
        ax.errorbar(sub["occupants"], sub["sim_min"], yerr=sub["clearance_sd"] / 60, marker="o", color=col, capsize=3, label=f"{lay}: simulated")
        ax.plot(sub["occupants"], sub["nfpa_min"], ls="--", color=col, label=f"{lay}: NFPA hand calc")
    ax.axhline(4.0, color="k", ls=":", lw=1)
    ax.text(410, 4.05, "4-min platform clearance criterion", fontsize=8)
    ax.set_xlabel("platform occupant load (persons)")
    ax.set_ylabel("time to clear the platform (min)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "clearance_vs_load.png", dpi=130)
    plt.close(fig)


def write_report(s: dict[str, Any], out: Path) -> Path:
    g = pd.DataFrame(s["table"]).sort_values(["v0_mean", "layout", "occupants"])
    L = ["# Platform evacuation results", "", f"{s['runs']} runs (5 seeds per cell) in {s['wall_s']:.0f} s. "
         "Clearance = time the last person leaves the platform.", "",
         "| Layout | Occupants | Desired speed | Simulated clearance (min) | t90 (min) | Specific flow (p/m/s) | Peak local density (p/m²) | NFPA hand calc (min) | NFPA / sim | Meets 4 min (sim / NFPA) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in g.itertuples():
        L.append(f"| {r.layout} | {r.occupants} | {r.v0_mean:.2f} m/s | {r.sim_min:.2f} ± {r.clearance_sd / 60:.2f} | {r.t90_s / 60:.2f} | "
                 f"{r.flow:.2f} | {r.density:.1f} | {r.nfpa_min:.2f} | {r.nfpa_over_sim:.2f} | "
                 f"{'yes' if r.sim_meets_4min else 'NO'} / {'yes' if r.nfpa_meets_4min else 'NO'} |")
    L += ["", "![Clearance](figures/clearance_vs_load.png)", ""]
    md = out / "RESULTS.md"
    md.write_text("\n".join(L), encoding="utf-8")
    return md
