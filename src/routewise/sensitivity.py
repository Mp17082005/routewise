"""Sensitivity analysis and confidence labelling (FR-9).

Weights are judgemental. The honest response is not to pretend otherwise but
to measure how much the recommendation depends on them: if the same route wins
under 93% of perturbed weightings, the exact numbers did not decide it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import CRITERIA, load_config, profile_for, weights_for
from .constraints import check_route
from .pareto import pareto_filter
from .schemas import Recommendation, Route, Shipment
from .scoring import normalise, rank, score
from .simulate import simulate_all


def _winner(crit_by_id: dict[str, dict[str, float]], weights: dict[str, float], flat_ratio: float) -> str:
    norms = normalise(crit_by_id, flat_ratio)
    scores, _ = score(norms, weights)
    return rank(scores)[0]


def _perturb_weights(rng: np.random.Generator, w: dict[str, float], noise: float) -> dict[str, float]:
    raw = {c: w[c] * rng.uniform(1 - noise, 1 + noise) for c in CRITERIA}
    total = sum(raw.values())
    return {c: v / total for c, v in raw.items()}


def weight_stability(rec: Recommendation, cfg: dict[str, Any] | None = None) -> float:
    """Share of perturbed-weight runs in which the chosen route still wins."""
    cfg = cfg or load_config()
    sc = cfg["sensitivity"]
    flat = cfg["scoring"]["flat_spread_ratio"]
    scored = [e for e in rec.evaluations if e.score is not None]
    if len(scored) <= 1:
        return 1.0
    crit = {e.route.route_id: e.criteria for e in scored}

    rng = np.random.default_rng(sc["seed"])
    wins = 0
    for _ in range(sc["n_runs"]):
        w = _perturb_weights(rng, rec.weights, sc["weight_noise"])
        wins += _winner(crit, w, flat) == rec.chosen_route_id
    return wins / sc["n_runs"]


def stability_by_criterion(rec: Recommendation, cfg: dict[str, Any] | None = None) -> dict[str, float]:
    """Flip rate when only ONE criterion's weight is perturbed (+/-noise),
    others rescaled. Identifies which criterion the decision is most fragile to."""
    cfg = cfg or load_config()
    sc = cfg["sensitivity"]
    flat = cfg["scoring"]["flat_spread_ratio"]
    scored = [e for e in rec.evaluations if e.score is not None]
    if len(scored) <= 1:
        return dict.fromkeys(CRITERIA, 0.0)
    crit = {e.route.route_id: e.criteria for e in scored}
    out = {}
    for c in CRITERIA:
        rng = np.random.default_rng(sc["seed"] + 11)
        flips = 0
        n = sc["n_runs"] // 2
        for _ in range(n):
            w = dict(rec.weights)
            w[c] = w[c] * rng.uniform(1 - 2 * sc["weight_noise"], 1 + 2 * sc["weight_noise"])
            tot = sum(w.values())
            w = {k: v / tot for k, v in w.items()}
            flips += _winner(crit, w, flat) != rec.chosen_route_id
        out[c] = flips / n
    return out


def input_stability(rec: Recommendation, cfg: dict[str, Any] | None = None) -> float:
    """Share of runs where the winner survives noise in delay and risk estimates."""
    cfg = cfg or load_config()
    sc = cfg["sensitivity"]
    flat = cfg["scoring"]["flat_spread_ratio"]
    scored = [e for e in rec.evaluations if e.score is not None]
    if len(scored) <= 1:
        return 1.0

    rng = np.random.default_rng(sc["seed"] + 1)
    wins = 0
    for _ in range(sc["n_runs"]):
        crit = {}
        for e in scored:
            c = dict(e.criteria)
            c["reliability"] = float(
                np.clip(
                    c["reliability"] * rng.uniform(1 - sc["delay_prob_noise"], 1 + sc["delay_prob_noise"]),
                    0,
                    1,
                )
            )
            c["risk"] = float(np.clip(c["risk"] + rng.uniform(-sc["risk_noise"], sc["risk_noise"]), 0, 1))
            c["time"] = c["time"] * rng.uniform(1 - 0.05, 1 + 0.05)
            crit[e.route.route_id] = c
        wins += _winner(crit, rec.weights, flat) == rec.chosen_route_id
    return wins / sc["n_runs"]


def confidence_label(stability: float, margin: float | None, cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg or load_config()
    c = cfg["sensitivity"]["confidence"]
    if stability >= c["high_stability"] and (margin is None or margin >= c["high_margin"]):
        return "High"
    if stability >= c["medium_stability"]:
        return "Medium"
    return "Low"


def attach_confidence(
    rec: Recommendation, routes: list[Route], cfg: dict[str, Any] | None = None
) -> Recommendation:
    cfg = cfg or load_config()
    stab = weight_stability(rec, cfg)
    rec.stability = round(stab, 4)
    rec.confidence = confidence_label(stab, rec.margin, cfg)
    if rec.raw_near_tie and rec.confidence == "High":
        # Stable only because the routes are near-identical; never "High".
        rec.confidence = "Medium"

    if rec.confidence == "Low":
        rec.warnings.append("Close call — review manually before booking.")
    return rec


def weight_sweep(
    shipment: Shipment,
    routes: list[Route],
    criterion: str,
    cfg: dict[str, Any] | None = None,
    steps: int = 41,
) -> list[tuple[float, str]]:
    """One-at-a-time sweep: vary one weight from 0 to 1, rescaling the others.

    Returns [(weight_value, winning_route_id)] — the points where the winner
    changes are the *switching points*, which is the most operator-useful
    output of the whole sensitivity module.
    """
    cfg = cfg or load_config()
    if criterion not in CRITERIA:
        raise ValueError(f"unknown criterion {criterion!r}")

    prof = profile_for(shipment.priority.value, shipment.cargo_type.value, cfg)
    base = weights_for(prof, cfg)
    others = [c for c in CRITERIA if c != criterion]
    base_other_total = sum(base[c] for c in others)

    sim_cfg = cfg["simulation"]
    sims = simulate_all(
        routes, shipment.deadline_days, sim_cfg["n_runs"], sim_cfg["seed"], sim_cfg["lognormal_sigma"]
    )
    from .recommend import criteria_values

    crit_all = {r.route_id: criteria_values(r, sims[r.route_id]) for r in routes}
    feasible = [
        r.route_id
        for r in routes
        if check_route(shipment, r, sims[r.route_id], cfg["constraints"]["hard_deadline_quantile"])[0]
    ]
    if not feasible:
        feasible = list(crit_all)
    nd, _ = pareto_filter(crit_all, feasible)
    crit = {rid: crit_all[rid] for rid in nd}

    out = []
    for wv in np.linspace(0.0, 1.0, steps):
        if base_other_total <= 0:
            w = {c: (1.0 if c == criterion else 0.0) for c in CRITERIA}
        else:
            w = {c: base[c] / base_other_total * (1 - wv) for c in others}
            w[criterion] = float(wv)
        out.append((round(float(wv), 4), _winner(crit, w, cfg["scoring"]["flat_spread_ratio"])))
    return out


def switching_points(sweep: list[tuple[float, str]]) -> list[tuple[float, str, str]]:
    pts = []
    for (w0, a), (w1, b) in zip(sweep[:-1], sweep[1:], strict=True):
        if a != b:
            pts.append((round((w0 + w1) / 2, 4), a, b))
    return pts
