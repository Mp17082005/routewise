"""Pipeline orchestration (FR-6).

shipment -> validate -> hard constraints -> Pareto -> Monte Carlo -> score
-> rank -> explain -> confidence
"""

from __future__ import annotations

from typing import Any

from .config import CRITERIA, load_config, profile_for, weights_for
from .constraints import check_route
from .pareto import pareto_filter
from .schemas import Recommendation, Route, RouteEvaluation, Shipment
from .scoring import normalise, rank, score
from .simulate import simulate_all


def criteria_values(route: Route, sim) -> dict[str, float]:
    """The four decision criteria, all lower-is-better."""
    return {
        "cost": route.total_cost,
        "time": route.expected_transit_days,
        "reliability": sim.p_miss_deadline,
        "risk": route.risk_score,
    }


def recommend(
    shipment: Shipment,
    routes: list[Route],
    profile: str | None = None,
    cfg: dict[str, Any] | None = None,
    weights: dict[str, float] | None = None,
    explain: bool = True,
    with_sensitivity: bool = True,
) -> Recommendation:
    if not routes:
        raise ValueError(f"{shipment.shipment_id}: no route options supplied")

    cfg = cfg or load_config()
    prof = profile or profile_for(shipment.priority.value, shipment.cargo_type.value, cfg)
    w = dict(weights) if weights else weights_for(prof, cfg)

    sim_cfg = cfg["simulation"]
    sims = simulate_all(
        routes,
        shipment.deadline_days,
        n_runs=sim_cfg["n_runs"],
        seed=sim_cfg["seed"],
        sigma=sim_cfg["lognormal_sigma"],
    )

    evals: dict[str, RouteEvaluation] = {}
    for r in routes:
        s = sims[r.route_id]
        feasible, violations, magnitude = check_route(
            shipment, r, s, cfg["constraints"]["hard_deadline_quantile"]
        )
        evals[r.route_id] = RouteEvaluation(
            route=r,
            sim=s,
            feasible=feasible,
            violations=violations,
            violation_magnitude=magnitude,
            criteria=criteria_values(r, s),
        )

    # ---- Stage 1: hard constraints, with documented fallback --------------
    feasible_ids = [rid for rid, e in evals.items() if e.feasible]
    fallback_used = False
    if feasible_ids:
        candidate_ids = feasible_ids
    else:
        fallback_used = True
        # least-violating route(s): smallest normalised total violation
        best_mag = min(e.violation_magnitude for e in evals.values())
        candidate_ids = [rid for rid, e in evals.items() if e.violation_magnitude <= best_mag + 1e-9]

    # ---- Stage 2: Pareto filter -------------------------------------------
    crit = {rid: evals[rid].criteria for rid in candidate_ids}
    non_dominated, dominated_by = pareto_filter(crit, candidate_ids)
    for rid, dom in dominated_by.items():
        evals[rid].dominated_by = dom

    # ---- Stage 3: score the survivors -------------------------------------
    sub = {rid: evals[rid].criteria for rid in non_dominated}
    norms = normalise(sub, cfg["scoring"]["flat_spread_ratio"])
    scores, contribs = score(norms, w)
    order = rank(scores)

    for pos, rid in enumerate(order, start=1):
        evals[rid].normalised = norms[rid]
        evals[rid].contributions = contribs[rid]
        evals[rid].score = round(scores[rid], 6)
        evals[rid].rank = pos

    chosen_id = order[0]
    margin = round(scores[order[1]] - scores[order[0]], 6) if len(order) > 1 else None

    # Raw-closeness check. Min-max normalisation stretches any spread to 0..1,
    # so two routes 2% apart can show a large *score* margin. We therefore also
    # compare the winner and runner-up on raw values: if no criterion differs
    # by more than `raw_tie_ratio` (relative; absolute for probabilities and
    # risk), the call is a near-tie regardless of the score margin.
    raw_near_tie = False
    if len(order) > 1:
        a, b = evals[order[0]].criteria, evals[order[1]].criteria
        tol = cfg["scoring"].get("raw_tie_ratio", 0.05)
        diffs = []
        for c in CRITERIA:
            if c in ("reliability", "risk"):
                diffs.append(abs(a[c] - b[c]))  # already on a 0..1 scale
            else:
                diffs.append(abs(a[c] - b[c]) / max(abs(a[c]), abs(b[c]), 1e-9))
        raw_near_tie = max(diffs) < tol

    warnings: list[str] = []
    if fallback_used:
        warnings.append(
            "NON-COMPLIANT: no route satisfies all hard constraints; "
            "the least-violating option is shown and must be approved manually."
        )
    if margin is not None and margin < cfg["scoring"]["near_tie_margin"]:
        warnings.append(f"Near-tie: the runner-up is only {margin:.3f} score points behind.")
    elif raw_near_tie:
        warnings.append(
            f"Near-tie: {order[1]} is within "
            f"{cfg['scoring'].get('raw_tie_ratio', 0.05):.0%} of the chosen route on every "
            f"criterion; either is defensible."
        )
    if evals[chosen_id].route.geo_risk >= 0.65:
        warnings.append(
            f"High geopolitical exposure: the chosen route peaks at "
            f"{evals[chosen_id].route.geo_risk:.2f} risk"
            + (" (chokepoint transit)." if evals[chosen_id].route.uses_chokepoint else ".")
        )

    rec = Recommendation(
        shipment=shipment,
        profile=prof,
        weights=w,
        chosen_route_id=chosen_id,
        evaluations=[evals[r.route_id] for r in routes],
        compliant=not fallback_used,
        fallback_used=fallback_used,
        margin=margin,
        raw_near_tie=raw_near_tie,
        warnings=warnings,
    )

    if with_sensitivity:
        from .sensitivity import attach_confidence

        attach_confidence(rec, routes, cfg)

    if explain:
        from .explain import explain_recommendation

        rec.explanation = explain_recommendation(rec)

    return rec


def recommend_all(
    shipments: list[Shipment],
    routes_by_shipment: dict[str, list[Route]],
    **kwargs,
) -> list[Recommendation]:
    return [
        recommend(s, routes_by_shipment[s.shipment_id], **kwargs)
        for s in shipments
        if routes_by_shipment.get(s.shipment_id)
    ]


__all__ = ["recommend", "recommend_all", "criteria_values", "CRITERIA"]
