"""Evaluation harness: baselines, metrics, experiments E1-E5 (FR-10).

This is a decision problem with no ground-truth labels, so we evaluate by
comparing *strategies* on shared metrics and by checking expected behaviours.

The key anti-circularity measure: the headline money metric (expected landed
cost) is NOT the scorer's objective. It is computed in dollars from freight,
inventory holding and late penalties, so RouteWise cannot win it by
construction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import CRITERIA, REPORTS_DIR, load_config, profile_for, weights_for
from .constraints import check_route
from .generate_data import load_dataset
from .pareto import pareto_filter
from .recommend import criteria_values, recommend
from .schemas import Route, Shipment
from .scoring import normalise, rank, score
from .sensitivity import stability_by_criterion
from .simulate import simulate_all

STRATEGIES = {
    "B1_cheapest": "Always cheapest",
    "B2_fastest": "Always fastest",
    "B3_lowest_risk": "Always lowest risk",
    "B4_equal_weights": "Equal weights (0.25 each), no profiles, no filters",
    "B5_random": "Random choice (mean of 100 draws)",
    "A1_routewise": "RouteWise (constraints + Pareto + profile weights)",
    "A2_no_filters": "Ablation: RouteWise weights without constraints or Pareto",
}


# --------------------------------------------------------------- yardstick ---
def expected_landed_cost(shipment: Shipment, route: Route, sim, cfg: dict) -> float:
    """Independent money yardstick (report section 1.3).

    freight + inventory holding over expected transit + late penalty x expected
    days late. In dollars, and deliberately not the scorer's objective.
    """
    ev = cfg["evaluation"]
    holding = ev["holding_rate_per_day"] * shipment.value_usd * sim.mean_transit
    penalty = ev["late_penalty_per_day_ratio"] * shipment.value_usd * sim.expected_days_late
    return route.total_cost + holding + penalty


# --------------------------------------------------------------- strategies --
def _choose(
    strategy: str,
    shipment: Shipment,
    routes: list[Route],
    sims: dict,
    cfg: dict,
    rng: np.random.Generator | None = None,
) -> str | list[str]:
    crit = {r.route_id: criteria_values(r, sims[r.route_id]) for r in routes}

    if strategy == "B1_cheapest":
        return min(crit, key=lambda k: crit[k]["cost"])
    if strategy == "B2_fastest":
        return min(crit, key=lambda k: crit[k]["time"])
    if strategy == "B3_lowest_risk":
        return min(crit, key=lambda k: crit[k]["risk"])
    if strategy == "B4_equal_weights":
        norms = normalise(crit, cfg["scoring"]["flat_spread_ratio"])
        scores, _ = score(norms, weights_for("equal", cfg))
        return rank(scores)[0]
    if strategy == "B5_random":
        n = cfg["evaluation"]["random_baseline_draws"]
        ids = list(crit)
        return [ids[int(rng.integers(len(ids)))] for _ in range(n)]
    if strategy == "A2_no_filters":
        prof = profile_for(shipment.priority.value, shipment.cargo_type.value, cfg)
        norms = normalise(crit, cfg["scoring"]["flat_spread_ratio"])
        scores, _ = score(norms, weights_for(prof, cfg))
        return rank(scores)[0]
    raise ValueError(strategy)


# ------------------------------------------------------------------ metrics --
def _regret(chosen: dict[str, float], best: dict[str, float], worst: dict[str, float]) -> dict[str, float]:
    """Normalised regret per criterion: 0 = best available, 1 = worst available."""
    out = {}
    for c in CRITERIA:
        span = worst[c] - best[c]
        out[c] = 0.0 if span <= 1e-12 else (chosen[c] - best[c]) / span
    return out


def evaluate_all(data_dir: Path | None = None, cfg: dict[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_config()
    shipments, routes_by = load_dataset(data_dir)
    sim_cfg = cfg["simulation"]
    rng = np.random.default_rng(123)

    rows: list[dict] = []
    per_shipment: list[dict] = []

    for s in shipments:
        routes = routes_by[s.shipment_id]
        sims = simulate_all(
            routes, s.deadline_days, sim_cfg["n_runs"], sim_cfg["seed"], sim_cfg["lognormal_sigma"]
        )
        crit = {r.route_id: criteria_values(r, sims[r.route_id]) for r in routes}
        best = {c: min(v[c] for v in crit.values()) for c in CRITERIA}
        worst = {c: max(v[c] for v in crit.values()) for c in CRITERIA}
        by_id = {r.route_id: r for r in routes}

        # oracle on the money yardstick, for regret-vs-oracle
        elc = {rid: expected_landed_cost(s, by_id[rid], sims[rid], cfg) for rid in crit}
        oracle_elc = min(elc.values())

        # Pareto set over ALL routes (for "is the pick Pareto-optimal?")
        nd_all, _ = pareto_filter(crit)

        # Regret reference set restricted to routes that actually satisfy the
        # hard constraints. Regret measured against an option the operator is
        # not allowed to book is not regret they can act on -- a cheap route
        # that breaches a hard deadline is not a missed opportunity. Both the
        # all-routes and feasible-only versions are reported.
        feas_ids = [
            rid
            for rid in crit
            if check_route(s, by_id[rid], sims[rid], cfg["constraints"]["hard_deadline_quantile"])[0]
        ]
        ref_ids = feas_ids or list(crit)
        best_f = {c: min(crit[rid][c] for rid in ref_ids) for c in CRITERIA}
        worst_f = {c: max(crit[rid][c] for rid in ref_ids) for c in CRITERIA}

        rec = recommend(s, routes, cfg=cfg, explain=False, with_sensitivity=True)
        stability = rec.stability
        by_crit = stability_by_criterion(rec, cfg)

        for strat in STRATEGIES:
            if strat == "A1_routewise":
                picks = [rec.chosen_route_id]
            elif strat == "B5_random":
                picks = _choose(strat, s, routes, sims, cfg, rng)
            else:
                picks = [_choose(strat, s, routes, sims, cfg)]

            agg: list[dict] = []
            for rid in picks:
                r = by_id[rid]
                sim = sims[rid]
                reg = _regret(crit[rid], best, worst)
                reg_f = _regret(crit[rid], best_f, worst_f)
                feasible, violations, _ = check_route(s, r, sim, cfg["constraints"]["hard_deadline_quantile"])
                agg.append(
                    {
                        "cost": crit[rid]["cost"],
                        "time": crit[rid]["time"],
                        "p_miss": crit[rid]["reliability"],
                        "risk": crit[rid]["risk"],
                        "cost_regret": reg["cost"],
                        "time_regret": reg["time"],
                        "worst_case_regret": max(reg.values()),
                        "mean_regret": float(np.mean(list(reg.values()))),
                        "worst_case_regret_feasible": max(reg_f.values()),
                        "mean_regret_feasible": float(np.mean(list(reg_f.values()))),
                        "expected_landed_cost": elc[rid],
                        "elc_regret_vs_oracle": elc[rid] - oracle_elc,
                        "pareto_optimal": float(rid in nd_all),
                        "violates_constraints": float(not feasible),
                        "n_violations": float(len(violations)),
                    }
                )

            m = {k: float(np.mean([a[k] for a in agg])) for k in agg[0]}
            m.update(
                {
                    "strategy": strat,
                    "shipment_id": s.shipment_id,
                    "priority": s.priority.value,
                    "cargo_type": s.cargo_type.value,
                    "note": s.note or "",
                    "route_id": picks[0] if len(picks) == 1 else "random",
                    "stability": stability if strat == "A1_routewise" else np.nan,
                }
            )
            rows.append(m)

        per_shipment.append(
            {
                "shipment_id": s.shipment_id,
                "priority": s.priority.value,
                "cargo_type": s.cargo_type.value,
                "profile": rec.profile,
                "note": s.note or "",
                "chosen": rec.chosen_route_id,
                "compliant": rec.compliant,
                "margin": rec.margin,
                "raw_near_tie": rec.raw_near_tie,
                "confidence": rec.confidence,
                **{f"flip_{c}": by_crit[c] for c in CRITERIA},
                "stability": stability,
                "n_routes": len(routes),
            }
        )

    df = pd.DataFrame(rows)
    metric_cols = [
        "cost",
        "time",
        "p_miss",
        "risk",
        "cost_regret",
        "time_regret",
        "worst_case_regret",
        "mean_regret",
        "worst_case_regret_feasible",
        "mean_regret_feasible",
        "expected_landed_cost",
        "elc_regret_vs_oracle",
        "pareto_optimal",
        "violates_constraints",
    ]
    summary = (
        df.groupby("strategy")[metric_cols]
        .mean()
        .rename(
            columns={
                "cost": "mean_cost_usd",
                "time": "mean_exp_transit_d",
                "p_miss": "mean_deadline_miss_rate",
                "risk": "mean_risk",
                "pareto_optimal": "share_pareto_optimal",
                "violates_constraints": "constraint_violation_rate",
            }
        )
        .reset_index()
        .sort_values("worst_case_regret")
    )
    summary["description"] = summary["strategy"].map(STRATEGIES)
    return {"raw": df, "summary": summary, "per_shipment": pd.DataFrame(per_shipment)}


# ----------------------------------------------------- E2: profile behaviour --
def profile_behaviour(data_dir: Path | None = None, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Same shipments, forced to each profile. Expect: as priority moves toward
    budget, mean cost falls and mean transit rises."""
    cfg = cfg or load_config()
    shipments, routes_by = load_dataset(data_dir)
    rows = []
    for prof in ("urgent", "standard", "budget"):
        for s in shipments:
            routes = routes_by[s.shipment_id]
            rec = recommend(s, routes, profile=prof, cfg=cfg, explain=False, with_sensitivity=False)
            e = rec.chosen
            rows.append(
                {
                    "profile": prof,
                    "shipment_id": s.shipment_id,
                    "cost": e.criteria["cost"],
                    "exp_transit": e.criteria["time"],
                    "p_miss": e.criteria["reliability"],
                    "risk": e.criteria["risk"],
                }
            )
    df = pd.DataFrame(rows)
    return (
        df.groupby("profile")[["cost", "exp_transit", "p_miss", "risk"]]
        .mean()
        .reindex(["urgent", "standard", "budget"])
        .reset_index()
    )


# ------------------------------------------- E3: noisy-estimate robustness ----
def noisy_estimate_experiment(
    data_dir: Path | None = None, cfg: dict[str, Any] | None = None, seed: int = 99
) -> dict[str, float]:
    """The system sees estimates with 15% noise; ground truth is the clean data.

    Reports how much worse (in expected landed cost) the noisy decision is than
    the decision the system would have made with perfect information.
    """
    cfg = cfg or load_config()
    shipments, routes_by = load_dataset(data_dir)
    sigma = cfg["evaluation"]["noise_experiment_sigma"]
    sim_cfg = cfg["simulation"]
    rng = np.random.default_rng(seed)

    gaps, flips = [], 0
    for s in shipments:
        clean = routes_by[s.shipment_id]
        noisy = [
            r.model_copy(
                update={
                    "total_cost": float(max(1.0, r.total_cost * rng.normal(1, sigma))),
                    "transit_days": float(max(0.1, r.transit_days * rng.normal(1, sigma))),
                    "delay_prob": float(np.clip(r.delay_prob * rng.normal(1, sigma), 0, 1)),
                    "delay_mean_days": float(max(0.0, r.delay_mean_days * rng.normal(1, sigma))),
                }
            )
            for r in clean
        ]
        rec_noisy = recommend(s, noisy, cfg=cfg, explain=False, with_sensitivity=False)
        rec_clean = recommend(s, clean, cfg=cfg, explain=False, with_sensitivity=False)
        flips += rec_noisy.chosen_route_id != rec_clean.chosen_route_id

        # score BOTH picks against the clean ground truth
        sims = simulate_all(
            clean, s.deadline_days, sim_cfg["n_runs"], sim_cfg["seed"], sim_cfg["lognormal_sigma"]
        )
        by_id = {r.route_id: r for r in clean}
        elc = {rid: expected_landed_cost(s, by_id[rid], sims[rid], cfg) for rid in by_id}
        oracle = min(elc.values())
        gaps.append(elc[rec_noisy.chosen_route_id] - oracle)

    return {
        "noise_sigma": sigma,
        "decision_flip_rate": flips / len(shipments),
        "mean_elc_regret_vs_oracle_usd": float(np.mean(gaps)),
        "median_elc_regret_vs_oracle_usd": float(np.median(gaps)),
        "share_matching_oracle": float(np.mean([g <= 1e-9 for g in gaps])),
    }


# ------------------------------------------------------------- E5: cases ----
def write_case_studies(path: Path, cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_config()
    shipments, routes_by = load_dataset()
    picks = []
    for note, title in (
        ("trap", "Trap: the cheapest route misses a hard deadline"),
        ("near_tie", "Near-tie: two almost identical options"),
        ("infeasible", "No feasible route: documented fallback"),
    ):
        s = next(x for x in shipments if x.note == note)
        picks.append((title, s))
    # the lowest-confidence real decision, if distinct
    low = None
    for s in shipments:
        r = recommend(s, routes_by[s.shipment_id], cfg=cfg, explain=False)
        if r.confidence == "Low":
            low = s
            break
    if low is not None:
        picks.append(("Low confidence: flagged for manual review", low))

    lines = [
        "# E5 case studies",
        "",
        "Generated by `routewise evaluate`. Every number below is produced by the engine.",
        "",
    ]
    for title, s in picks:
        rec = recommend(s, routes_by[s.shipment_id], cfg=cfg)
        lines += [
            f"## {title} - {s.shipment_id} ({s.origin} to {s.destination}, "
            f"{s.priority.value} / {s.cargo_type.value})",
            "",
            "```",
            rec.explanation,
            "```",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ------------------------------------------------------------------ runner ----
def run_full_evaluation(out_dir: Path | None = None, cfg: dict[str, Any] | None = None) -> dict:
    cfg = cfg or load_config()
    out = Path(out_dir) if out_dir else REPORTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    res = evaluate_all(cfg=cfg)
    res["raw"].to_csv(out / "e1_raw.csv", index=False)
    res["summary"].to_csv(out / "e1_summary.csv", index=False)
    res["per_shipment"].to_csv(out / "per_shipment.csv", index=False)

    prof = profile_behaviour(cfg=cfg)
    prof.to_csv(out / "e2_profile_behaviour.csv", index=False)

    noisy = noisy_estimate_experiment(cfg=cfg)
    (out / "e3_noisy_estimates.json").write_text(json.dumps(noisy, indent=2), encoding="utf-8")

    ps = res["per_shipment"]
    dataset_level = {
        "n_shipments": int(len(ps)),
        "mean_stability": float(ps["stability"].mean()),
        "flip_rate_under_weight_noise": float(1 - ps["stability"].mean()),
        "low_confidence_shipments": ps.loc[ps["stability"] < 0.70, "shipment_id"].tolist(),
        "non_compliant_shipments": ps.loc[~ps["compliant"], "shipment_id"].tolist(),
        "share_recommendations_pareto_optimal": float(
            res["raw"].query("strategy == 'A1_routewise'")["pareto_optimal"].mean()
        ),
        "near_tie_shipments": ps.loc[
            (ps["margin"].notna() & (ps["margin"] < cfg["scoring"]["near_tie_margin"])) | ps["raw_near_tie"],
            "shipment_id",
        ].tolist(),
        "confidence_counts": {k: int(v) for k, v in ps["confidence"].value_counts().items()},
        "mean_flip_rate_by_criterion_weight": {c: round(float(ps[f"flip_{c}"].mean()), 4) for c in CRITERIA},
        "most_fragile_criterion": max(CRITERIA, key=lambda c: ps[f"flip_{c}"].mean()),
    }
    (out / "e4_sensitivity_summary.json").write_text(json.dumps(dataset_level, indent=2), encoding="utf-8")

    # ---- E5: case studies (trap, near-tie, no-feasible-route) ---------------
    write_case_studies(out / "e5_case_studies.md", cfg)

    return {
        "summary": res["summary"],
        "per_shipment": ps,
        "profile_behaviour": prof,
        "noisy": noisy,
        "sensitivity": dataset_level,
    }


if __name__ == "__main__":  # pragma: no cover
    r = run_full_evaluation()
    print(r["summary"].to_string(index=False))
