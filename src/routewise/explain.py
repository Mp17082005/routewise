"""Template-based explanation engine (FR-7).

Deliberately not an LLM: every number printed here is read directly from the
data or from a computed score, so explanations are exact, testable, and cannot
hallucinate. A test asserts that every figure in the text matches the data.
"""

from __future__ import annotations

from .config import CRITERIA
from .schemas import Recommendation, RouteEvaluation

CRITERION_LABEL = {
    "cost": "cost",
    "time": "expected transit time",
    "reliability": "risk of missing the deadline",
    "risk": "route risk (congestion + geopolitical)",
}


def _fmt_delta_for(crit: str, chosen: RouteEvaluation, other: RouteEvaluation) -> str:
    a, b = chosen.criteria[crit], other.criteria[crit]
    if crit == "cost":
        pct = (a - b) / b * 100 if b else 0.0
        return f"${a:,.0f} vs ${b:,.0f} ({pct:+.1f}%)"
    if crit == "time":
        return f"{a:.1f}d vs {b:.1f}d ({a - b:+.1f}d)"
    if crit == "reliability":
        return f"{a:.0%} vs {b:.0%} ({(a - b) * 100:+.0f} pts)"
    return f"{a:.2f} vs {b:.2f} ({a - b:+.2f})"


def _material(crit: str, a: RouteEvaluation, b: RouteEvaluation) -> bool:
    """True if the difference survives display rounding (so we never print
    "+0.0d" as a trade-off)."""
    x, y = a.criteria[crit], b.criteria[crit]
    if crit == "cost":
        return abs(x - y) >= 1.0
    if crit == "time":
        return abs(x - y) >= 0.05
    if crit == "reliability":
        return abs(x - y) >= 0.005
    return abs(x - y) >= 0.005


def _gains(chosen: RouteEvaluation, other: RouteEvaluation) -> str:
    """Criteria where `chosen` is genuinely better than `other`, phrased."""
    out = []
    pa, pb = chosen.sim.p_miss_deadline, other.sim.p_miss_deadline
    if pb - pa >= 0.01:
        out.append(f"deadline-miss risk falls from {pb:.0%} to {pa:.0%}")
    ta, tb = chosen.criteria["time"], other.criteria["time"]
    if tb - ta >= 0.1:
        out.append(f"it arrives {tb - ta:.1f}d sooner on average")
    ra, rb = chosen.route.risk_score, other.route.risk_score
    if rb - ra >= 0.01:
        out.append(f"route risk falls from {rb:.2f} to {ra:.2f}")
    return ", ".join(out)


def explain_recommendation(rec: Recommendation) -> str:
    s = rec.shipment
    chosen = rec.chosen
    r = chosen.route
    scored = rec.scored
    all_evals = rec.evaluations
    lines: list[str] = []

    # ---- 1. Headline -------------------------------------------------------
    path = " > ".join(r.hub_path)
    lines.append(
        f"Recommended: {r.route_id} via {path} "
        f"({r.mode_mix}, {r.n_transshipments} transshipment"
        f"{'s' if r.n_transshipments != 1 else ''})."
    )
    lines.append(
        f"Cost ${r.total_cost:,.0f} | expected transit "
        f"{chosen.criteria['time']:.1f}d (P50 {chosen.sim.p50_transit:.1f}d, "
        f"P90 {chosen.sim.p90_transit:.1f}d) | "
        f"{chosen.sim.p_miss_deadline:.0%} chance of missing the "
        f"{s.deadline_days:.1f}d {'hard' if s.hard_deadline else 'soft'} deadline | "
        f"route risk {r.risk_score:.2f}."
    )
    lines.append(
        f"Weighting profile: {rec.profile} "
        f"(cost {rec.weights['cost']:.2f}, time {rec.weights['time']:.2f}, "
        f"reliability {rec.weights['reliability']:.2f}, risk {rec.weights['risk']:.2f}) — "
        f"{s.priority.value} priority, {s.cargo_type.value} cargo."
    )

    # ---- 2. Why it won, vs the runner-up ----------------------------------
    if len(scored) > 1:
        runner = scored[1]
        gaps = sorted(
            CRITERIA,
            key=lambda c: chosen.contributions[c] - runner.contributions[c],
        )
        wins = [
            c
            for c in gaps
            if chosen.contributions[c] < runner.contributions[c] - 1e-9 and _material(c, chosen, runner)
        ][:3]
        losses = [
            c
            for c in gaps[::-1]
            if chosen.contributions[c] > runner.contributions[c] + 1e-9 and _material(c, chosen, runner)
        ]

        lines.append("")
        lines.append(f"Why it beats the runner-up {runner.route.route_id}:")
        if not wins and not any(_material(c, chosen, runner) for c in CRITERIA):
            lines.append(
                "  - the two routes are effectively equal on every criterion; either is "
                "defensible and the pick is decided by rounding-level differences."
            )
        elif wins:
            for c in wins:
                lines.append(f"  - better {CRITERION_LABEL[c]}: {_fmt_delta_for(c, chosen, runner)}")
        else:
            lines.append("  - it wins on the weighted total, with no single dominant criterion.")

        # ---- 3. Trade-off accepted ----------------------------------------
        if not any(_material(c, chosen, runner) for c in CRITERIA):
            pass
        elif losses:
            lines.append("Trade-off accepted:")
            for c in losses[:2]:
                lines.append(f"  - worse {CRITERION_LABEL[c]}: {_fmt_delta_for(c, chosen, runner)}")
        else:
            lines.append(
                f"Trade-off accepted: none against {runner.route.route_id} — "
                f"it is at least as good on every criterion."
            )
        lines.append(
            f"Score {chosen.score:.3f} vs {runner.score:.3f} (margin {rec.margin:.3f}; lower is better)."
        )

    # ---- 4. Versus cheapest and versus fastest -----------------------------
    # Phrasing is driven by the sign of each delta, so a sentence never claims
    # an improvement that did not happen.
    cheapest = min(all_evals, key=lambda e: (e.criteria["cost"], e.route.route_id))
    fastest = min(all_evals, key=lambda e: (e.criteria["time"], e.route.route_id))
    lines.append("")
    if cheapest.route.route_id != r.route_id:
        d_cost = chosen.criteria["cost"] - cheapest.criteria["cost"]
        pct = d_cost / cheapest.criteria["cost"] * 100
        gains = _gains(chosen, cheapest)
        lines.append(
            f"vs cheapest ({cheapest.route.route_id}): we pay ${d_cost:,.0f} more ({pct:+.1f}%)"
            + (
                f"; in exchange {gains}."
                if gains
                else "; it is not better on any other criterion, so the premium is driven by "
                "the hard constraints."
                if not cheapest.feasible
                else "."
            )
        )
        if not cheapest.feasible and cheapest.violations:
            lines.append(f"  The cheapest option is not allowed: {cheapest.violations[0]}.")
    else:
        lines.append("vs cheapest: this IS the cheapest option.")

    if fastest.route.route_id != r.route_id:
        d_time = chosen.criteria["time"] - fastest.criteria["time"]
        d_cost = fastest.criteria["cost"] - chosen.criteria["cost"]
        cost_txt = f"cost ${d_cost:,.0f} more" if d_cost > 0 else f"cost ${-d_cost:,.0f} less"
        why = []
        if not fastest.feasible and fastest.violations:
            why.append(f"it is not allowed ({fastest.violations[0]})")
        if fastest.route.geo_risk > r.geo_risk + 0.05:
            why.append(f"it carries {fastest.route.geo_risk:.2f} geopolitical risk vs {r.geo_risk:.2f}")
        if fastest.dominated_by:
            why.append(f"it is dominated by {fastest.dominated_by}")
        when = (
            f"it would arrive {d_time:.1f}d sooner"
            if d_time >= 0.05
            else "it arrives at effectively the same time"
        )
        lines.append(
            f"vs fastest ({fastest.route.route_id}): {when} "
            f"and {cost_txt}"
            + (
                f", but {'; '.join(why)}."
                if why
                else f"; it loses on the weighted trade-off under the {rec.profile} profile."
            )
        )
    else:
        lines.append("vs fastest: this IS the fastest option.")

    # ---- 5. Eliminated routes ---------------------------------------------
    eliminated = [e for e in all_evals if e.score is None]
    if eliminated:
        lines.append("")
        lines.append("Eliminated options:")
        for e in eliminated:
            if e.violations:
                lines.append(f"  - {e.route.route_id}: {'; '.join(e.violations)}")
            elif e.dominated_by:
                dom = rec.by_id(e.dominated_by).route
                lines.append(
                    f"  - {e.route.route_id}: dominated by {dom.route_id}, which is no worse "
                    f"on cost, time, reliability and risk — and better on at least one."
                )
            else:
                lines.append(f"  - {e.route.route_id}: removed before scoring.")

    # ---- 6. Confidence ------------------------------------------------------
    lines.append("")
    if rec.confidence:
        lines.append(
            f"Confidence: {rec.confidence} — the same route wins in "
            f"{rec.stability:.0%} of 500 runs with every weight randomly "
            f"perturbed by up to +/-20%."
        )

    # ---- 7. Warnings --------------------------------------------------------
    if rec.warnings:
        lines.append("")
        for w in rec.warnings:
            lines.append(f"WARNING: {w}")
        if rec.fallback_used:
            lines.append("  Failed constraints on the selected option: " + "; ".join(chosen.violations))

    return "\n".join(lines)


def comparison_table(rec: Recommendation) -> list[dict]:
    """Full route table — every option, scored or eliminated, with reason."""
    rows = []
    for e in rec.evaluations:
        rows.append(
            {
                "route_id": e.route.route_id,
                "chosen": e.route.route_id == rec.chosen_route_id,
                "path": " > ".join(e.route.hub_path),
                "modes": e.route.mode_mix,
                "cost_usd": round(e.criteria["cost"], 2),
                "exp_transit_d": round(e.criteria["time"], 2),
                "p50_d": round(e.sim.p50_transit, 2),
                "p90_d": round(e.sim.p90_transit, 2),
                "p_miss_deadline": round(e.criteria["reliability"], 4),
                "risk": round(e.criteria["risk"], 4),
                "geo_risk": e.route.geo_risk,
                "congestion": e.route.congestion,
                "score": e.score,
                "rank": e.rank,
                "status": (
                    "chosen"
                    if e.route.route_id == rec.chosen_route_id
                    else "scored"
                    if e.score is not None
                    else "infeasible"
                    if e.violations
                    else f"dominated by {e.dominated_by}"
                    if e.dominated_by
                    else "excluded"
                ),
                "reason": "; ".join(e.violations)
                if e.violations
                else (f"dominated by {e.dominated_by}" if e.dominated_by else ""),
            }
        )
    return rows
