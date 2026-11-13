"""Stage 1: hard constraints.

Hard constraints are not traded away. A cheap route that misses a hard
deadline is *invalid*, not merely low-scoring. If nothing passes we never
return an empty answer: we return the least-violating route, flagged
NON-COMPLIANT, with the failed constraints listed.
"""

from __future__ import annotations

from .schemas import Route, Shipment, SimResult


def check_route(
    shipment: Shipment,
    route: Route,
    sim: SimResult,
    hard_deadline_quantile: int = 90,
) -> tuple[bool, list[str], float]:
    """Return (feasible, violations, normalised violation magnitude)."""
    violations: list[str] = []
    magnitude = 0.0

    if shipment.hard_deadline:
        transit_check = sim.p90_transit if hard_deadline_quantile == 90 else sim.p50_transit
        if transit_check > shipment.deadline_days:
            over = transit_check - shipment.deadline_days
            violations.append(
                f"P{hard_deadline_quantile} transit {transit_check:.1f}d exceeds "
                f"hard deadline {shipment.deadline_days:.1f}d by {over:.1f}d"
            )
            magnitude += over / max(shipment.deadline_days, 1e-9)

    if shipment.budget_cap is not None and route.total_cost > shipment.budget_cap:
        over = route.total_cost - shipment.budget_cap
        violations.append(
            f"cost ${route.total_cost:,.0f} exceeds budget cap ${shipment.budget_cap:,.0f} by ${over:,.0f}"
        )
        magnitude += over / max(shipment.budget_cap, 1e-9)

    if shipment.max_geo_risk is not None and route.geo_risk > shipment.max_geo_risk:
        over = route.geo_risk - shipment.max_geo_risk
        violations.append(
            f"geopolitical risk {route.geo_risk:.2f} exceeds the "
            f"{shipment.cargo_type.value} cargo limit {shipment.max_geo_risk:.2f}"
        )
        magnitude += over / max(shipment.max_geo_risk, 1e-9)

    return (not violations), violations, round(magnitude, 6)
