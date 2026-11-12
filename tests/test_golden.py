"""Golden cases: hand-built shipments with a known right answer (TRD 3.9)."""

from __future__ import annotations

from routewise.recommend import recommend
from tests.conftest import make_route, make_shipment


# A deliberate 3-option menu used across the golden cases:
#   FAST  - quick, reliable, expensive, clean corridor
#   CHEAP - slow, unreliable, cheapest, congested
#   SAFE  - middling speed/cost, lowest risk, avoids the chokepoint
#   CHOKE - fastest-cheap combo but transits a high geopolitical-risk chokepoint
def menu(sid="S1"):
    return [
        make_route(
            "FAST",
            sid,
            cost=30_000,
            transit=14,
            delay_prob=0.08,
            delay_mean=1.0,
            congestion=0.30,
            geo_risk=0.20,
        ),
        make_route(
            "CHEAP",
            sid,
            cost=10_000,
            transit=30,
            delay_prob=0.45,
            delay_mean=5.0,
            congestion=0.70,
            geo_risk=0.30,
        ),
        make_route(
            "SAFE",
            sid,
            cost=20_000,
            transit=22,
            delay_prob=0.12,
            delay_mean=1.5,
            congestion=0.25,
            geo_risk=0.10,
        ),
    ]


def test_urgent_shipment_picks_the_fast_route():
    s = make_shipment(priority="urgent", deadline=40, hard=True)
    rec = recommend(s, menu(), explain=False, with_sensitivity=False)
    assert rec.chosen_route_id == "FAST"


def test_budget_shipment_picks_the_cheap_route():
    s = make_shipment(priority="budget", deadline=60, hard=True)
    rec = recommend(s, menu(), explain=False, with_sensitivity=False)
    assert rec.chosen_route_id == "CHEAP"


def test_fragile_cargo_avoids_the_chokepoint():
    """Critical cargo weights risk heavily and has a geo-risk cap."""
    routes = menu() + [
        make_route(
            "CHOKE",
            "S1",
            cost=12_000,
            transit=16,
            delay_prob=0.20,
            delay_mean=3.0,
            congestion=0.60,
            geo_risk=0.85,
            uses_chokepoint=True,
        )
    ]
    s = make_shipment(priority="standard", cargo_type="fragile", deadline=40, hard=True, max_geo=0.65)
    rec = recommend(s, routes, explain=False, with_sensitivity=False)
    assert rec.profile == "critical"
    assert rec.chosen_route_id != "CHOKE"
    assert rec.chosen_route_id == "SAFE"
    choke = rec.by_id("CHOKE")
    assert not choke.feasible and any("geopolitical" in v for v in choke.violations)


def test_trap_case_cheapest_route_misses_hard_deadline():
    """The cheapest route's P90 breaches a hard deadline: it must be excluded
    even though a budget profile weights cost at 0.60."""
    s = make_shipment(priority="budget", deadline=31, hard=True)
    rec = recommend(s, menu(), explain=False, with_sensitivity=False)
    assert not rec.by_id("CHEAP").feasible
    assert rec.chosen_route_id != "CHEAP"
    assert rec.compliant


def test_no_feasible_route_triggers_fallback_and_flag():
    s = make_shipment(priority="standard", deadline=5, hard=True, budget=1_000)
    rec = recommend(s, menu(), explain=True, with_sensitivity=False)
    assert rec.fallback_used and not rec.compliant
    assert any("NON-COMPLIANT" in w for w in rec.warnings)
    assert rec.chosen_route_id  # never empty
    assert "NON-COMPLIANT" in rec.explanation
    # the fallback must be the LEAST-violating option
    chosen_mag = rec.chosen.violation_magnitude
    assert chosen_mag == min(e.violation_magnitude for e in rec.evaluations)


def test_urgent_is_never_slower_than_budget_on_the_same_menu():
    routes = menu()
    urgent = recommend(
        make_shipment(priority="urgent", deadline=60), routes, explain=False, with_sensitivity=False
    )
    budget = recommend(
        make_shipment(priority="budget", deadline=60), routes, explain=False, with_sensitivity=False
    )
    assert urgent.chosen.criteria["time"] <= budget.chosen.criteria["time"]
    assert urgent.chosen.criteria["cost"] >= budget.chosen.criteria["cost"]


def test_dominated_route_is_never_recommended():
    routes = menu() + [
        make_route(
            "WORSE",
            "S1",
            cost=35_000,
            transit=32,
            delay_prob=0.50,
            delay_mean=6.0,
            congestion=0.80,
            geo_risk=0.40,
        )
    ]
    for prio in ("urgent", "standard", "budget"):
        rec = recommend(
            make_shipment(priority=prio, deadline=60), routes, explain=False, with_sensitivity=False
        )
        assert rec.chosen_route_id != "WORSE"
        assert rec.by_id("WORSE").dominated_by is not None


def test_recommendation_is_always_pareto_optimal_on_the_real_dataset(dataset):
    """Success criterion from the PRD: 100% of picks are non-dominated."""
    from routewise.pareto import pareto_filter

    shipments, routes_by = dataset
    for s in shipments:
        rec = recommend(s, routes_by[s.shipment_id], explain=False, with_sensitivity=False)
        crit = {e.route.route_id: e.criteria for e in rec.evaluations}
        nd, _ = pareto_filter(crit)
        assert rec.chosen_route_id in nd, f"{s.shipment_id} picked a dominated route"
