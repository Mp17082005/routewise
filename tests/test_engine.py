"""Constraints, Pareto, scoring and simulation tests."""

from __future__ import annotations

import pytest

from routewise.constraints import check_route
from routewise.pareto import dominates, pareto_filter
from routewise.scoring import normalise, rank, score
from routewise.simulate import simulate_route
from tests.conftest import make_route, make_shipment


def sim_for(route, deadline=30.0, **kw):
    return simulate_route(route, deadline, **kw)


# ----------------------------------------------------------- Constraints ----
def test_route_over_budget_is_removed():
    s = make_shipment(budget=9_000)
    r = make_route(cost=10_000)
    ok, viol, mag = check_route(s, r, sim_for(r))
    assert not ok and any("budget" in v for v in viol) and mag > 0


def test_route_failing_p90_deadline_is_removed():
    s = make_shipment(deadline=10.0, hard=True)
    r = make_route(transit=20.0)
    ok, viol, _ = check_route(s, r, sim_for(r, 10.0))
    assert not ok and any("deadline" in v for v in viol)


def test_soft_deadline_does_not_block():
    s = make_shipment(deadline=10.0, hard=False)
    r = make_route(transit=20.0)
    ok, viol, _ = check_route(s, r, sim_for(r, 10.0))
    assert ok and not viol


def test_geo_risk_cap_blocks_chokepoint_route():
    s = make_shipment(cargo_type="fragile", max_geo=0.5, deadline=99)
    r = make_route(geo_risk=0.78)
    ok, viol, _ = check_route(s, r, sim_for(r, 99))
    assert not ok and any("geopolitical" in v for v in viol)


def test_compliant_route_passes_cleanly():
    s = make_shipment(deadline=99, budget=50_000, max_geo=0.9)
    r = make_route()
    ok, viol, mag = check_route(s, r, sim_for(r, 99))
    assert ok and viol == [] and mag == 0.0


def test_violation_magnitude_grows_with_severity():
    s = make_shipment(budget=10_000, deadline=99)
    mild = make_route(cost=11_000)
    bad = make_route(cost=20_000)
    _, _, m1 = check_route(s, mild, sim_for(mild, 99))
    _, _, m2 = check_route(s, bad, sim_for(bad, 99))
    assert m2 > m1


# ---------------------------------------------------------------- Pareto ----
def c(cost, time, rel, risk):
    return {"cost": cost, "time": time, "reliability": rel, "risk": risk}


def test_dominated_route_is_removed():
    crit = {"A": c(100, 10, 0.1, 0.2), "B": c(120, 12, 0.2, 0.3)}
    nd, dom = pareto_filter(crit)
    assert nd == ["A"] and dom == {"B": "A"}


def test_ties_are_not_both_removed():
    crit = {"A": c(100, 10, 0.1, 0.2), "B": c(100, 10, 0.1, 0.2)}
    nd, dom = pareto_filter(crit)
    assert sorted(nd) == ["A", "B"] and dom == {}


def test_single_route_returns_itself():
    crit = {"A": c(100, 10, 0.1, 0.2)}
    nd, dom = pareto_filter(crit)
    assert nd == ["A"] and dom == {}


def test_tradeoff_routes_both_survive():
    crit = {"cheap_slow": c(100, 30, 0.4, 0.2), "dear_fast": c(300, 10, 0.1, 0.2)}
    nd, _ = pareto_filter(crit)
    assert sorted(nd) == ["cheap_slow", "dear_fast"]


def test_dominance_requires_strict_improvement():
    a = c(100, 10, 0.1, 0.2)
    assert not dominates(a, dict(a))


# --------------------------------------------------------------- Scoring ----
def test_weights_must_sum_to_one():
    norms = {"A": c(0.0, 0.0, 0.0, 0.0)}
    with pytest.raises(ValueError, match="sum to 1.0"):
        score(norms, {"cost": 0.5, "time": 0.2, "reliability": 0.2, "risk": 0.2})


def test_scores_are_within_zero_and_one():
    crit = {"A": c(100, 10, 0.1, 0.1), "B": c(200, 20, 0.5, 0.6), "C": c(150, 15, 0.3, 0.3)}
    norms = normalise(crit)
    s, _ = score(norms, {"cost": 0.25, "time": 0.25, "reliability": 0.25, "risk": 0.25})
    assert all(0.0 - 1e-9 <= v <= 1.0 + 1e-9 for v in s.values())


def test_identical_routes_get_equal_scores():
    crit = {"A": c(100, 10, 0.2, 0.3), "B": c(100, 10, 0.2, 0.3)}
    norms = normalise(crit)
    s, _ = score(norms, {"cost": 0.4, "time": 0.3, "reliability": 0.2, "risk": 0.1})
    assert s["A"] == s["B"]


def test_two_percent_spread_rule_flattens_tiny_differences():
    # cost differs by 1% -> treated as flat; time differs materially
    crit = {"A": c(100.0, 10, 0.2, 0.3), "B": c(101.0, 20, 0.2, 0.3)}
    norms = normalise(crit, flat_spread_ratio=0.02)
    assert norms["A"]["cost"] == 0.0 and norms["B"]["cost"] == 0.0
    assert norms["A"]["time"] == 0.0 and norms["B"]["time"] == 1.0


def test_spread_above_threshold_is_not_flattened():
    crit = {"A": c(100.0, 10, 0.2, 0.3), "B": c(110.0, 10, 0.2, 0.3)}
    norms = normalise(crit, flat_spread_ratio=0.02)
    assert norms["B"]["cost"] == 1.0


def test_rank_is_deterministic_under_tie():
    assert rank({"B": 0.5, "A": 0.5, "C": 0.1}) == ["C", "A", "B"]


# ------------------------------------------------------------ Simulation ----
def test_same_seed_gives_same_result():
    r = make_route()
    assert simulate_route(r, 30.0, seed=42) == simulate_route(r, 30.0, seed=42)


def test_different_seeds_give_different_results():
    r = make_route()
    assert simulate_route(r, 30.0, seed=1) != simulate_route(r, 30.0, seed=2)


def test_p50_le_p90():
    for dp in (0.0, 0.3, 0.7, 1.0):
        r = make_route(delay_prob=dp)
        s = simulate_route(r, 30.0)
        assert s.p50_transit <= s.p90_transit


def test_zero_delay_probability_gives_zero_variance():
    r = make_route(transit=15.0, delay_prob=0.0, delay_mean=5.0)
    s = simulate_route(r, 30.0)
    assert s.p50_transit == pytest.approx(15.0)
    assert s.p90_transit == pytest.approx(15.0)
    assert s.mean_transit == pytest.approx(15.0)


def test_certain_delay_always_delays():
    r = make_route(transit=15.0, delay_prob=1.0, delay_mean=4.0)
    s = simulate_route(r, 30.0)
    assert s.p50_transit > 15.0
    assert s.mean_transit == pytest.approx(19.0, rel=0.08)  # 15 + mean delay 4


def test_deadline_miss_probability_bounds():
    r = make_route(transit=15.0, delay_prob=0.5, delay_mean=3.0)
    assert simulate_route(r, 1.0).p_miss_deadline == 1.0
    assert simulate_route(r, 1000.0).p_miss_deadline == 0.0


def test_expected_days_late_is_zero_when_never_late():
    r = make_route(transit=10.0, delay_prob=0.2, delay_mean=1.0)
    assert simulate_route(r, 500.0).expected_days_late == 0.0
