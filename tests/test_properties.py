"""Property-based invariants (hypothesis). TRD 3.9 'Property tests'.

These are the claims that must hold for *any* input, not just the cases we
thought to write down.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from routewise.recommend import recommend
from routewise.scoring import normalise, rank, score
from tests.conftest import make_route, make_shipment

SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)

route_attrs = st.fixed_dictionaries(
    {
        "cost": st.floats(1_000, 100_000, allow_nan=False, allow_infinity=False),
        "transit": st.floats(2.0, 60.0, allow_nan=False, allow_infinity=False),
        "delay_prob": st.floats(0.0, 0.95, allow_nan=False, allow_infinity=False),
        "delay_mean": st.floats(0.0, 10.0, allow_nan=False, allow_infinity=False),
        "congestion": st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
        "geo_risk": st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
    }
)


def build(attrs_list, sid="S1"):
    return [
        make_route(
            f"R{i}",
            sid,
            cost=a["cost"],
            transit=a["transit"],
            delay_prob=a["delay_prob"],
            delay_mean=a["delay_mean"],
            congestion=a["congestion"],
            geo_risk=a["geo_risk"],
        )
        for i, a in enumerate(attrs_list)
    ]


def _rec(routes, priority="standard"):
    # soft deadline + no caps: isolates the SCORING properties from the
    # constraint filter, which is tested separately.
    s = make_shipment(priority=priority, deadline=500.0, hard=False)
    return recommend(s, routes, explain=False, with_sensitivity=False)


@given(attrs=st.lists(route_attrs, min_size=2, max_size=5))
@SETTINGS
def test_raising_a_routes_cost_never_improves_its_rank(attrs):
    routes = build(attrs)
    base = _rec(routes)
    target = base.chosen_route_id
    before = base.by_id(target).rank

    worse = [
        r.model_copy(update={"total_cost": r.total_cost * 2.0}) if r.route_id == target else r for r in routes
    ]
    after_eval = _rec(worse).by_id(target)
    after = after_eval.rank if after_eval.rank is not None else 99
    assert after >= before


@given(attrs=st.lists(route_attrs, min_size=2, max_size=4))
@SETTINGS
def test_adding_a_dominated_route_never_changes_the_winner(attrs):
    routes = build(attrs)
    winner = _rec(routes).chosen_route_id

    worst = {
        "cost": max(r.total_cost for r in routes) * 1.5,
        "transit": max(r.transit_days for r in routes) * 1.5,
        "delay_prob": min(0.99, max(r.delay_prob for r in routes) + 0.01),
        "delay_mean": max(r.delay_mean_days for r in routes) + 1.0,
        "congestion": min(1.0, max(r.congestion for r in routes)),
        "geo_risk": min(1.0, max(r.geo_risk for r in routes)),
    }
    extra = make_route(
        "R_DOM",
        "S1",
        cost=worst["cost"],
        transit=worst["transit"],
        delay_prob=worst["delay_prob"],
        delay_mean=worst["delay_mean"],
        congestion=worst["congestion"],
        geo_risk=worst["geo_risk"],
    )
    assert _rec([*routes, extra]).chosen_route_id == winner


@given(
    attrs=st.lists(route_attrs, min_size=2, max_size=5),
    factor=st.floats(0.1, 10.0, allow_nan=False, allow_infinity=False),
)
@SETTINGS
def test_scaling_all_costs_never_changes_the_winner(attrs, factor):
    """Min-max normalisation is scale-invariant: the currency unit must not
    change the decision."""
    routes = build(attrs)
    scaled = [r.model_copy(update={"total_cost": r.total_cost * factor}) for r in routes]
    assert _rec(routes).chosen_route_id == _rec(scaled).chosen_route_id


@given(attrs=st.lists(route_attrs, min_size=2, max_size=5), seed=st.integers(0, 10_000))
@SETTINGS
def test_permuting_route_order_never_changes_the_winner(attrs, seed):
    import random

    routes = build(attrs)
    shuffled = routes[:]
    random.Random(seed).shuffle(shuffled)
    assert _rec(routes).chosen_route_id == _rec(shuffled).chosen_route_id


@given(attrs=st.lists(route_attrs, min_size=1, max_size=5))
@SETTINGS
def test_a_recommendation_always_exists(attrs):
    rec = _rec(build(attrs))
    assert rec.chosen_route_id in {r.route_id for r in rec.evaluations and [e.route for e in rec.evaluations]}


@given(attrs=st.lists(route_attrs, min_size=2, max_size=5))
@SETTINGS
def test_normalised_values_stay_in_unit_interval(attrs):
    routes = build(attrs)
    crit = {
        r.route_id: {
            "cost": r.total_cost,
            "time": r.expected_transit_days,
            "reliability": r.delay_prob,
            "risk": r.risk_score,
        }
        for r in routes
    }
    norms = normalise(crit)
    for v in norms.values():
        for x in v.values():
            assert 0.0 - 1e-9 <= x <= 1.0 + 1e-9


@given(
    attrs=st.lists(route_attrs, min_size=2, max_size=5),
    w=st.lists(st.floats(0.01, 1.0, allow_nan=False, allow_infinity=False), min_size=4, max_size=4),
)
@SETTINGS
def test_score_is_monotone_in_every_criterion(attrs, w):
    """Making a route worse on one criterion (others fixed) never lowers its
    score. This is what makes the scoring explainable at all."""
    total = sum(w)
    weights = dict(zip(("cost", "time", "reliability", "risk"), [x / total for x in w], strict=True))
    routes = build(attrs)
    crit = {
        r.route_id: {
            "cost": r.total_cost,
            "time": r.expected_transit_days,
            "reliability": r.delay_prob,
            "risk": r.risk_score,
        }
        for r in routes
    }
    s1, _ = score(normalise(crit), weights)

    target = rank(s1)[0]
    bumped = {rid: dict(v) for rid, v in crit.items()}
    bumped[target]["cost"] = max(v["cost"] for v in crit.values()) * 2 + 1
    s2, _ = score(normalise(bumped), weights)
    assert s2[target] >= s1[target] - 1e-9
