"""Explanation, sensitivity, edge-case and integration tests."""

from __future__ import annotations

import re

import pytest

from routewise.config import profile_for, weights_for
from routewise.explain import comparison_table
from routewise.recommend import recommend
from routewise.schemas import Shipment
from routewise.sensitivity import (
    confidence_label,
    input_stability,
    switching_points,
    weight_stability,
    weight_sweep,
)
from tests.conftest import make_route, make_shipment
from tests.test_golden import menu


# ------------------------------------------------------------ Explanation ---
def test_explanation_names_the_chosen_route_and_runner_up():
    rec = recommend(make_shipment(deadline=60), menu(), explain=True, with_sensitivity=True)
    assert rec.chosen_route_id in rec.explanation
    runner = rec.scored[1].route.route_id
    assert runner in rec.explanation
    assert "Confidence:" in rec.explanation


def test_explanation_numbers_match_the_data():
    """Every cost figure printed must exist in the evaluation data."""
    rec = recommend(make_shipment(deadline=60), menu(), explain=True, with_sensitivity=False)
    printed = {int(m.replace(",", "")) for m in re.findall(r"\$([0-9,]+)", rec.explanation)}
    real_costs = {round(e.criteria["cost"]) for e in rec.evaluations}
    real_deltas = {
        abs(round(a.criteria["cost"] - b.criteria["cost"])) for a in rec.evaluations for b in rec.evaluations
    }
    for p in printed:
        assert p in real_costs or p in real_deltas, f"${p} is not traceable to the data"


def test_explanation_states_the_tradeoff_accepted():
    s = make_shipment(priority="standard", deadline=60)
    rec = recommend(s, menu(), explain=True, with_sensitivity=False)
    assert "vs cheapest" in rec.explanation and "vs fastest" in rec.explanation


def test_explanation_lists_eliminated_routes_with_reasons():
    s = make_shipment(priority="budget", deadline=31, hard=True)
    rec = recommend(s, menu(), explain=True, with_sensitivity=False)
    assert "Eliminated options:" in rec.explanation
    assert "CHEAP" in rec.explanation
    assert "deadline" in rec.explanation


def test_explanation_flags_high_geopolitical_exposure():
    routes = [
        make_route("A", cost=10_000, transit=20, geo_risk=0.85, uses_chokepoint=True),
        make_route("B", cost=40_000, transit=40, geo_risk=0.90, uses_chokepoint=True),
    ]
    rec = recommend(make_shipment(deadline=99, hard=False), routes, explain=True, with_sensitivity=False)
    assert any("geopolitical" in w.lower() for w in rec.warnings)


def test_comparison_table_covers_every_route():
    rec = recommend(make_shipment(deadline=60), menu(), explain=False, with_sensitivity=False)
    rows = comparison_table(rec)
    assert len(rows) == len(rec.evaluations)
    assert sum(r["chosen"] for r in rows) == 1
    assert all(r["status"] for r in rows)


# ------------------------------------------------------------ Sensitivity ---
def test_stability_is_a_probability():
    rec = recommend(make_shipment(deadline=60), menu(), explain=False, with_sensitivity=False)
    assert 0.0 <= weight_stability(rec) <= 1.0
    assert 0.0 <= input_stability(rec) <= 1.0


def test_zero_noise_gives_full_stability(cfg):
    import copy

    c = copy.deepcopy(cfg)
    c["sensitivity"]["weight_noise"] = 0.0
    rec = recommend(make_shipment(deadline=60), menu(), cfg=cfg, explain=False, with_sensitivity=False)
    assert weight_stability(rec, c) == 1.0


def test_single_option_is_perfectly_stable():
    one = [make_route("ONLY", cost=10_000, transit=20)]
    rec = recommend(make_shipment(deadline=99, hard=False), one, explain=False, with_sensitivity=True)
    assert rec.chosen_route_id == "ONLY"
    assert rec.stability == 1.0


def test_near_tie_gets_low_or_medium_confidence():
    """Two nearly identical routes must not be sold as a confident call."""
    routes = [
        make_route(
            "A", cost=20_000, transit=20.0, delay_prob=0.20, delay_mean=2.0, congestion=0.40, geo_risk=0.30
        ),
        make_route(
            "B", cost=20_400, transit=19.6, delay_prob=0.205, delay_mean=2.0, congestion=0.41, geo_risk=0.31
        ),
    ]
    rec = recommend(make_shipment(deadline=99, hard=False), routes, explain=True, with_sensitivity=True)
    assert rec.confidence in ("Low", "Medium")
    assert any("near-tie" in w.lower() or "close call" in w.lower() for w in rec.warnings)


def test_confidence_label_thresholds(cfg):
    assert confidence_label(0.99, 0.20, cfg) == "High"
    assert confidence_label(0.99, 0.01, cfg) == "Medium"  # stable but tiny margin
    assert confidence_label(0.80, 0.20, cfg) == "Medium"
    assert confidence_label(0.50, 0.20, cfg) == "Low"


def test_weight_sweep_covers_the_full_range_and_finds_switches():
    s = make_shipment(priority="standard", deadline=99, hard=False)
    sweep = weight_sweep(s, menu(), "cost", steps=21)
    assert len(sweep) == 21
    assert sweep[0][0] == 0.0 and sweep[-1][0] == 1.0
    # at weight(cost)=1 the cheapest route must win
    assert sweep[-1][1] == "CHEAP"
    assert switching_points(sweep), "expected at least one switching point"


def test_switching_points_are_empty_for_a_dominant_route():
    routes = [
        make_route(
            "BEST", cost=10_000, transit=10, delay_prob=0.05, delay_mean=1.0, congestion=0.1, geo_risk=0.1
        ),
        make_route(
            "WORSE", cost=20_000, transit=20, delay_prob=0.40, delay_mean=4.0, congestion=0.8, geo_risk=0.8
        ),
    ]
    s = make_shipment(deadline=99, hard=False)
    assert switching_points(weight_sweep(s, routes, "cost", steps=21)) == []


# ------------------------------------------------------------- Edge cases ---
def test_one_route_only():
    rec = recommend(
        make_shipment(deadline=99, hard=False), [make_route("ONLY")], explain=True, with_sensitivity=True
    )
    assert rec.chosen_route_id == "ONLY" and rec.margin is None
    assert "ONLY" in rec.explanation


def test_all_routes_identical():
    routes = [make_route(f"R{i}") for i in range(3)]
    rec = recommend(make_shipment(deadline=99, hard=False), routes, explain=True, with_sensitivity=False)
    scores = [e.score for e in rec.evaluations if e.score is not None]
    assert len(set(scores)) == 1 and rec.chosen_route_id == "R0"  # deterministic tie-break


def test_no_routes_raises_a_clear_error():
    with pytest.raises(ValueError, match="no route options"):
        recommend(make_shipment(), [], explain=False, with_sensitivity=False)


def test_invalid_priority_is_rejected_at_the_schema():
    with pytest.raises(Exception, match="priority"):
        Shipment(
            shipment_id="S1",
            origin="MUM",
            destination="RTM",
            cargo_type="general",
            priority="super_urgent",
            volume_teu=1,
            value_usd=1000,
            deadline_days=10,
            hard_deadline=True,
        )


def test_unknown_profile_raises():
    with pytest.raises(KeyError, match="unknown profile"):
        weights_for("does_not_exist")


def test_critical_cargo_overrides_priority():
    for cargo in ("perishable", "fragile", "high_value"):
        assert profile_for("budget", cargo) == "critical"
    assert profile_for("budget", "general") == "budget"


def test_all_config_profiles_have_weights_summing_to_one(cfg):
    for name, w in cfg["profiles"].items():
        assert abs(sum(w.values()) - 1.0) < 1e-9, name


# ------------------------------------------------------------- Integration --
def test_recommend_runs_on_every_shipment_without_exceptions(dataset):
    shipments, routes_by = dataset
    for s in shipments:
        rec = recommend(s, routes_by[s.shipment_id], explain=True, with_sensitivity=True)
        assert rec.chosen_route_id
        assert rec.explanation and len(rec.explanation) > 100
        assert rec.confidence in ("High", "Medium", "Low")
        assert 0.0 <= rec.stability <= 1.0


def test_cli_recommend_exits_zero():
    from typer.testing import CliRunner

    from routewise.cli import app

    res = CliRunner().invoke(app, ["recommend", "--shipment", "S001"])
    assert res.exit_code == 0, res.output


def test_cli_validate_exits_zero():
    from typer.testing import CliRunner

    from routewise.cli import app

    res = CliRunner().invoke(app, ["validate"])
    assert res.exit_code == 0, res.output


def test_recommendation_under_one_second(dataset):
    """NFR: one shipment in under 1s including 1000 Monte Carlo runs."""
    import time

    shipments, routes_by = dataset
    s = shipments[0]
    t0 = time.perf_counter()
    recommend(s, routes_by[s.shipment_id], explain=True, with_sensitivity=False)
    assert time.perf_counter() - t0 < 1.0


def test_pipeline_is_deterministic(dataset):
    shipments, routes_by = dataset
    s = shipments[3]
    a = recommend(s, routes_by[s.shipment_id], explain=True, with_sensitivity=True)
    b = recommend(s, routes_by[s.shipment_id], explain=True, with_sensitivity=True)
    assert a.chosen_route_id == b.chosen_route_id
    assert a.explanation == b.explanation
    assert a.stability == b.stability


def test_injected_near_tie_shipments_never_get_high_confidence(dataset):
    """The near-tie shipments get Low or Medium confidence."""
    shipments, routes_by = dataset
    near = [s for s in shipments if s.note == "near_tie"]
    assert len(near) >= 3
    for s in near:
        rec = recommend(s, routes_by[s.shipment_id], explain=False, with_sensitivity=True)
        assert rec.confidence in ("Low", "Medium"), s.shipment_id
