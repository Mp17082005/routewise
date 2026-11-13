"""Data tests and data-realism tests."""

from __future__ import annotations

import pandas as pd
import pytest

from routewise.config import DATA_DIR
from routewise.generate_data import generate
from routewise.validate import validate_dataset


@pytest.fixture(scope="module")
def gen(tmp_path_factory):
    out = tmp_path_factory.mktemp("data42")
    return generate(seed=42, out_dir=out)["stats"], out


# ------------------------------------------------------------------ Data ----
def test_dataset_validates(dataset):
    shipments, routes = dataset
    rep = validate_dataset(shipments, routes)
    assert rep.ok, str(rep)


def test_shipment_and_route_counts(dataset):
    shipments, routes = dataset
    assert len(shipments) == 40
    for s in shipments:
        assert 3 <= len(routes[s.shipment_id]) <= 5, s.shipment_id


def test_no_nulls_in_required_columns():
    for name in ("shipments", "routes", "hubs", "legs"):
        df = pd.read_csv(DATA_DIR / f"{name}.csv")
        optional = {"budget_cap", "max_geo_risk", "note", "lane_id"}
        required = [c for c in df.columns if c not in optional]
        assert not df[required].isna().any().any(), f"{name}.csv has nulls"


def test_positive_costs_and_times_and_valid_probabilities(dataset):
    _, routes = dataset
    for rs in routes.values():
        for r in rs:
            assert r.total_cost > 0 and r.transit_days > 0
            assert 0.0 <= r.delay_prob <= 1.0
            assert 0.0 <= r.congestion <= 1.0
            assert 0.0 <= r.geo_risk <= 1.0
            assert r.delay_mean_days >= 0


def test_seed_is_reproducible(tmp_path):
    a = generate(seed=42, out_dir=tmp_path / "a")
    b = generate(seed=42, out_dir=tmp_path / "b")
    for name in ("shipments", "routes", "hubs", "legs"):
        fa = (tmp_path / "a" / f"{name}.csv").read_bytes()
        fb = (tmp_path / "b" / f"{name}.csv").read_bytes()
        assert fa == fb, f"{name}.csv differs between identical seeds"
    assert a["stats"] == b["stats"]


def test_different_seeds_differ(tmp_path):
    a = generate(seed=42, out_dir=tmp_path / "a")
    b = generate(seed=7, out_dir=tmp_path / "b")
    assert a["stats"] != b["stats"] or not (
        (tmp_path / "a" / "routes.csv").read_bytes() == (tmp_path / "b" / "routes.csv").read_bytes()
    )


def test_shipped_data_matches_seed_42(gen):
    """The committed CSVs are exactly what seed 42 produces."""
    _, out = gen
    for name in ("shipments", "routes", "hubs", "legs"):
        assert (out / f"{name}.csv").read_bytes() == (DATA_DIR / f"{name}.csv").read_bytes(), (
            f"{name}.csv in data/ is stale — re-run `routewise generate-data --seed 42`"
        )


# --------------------------------------------------------- Data realism -----
def test_cost_time_tradeoff_is_negative_within_option_set(gen):
    """The trade-off must exist where the decision happens: within a shipment's
    own option set. Design target is -0.4 to -0.6; we assert a
    clearly negative correlation and warn-by-bound on the upper side."""
    stats, _ = gen
    corr = stats["cost_time_corr_within_lane"]
    assert corr < -0.25, f"cost-vs-time correlation {corr} is not a real trade-off"
    assert corr > -0.95, f"cost-vs-time correlation {corr} is implausibly deterministic"


def test_at_least_10pct_routes_dominated(gen):
    stats, _ = gen
    assert stats["dominated_share"] >= 0.10, stats["dominated_share"]


def test_injected_cases_present(gen):
    stats, _ = gen
    assert stats["trap_cases"] >= 3
    assert stats["near_tie_cases"] >= 3
    assert stats["infeasible_cases"] == 2


def test_deadlines_are_meaningful(dataset):
    """Every deadline (apart from the deliberately infeasible cases) is both
    achievable by at least one route and tight relative to its anchor route,
    so it actually constrains the decision."""
    from routewise.generate_data import route_p90

    shipments, routes = dataset
    for s in shipments:
        if s.note == "infeasible":
            continue
        rs = routes[s.shipment_id]
        p90s = [route_p90(r) for r in rs]
        assert min(p90s) <= s.deadline_days, f"{s.shipment_id}: no route can make it"
        surface = [route_p90(r) for r in rs if "air" not in r.mode_mix] or p90s
        anchor = min(p90s) if s.priority.value == "urgent" else min(surface)
        assert s.deadline_days <= anchor * 1.7, f"{s.shipment_id}: deadline is not binding"


def test_route_endpoints_match_shipment(dataset):
    shipments, routes = dataset
    for s in shipments:
        for r in routes[s.shipment_id]:
            assert r.hub_path[0] == s.origin
            assert r.hub_path[-1] == s.destination


def test_chokepoint_routes_carry_high_geo_risk(dataset):
    _, routes = dataset
    choke = [r for rs in routes.values() for r in rs if r.uses_chokepoint]
    assert choke, "no chokepoint routes generated"
    assert min(r.geo_risk for r in choke) >= 0.6


def test_data_card_exists_and_reports_real_numbers(gen):
    stats, out = gen
    card = (out / "data_card.md").read_text(encoding="utf-8")
    assert str(stats["n_shipments"]) in card
    assert str(stats["cost_time_corr_within_lane"]) in card
    assert "geo_waiver" in card or "waiver" in card
