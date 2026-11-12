"""Evaluation harness and PRD success criteria (TRD 3.8, PRD 1.8).

The evaluation is run once per session on the real dataset; the assertions
below are the PRD success criteria. They are checked, not assumed.
"""

from __future__ import annotations

import json

import pytest

from routewise.evaluate import (
    noisy_estimate_experiment,
    profile_behaviour,
    run_full_evaluation,
)


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    out = tmp_path_factory.mktemp("reports")
    return run_full_evaluation(out_dir=out), out


def _row(summary, strat):
    return summary.set_index("strategy").loc[strat]


def test_all_strategies_evaluated(results):
    res, _ = results
    assert set(res["summary"]["strategy"]) == {
        "B1_cheapest",
        "B2_fastest",
        "B3_lowest_risk",
        "B4_equal_weights",
        "B5_random",
        "A1_routewise",
        "A2_no_filters",
    }


def test_success_routewise_picks_are_always_pareto_optimal(results):
    res, _ = results
    assert _row(res["summary"], "A1_routewise")["share_pareto_optimal"] == 1.0


def test_success_lowest_worst_case_regret_vs_single_metric_baselines(results):
    res, _ = results
    sm = res["summary"]
    rw = _row(sm, "A1_routewise")["worst_case_regret_feasible"]
    for b in ("B1_cheapest", "B2_fastest", "B3_lowest_risk"):
        assert rw < _row(sm, b)["worst_case_regret_feasible"], b


def test_success_deadline_miss_rate_below_cheapest(results):
    res, _ = results
    sm = res["summary"]
    assert (
        _row(sm, "A1_routewise")["mean_deadline_miss_rate"]
        < _row(sm, "B1_cheapest")["mean_deadline_miss_rate"]
    )


def test_success_cost_premium_is_reported_honestly(results):
    """RouteWise must NOT claim to be cheapest: its premium is reported."""
    res, _ = results
    sm = res["summary"]
    assert _row(sm, "A1_routewise")["mean_cost_usd"] > _row(sm, "B1_cheapest")["mean_cost_usd"]


def test_constraints_reduce_violations_vs_ablation(results):
    res, _ = results
    sm = res["summary"]
    assert (
        _row(sm, "A1_routewise")["constraint_violation_rate"]
        < _row(sm, "A2_no_filters")["constraint_violation_rate"]
    )
    # the only remaining violations are the 2 deliberately infeasible shipments
    assert _row(sm, "A1_routewise")["constraint_violation_rate"] == pytest.approx(2 / 40)


def test_success_flip_rate_reported_and_low_confidence_flagged(results):
    res, out = results
    sens = json.loads((out / "e4_sensitivity_summary.json").read_text())
    assert 0.0 <= sens["flip_rate_under_weight_noise"] <= 1.0
    assert isinstance(sens["low_confidence_shipments"], list)
    assert sorted(sens["non_compliant_shipments"]) == ["S010", "S030"]


def test_success_urgent_faster_than_budget_and_budget_cheaper():
    df = profile_behaviour().set_index("profile")
    assert (
        df.loc["urgent", "exp_transit"] < df.loc["standard", "exp_transit"] < df.loc["budget", "exp_transit"]
    )
    assert df.loc["urgent", "cost"] > df.loc["budget", "cost"]


def test_noisy_estimate_experiment_is_bounded():
    r = noisy_estimate_experiment()
    assert 0.0 <= r["decision_flip_rate"] <= 1.0
    assert r["mean_elc_regret_vs_oracle_usd"] >= 0
    assert 0.0 <= r["share_matching_oracle"] <= 1.0


def test_outputs_written(results):
    _, out = results
    for f in (
        "e1_summary.csv",
        "e1_raw.csv",
        "per_shipment.csv",
        "e2_profile_behaviour.csv",
        "e3_noisy_estimates.json",
        "e4_sensitivity_summary.json",
    ):
        assert (out / f).exists(), f


def test_figures_render(results, tmp_path):
    from routewise.plots import (
        pareto_plot,
        sensitivity_plot,
        strategy_comparison_plot,
        transit_distribution_plot,
    )

    res, _ = results
    for p in (
        pareto_plot("S004", tmp_path / "p.png"),
        sensitivity_plot("S004", "cost", tmp_path / "s.png"),
        strategy_comparison_plot(res["summary"], tmp_path / "c.png"),
        transit_distribution_plot("S004", tmp_path / "t.png"),
    ):
        assert p.exists() and p.stat().st_size > 5_000


def test_cli_sensitivity_and_recommend_json(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from routewise.cli import app

    monkeypatch.chdir(tmp_path)
    r = CliRunner()
    res = r.invoke(app, ["recommend", "-s", "S004", "--json"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["chosen"] and payload["routes"]

    res = r.invoke(app, ["recommend", "-s", "NOPE"])
    assert res.exit_code == 2

    res = r.invoke(app, ["recommend", "-s", "S004", "--profile", "urgent"])
    assert res.exit_code == 0, res.output


def test_cli_recommend_all(tmp_path):
    from typer.testing import CliRunner

    from routewise.cli import app

    out = tmp_path / "recs.csv"
    res = CliRunner().invoke(app, ["recommend-all", "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert out.exists() and len(out.read_text().splitlines()) > 40
