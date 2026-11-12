"""RouteWise CLI (FR-11)."""

from __future__ import annotations

import json

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from .config import load_config
from .generate_data import generate, load_dataset
from .recommend import recommend as _recommend
from .validate import validate_dataset

app = typer.Typer(
    add_completion=False, help="RouteWise: shipment route recommendation under cost, time and risk."
)
console = Console()


@app.command("generate-data")
def generate_data(seed: int = typer.Option(42, help="Random seed (42 reproduces the shipped dataset).")):
    """Generate the synthetic dataset and the data card."""
    out = generate(seed=seed)
    t = Table(title=f"Dataset generated (seed {seed})")
    t.add_column("property")
    t.add_column("value", justify="right")
    for k, v in out["stats"].items():
        t.add_row(k, str(v))
    console.print(t)


@app.command()
def validate():
    """Validate the generated dataset."""
    shipments, routes = load_dataset()
    rep = validate_dataset(shipments, routes)
    console.print(str(rep))
    raise typer.Exit(0 if rep.ok else 1)


@app.command()
def recommend(
    shipment: str = typer.Option(..., "--shipment", "-s", help="Shipment id, e.g. S012"),
    profile: str | None = typer.Option(None, "--profile", "-p", help="Override the weight profile."),
    explain: bool = typer.Option(True, "--explain/--no-explain"),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
):
    """Recommend a route for one shipment."""
    cfg = load_config()
    shipments, routes_by = load_dataset()
    try:
        s = next(x for x in shipments if x.shipment_id == shipment)
    except StopIteration:
        console.print(f"[red]unknown shipment {shipment}[/red]")
        raise typer.Exit(2) from None

    rec = _recommend(s, routes_by[shipment], profile=profile, cfg=cfg, explain=explain)

    if as_json:
        from .explain import comparison_table

        print(
            json.dumps(
                {
                    "shipment_id": s.shipment_id,
                    "profile": rec.profile,
                    "weights": rec.weights,
                    "chosen": rec.chosen_route_id,
                    "compliant": rec.compliant,
                    "confidence": rec.confidence,
                    "stability": rec.stability,
                    "margin": rec.margin,
                    "warnings": rec.warnings,
                    "routes": comparison_table(rec),
                    "explanation": rec.explanation,
                },
                indent=2,
                default=str,
            )
        )
        return

    from .explain import comparison_table

    t = Table(
        title=f"{shipment}: {s.origin}->{s.destination} | {s.priority.value} / {s.cargo_type.value} | profile {rec.profile}"
    )
    for col in ("route", "path", "cost", "E[transit]", "P90", "P(miss)", "risk", "score", "status"):
        t.add_column(col, overflow="fold")
    for row in comparison_table(rec):
        style = "bold green" if row["chosen"] else ("dim" if row["score"] is None else "")
        t.add_row(
            row["route_id"],
            row["path"],
            f"${row['cost_usd']:,.0f}",
            f"{row['exp_transit_d']:.1f}d",
            f"{row['p90_d']:.1f}d",
            f"{row['p_miss_deadline']:.0%}",
            f"{row['risk']:.2f}",
            f"{row['score']:.3f}" if row["score"] is not None else "-",
            row["status"],
            style=style,
        )
    console.print(t)
    if explain:
        console.print()
        console.print(rec.explanation)


@app.command()
def evaluate(out: str = typer.Option("reports", help="Output directory.")):
    """Run the full evaluation (E1-E5) and write tables and figures."""
    from pathlib import Path

    from .evaluate import run_full_evaluation
    from .plots import pareto_plot, strategy_comparison_plot, transit_distribution_plot

    outp = Path(out)
    console.print("[cyan]Running evaluation across all shipments and 7 strategies...[/cyan]")
    res = run_full_evaluation(out_dir=outp)

    cols = [
        "strategy",
        "mean_cost_usd",
        "mean_exp_transit_d",
        "mean_deadline_miss_rate",
        "worst_case_regret_feasible",
        "expected_landed_cost",
        "constraint_violation_rate",
    ]
    pd.set_option("display.width", 250)
    console.print("\n[bold]E1 Strategy comparison[/bold]")
    console.print(res["summary"][cols].sort_values("worst_case_regret_feasible").to_string(index=False))
    console.print("\n[bold]E2 Profile behaviour[/bold]")
    console.print(res["profile_behaviour"].to_string(index=False))
    console.print("\n[bold]E3 Noisy estimates[/bold]")
    console.print(json.dumps(res["noisy"], indent=2))
    console.print("\n[bold]E4 Sensitivity[/bold]")
    console.print(json.dumps(res["sensitivity"], indent=2))

    console.print("\n[cyan]Writing figures...[/cyan]")
    strategy_comparison_plot(res["summary"], outp / "strategy_comparison.png")
    ps = res["per_shipment"]
    picks = [ps.iloc[0]["shipment_id"]]
    for note in ("trap", "near_tie", "infeasible"):
        m = ps.loc[ps["note"] == note, "shipment_id"]
        if len(m):
            picks.append(m.iloc[0])
    for sid in dict.fromkeys(picks):
        pareto_plot(sid, outp / f"pareto_{sid}.png")
        transit_distribution_plot(sid, outp / f"transit_{sid}.png")
    console.print(f"[green]Done. Tables and figures in {outp}/[/green]")


@app.command()
def sensitivity(
    shipment: str = typer.Option(..., "--shipment", "-s"),
    criterion: str = typer.Option("cost", "--criterion", "-c", help="cost|time|reliability|risk"),
):
    """Weight sweep and switching points for one shipment."""
    from .plots import sensitivity_plot
    from .sensitivity import input_stability, switching_points, weight_sweep

    cfg = load_config()
    shipments, routes_by = load_dataset()
    s = next(x for x in shipments if x.shipment_id == shipment)
    rec = _recommend(s, routes_by[shipment], cfg=cfg, explain=False)

    console.print(
        f"Baseline winner: [bold]{rec.chosen_route_id}[/bold]  "
        f"confidence {rec.confidence}  weight-stability {rec.stability:.0%}  "
        f"input-stability {input_stability(rec, cfg):.0%}"
    )

    sweep = weight_sweep(s, routes_by[shipment], criterion, cfg)
    pts = switching_points(sweep)
    if pts:
        for w, a, b in pts:
            console.print(f"  switch at weight({criterion}) = {w:.2f}: {a} -> {b}")
    else:
        console.print(f"  no switch: {sweep[0][1]} wins across the whole range of weight({criterion}).")
    p = sensitivity_plot(shipment, criterion)
    console.print(f"[green]figure: {p}[/green]")


@app.command("recommend-all")
def recommend_all_cmd(out: str = typer.Option("reports/recommendations.csv")):
    """Run the recommender over every shipment (integration smoke test)."""
    from pathlib import Path

    cfg = load_config()
    shipments, routes_by = load_dataset()
    rows = []
    for s in shipments:
        rec = _recommend(s, routes_by[s.shipment_id], cfg=cfg, explain=True)
        rows.append(
            {
                "shipment_id": s.shipment_id,
                "profile": rec.profile,
                "chosen": rec.chosen_route_id,
                "compliant": rec.compliant,
                "confidence": rec.confidence,
                "stability": rec.stability,
                "margin": rec.margin,
                "warnings": " | ".join(rec.warnings),
                "explanation": rec.explanation,
            }
        )
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p, index=False)
    console.print(f"[green]{len(rows)} recommendations written to {p}[/green]")


if __name__ == "__main__":  # pragma: no cover
    app()
