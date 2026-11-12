"""Figures for the report (FR-10)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import REPORTS_DIR, load_config  # noqa: E402
from .generate_data import load_dataset  # noqa: E402
from .recommend import recommend  # noqa: E402
from .sensitivity import switching_points, weight_sweep  # noqa: E402
from .simulate import simulate_all  # noqa: E402


def pareto_plot(shipment_id: str, out: Path | None = None) -> Path:
    """Cost vs expected transit; bubble size = risk; colour = status."""
    cfg = load_config()
    shipments, routes_by = load_dataset()
    s = next(x for x in shipments if x.shipment_id == shipment_id)
    routes = routes_by[shipment_id]
    rec = recommend(s, routes, cfg=cfg, explain=False, with_sensitivity=False)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for e in rec.evaluations:
        x, y = e.criteria["cost"], e.criteria["time"]
        size = 120 + 900 * e.criteria["risk"]
        if e.route.route_id == rec.chosen_route_id:
            c, lbl = "#1b7f3b", "chosen"
        elif e.score is not None:
            c, lbl = "#2c6fbb", "scored (Pareto front)"
        elif e.violations:
            c, lbl = "#c23b22", "infeasible"
        else:
            c, lbl = "#999999", "dominated"
        ax.scatter(x, y, s=size, c=c, alpha=0.75, edgecolors="black", linewidths=0.8, label=lbl)
        ax.annotate(
            e.route.route_id.replace(f"R-{shipment_id}-", "R"),
            (x, y),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=9,
        )

    ax.axhline(s.deadline_days, ls="--", c="#c23b22", lw=1.2, label=f"deadline {s.deadline_days:.1f}d")
    ax.set_xlabel("Total cost (USD)")
    ax.set_ylabel("Expected transit (days)")
    ax.set_title(
        f"{shipment_id}: {s.origin}->{s.destination}, {s.priority.value} / "
        f"{s.cargo_type.value}\nbubble size = route risk"
    )
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles, strict=True))
    ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="best", markerscale=0.35)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out = out or REPORTS_DIR / f"pareto_{shipment_id}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def sensitivity_plot(shipment_id: str, criterion: str = "cost", out: Path | None = None) -> Path:
    """One-at-a-time weight sweep showing the switching points."""
    cfg = load_config()
    shipments, routes_by = load_dataset()
    s = next(x for x in shipments if x.shipment_id == shipment_id)
    sweep = weight_sweep(s, routes_by[shipment_id], criterion, cfg)
    pts = switching_points(sweep)

    winners = sorted({w for _, w in sweep})
    idx = {w: i for i, w in enumerate(winners)}
    xs = [w for w, _ in sweep]
    ys = [idx[w] for _, w in sweep]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.step(xs, ys, where="post", lw=2, c="#2c6fbb")
    ax.set_yticks(range(len(winners)))
    ax.set_yticklabels(winners)
    ax.set_xlabel(f"weight on '{criterion}' (other weights rescaled)")
    ax.set_ylabel("winning route")
    for w, _a, _b in pts:
        ax.axvline(w, ls=":", c="#c23b22", lw=1.2)
        ax.annotate(
            f"switch @ {w:.2f}", (w, len(winners) - 0.6), fontsize=8, rotation=90, va="top", color="#c23b22"
        )
    ax.set_title(f"{shipment_id}: sensitivity to the '{criterion}' weight")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out = out or REPORTS_DIR / f"sensitivity_{shipment_id}_{criterion}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def strategy_comparison_plot(summary, out: Path | None = None) -> Path:
    """Baselines vs RouteWise. Fixed method order across panels, value labels,
    log scale on regret (values span ~2 orders of magnitude)."""
    order = [
        "A1_routewise",
        "A2_no_filters",
        "B4_equal_weights",
        "B2_fastest",
        "B3_lowest_risk",
        "B5_random",
        "B1_cheapest",
    ]
    d = summary.set_index("strategy").reindex([o for o in order if o in set(summary["strategy"])])
    names = list(d.index)
    pretty = {
        "A1_routewise": "RouteWise",
        "A2_no_filters": "Ablation (no filters)",
        "B4_equal_weights": "Equal weights",
        "B2_fastest": "Always fastest",
        "B3_lowest_risk": "Always lowest risk",
        "B5_random": "Random",
        "B1_cheapest": "Always cheapest",
    }
    colors = [
        "#1b7f3b" if n.startswith("A1") else "#7fbf8f" if n.startswith("A2") else "#8899aa" for n in names
    ]
    y = range(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    panels = [
        (
            "worst_case_regret_feasible",
            "Worst-case regret (log)\n1.0 = as bad as worst compliant option",
            True,
            "{:.2f}",
        ),
        ("mean_deadline_miss_rate", "Mean P(miss deadline)", False, "{:.0%}"),
        ("constraint_violation_rate", "Share of picks violating a hard constraint", False, "{:.0%}"),
    ]
    for ax, (col, title, log, fmt) in zip(axes, panels, strict=True):
        vals = d[col].values
        ax.barh(list(y), vals, color=colors)
        if log:
            ax.set_xscale("log")
            ax.axvline(1.0, ls=":", c="#c23b22", lw=1, zorder=0)
            ax.set_xlim(0.4, max(vals) * 3.0)
            ax.set_xticks([0.5, 1, 2, 5, 10, 20])
            ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20"])
        for i, v in enumerate(vals):
            ax.annotate(
                fmt.format(v), (v, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8
            )
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.25)
        if not log:
            ax.set_xlim(0, max(vals) * 1.3)
    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels([pretty.get(n, n) for n in names], fontsize=9.5)
    axes[0].invert_yaxis()
    fig.tight_layout()
    out = out or REPORTS_DIR / "strategy_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def transit_distribution_plot(shipment_id: str, out: Path | None = None) -> Path:
    """Simulated transit distributions — why the mean hides the tail."""
    import numpy as np

    cfg = load_config()
    shipments, routes_by = load_dataset()
    s = next(x for x in shipments if x.shipment_id == shipment_id)
    routes = routes_by[shipment_id]
    sc = cfg["simulation"]
    sims = simulate_all(routes, s.deadline_days, sc["n_runs"], sc["seed"], sc["lognormal_sigma"])
    rec = recommend(s, routes, cfg=cfg, explain=False, with_sensitivity=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for r in routes:
        sim = sims[r.route_id]
        chosen = r.route_id == rec.chosen_route_id
        np.linspace(sim.p50_transit - 6, sim.p90_transit + 8, 60)
        ax.plot(
            [sim.p50_transit, sim.p90_transit],
            [r.route_id, r.route_id],
            lw=6 if chosen else 3,
            c="#1b7f3b" if chosen else "#8899aa",
            solid_capstyle="round",
            label="chosen (P50-P90)" if chosen else "other (P50-P90)",
        )
        ax.plot(sim.mean_transit, r.route_id, "o", c="black", ms=4)
    ax.axvline(s.deadline_days, ls="--", c="#c23b22", label=f"deadline {s.deadline_days:.1f}d")
    ax.set_xlabel("transit days (dot = mean, bar = P50 to P90)")
    ax.set_title(f"{shipment_id}: simulated transit spread vs deadline")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles, strict=True))
    ax.legend(uniq.values(), uniq.keys(), fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    out = out or REPORTS_DIR / f"transit_{shipment_id}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
