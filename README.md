# RouteWise

[![CI](https://github.com/Mp17082005/routewise/actions/workflows/ci.yml/badge.svg)](https://github.com/Mp17082005/routewise/actions/workflows/ci.yml)

RouteWise takes a shipment and 3–5 candidate routes, then recommends one. It says why it chose that route in plain operator language, and it says how confident it is.

**Live demo:** https://routewise-123.streamlit.app/

This is my submission for the Aiqon AI Internship take-home, *Optimizing Shipment Routes Under Cost, Time, and Risk Constraints*. The technical report is [`report/RouteWise_Technical_Report.pdf`](report/RouteWise_Technical_Report.pdf).

```
Shipment → validate → hard constraints → Pareto filter → Monte Carlo → weighted score → rank → explain → confidence
```

## What "optimal" means here

> **The optimal route is the one that meets the shipment's hard constraints (deadline, budget, cargo risk limit). Among the routes left that no other route beats on every criterion, it has the lowest weighted trade-off score across cost, expected transit time, probability of missing the deadline, and route risk. The weights depend on shipment priority and cargo type.**

- Hard constraints are never traded away. A cheap route that misses a hard deadline is invalid. It is not just a low-scoring option.
- The system never recommends a route that another route beats on every criterion (Pareto filter).
- Weighted scoring is the most explainable method available, and explainability is what the brief asks for.
- Weights depend on context. Urgent pharma and bulk steel should not be scored the same way.

## Headline results (40 shipments, seed 42)

| Strategy | Mean cost | Mean E[transit] | Deadline-miss rate | Worst-case regret† | Hard-constraint violations |
|---|---:|---:|---:|---:|---:|
| **RouteWise (A1)** | $15,596 | 26.5 d | **6.0%** | **0.62** | **5%** (the 2 deliberately infeasible shipments) |
| Ablation: no constraints/Pareto (A2) | $13,708 | 28.6 d | 11.2% | 0.68 | 15% |
| Equal weights (B4) | $16,795 | 23.4 d | 8.5% | 3.62 | 33% |
| Always fastest (B2) | $21,902 | 18.5 d | 5.7% | 3.69 | 33% |
| Always lowest risk (B3) | $17,731 | 27.3 d | 26.4% | 8.67 | 28% |
| Random (B5) | $15,117 | 30.0 d | 23.6% | 11.00 | 40% |
| Always cheapest (B1) | $10,188 | 32.6 d | 22.7% | 18.19 | 55% |

† Worst-case regret is the largest normalised regret over the four criteria, measured against the routes the operator is *allowed* to book. Lower means the choice is better balanced.

- **100%** of recommendations are Pareto-optimal.
- RouteWise has the **lowest worst-case regret**, about 6–30× lower than the single-metric baselines.
- The deadline-miss rate is **6.0% vs 22.7%** for always-cheapest. The cost of that is a **53% freight premium** over always-cheapest, reported openly.
- Under ±20% random weight noise, the same route still wins **96.7%** of the time. Results: 31 High, 8 Medium and 1 Low confidence. The Low case is flagged for manual review.
- Profiles behave as intended. Mean transit is 23.5 d for urgent, 27.0 d for standard and 30.9 d for budget. Mean cost is $19.1k, $15.7k and $13.7k.
- Not every finding favours RouteWise. On *expected landed cost* (an independent money measure the scorer does not optimise), RouteWise is **not** the cheapest. The report discusses this.

## Quick start

Requires Python 3.11+.

```bash
git clone https://github.com/Mp17082005/routewise.git
cd routewise
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

routewise generate-data --seed 42      # rebuild data/ (byte-identical to the committed CSVs)
routewise validate                     # schema + range checks
pytest                                 # 92 tests, ~10 s
routewise evaluate                     # E1–E5 tables + figures → reports/
streamlit run app/streamlit_app.py     # interactive demo
```

Or use `make all`, which regenerates the data, then validates, tests and evaluates in one command.

### CLI

```bash
routewise recommend --shipment S004                 # route table + explanation
routewise recommend --shipment S004 --profile urgent
routewise recommend --shipment S004 --json          # machine-readable
routewise sensitivity --shipment S013 --criterion cost   # weight sweep + switching points
routewise recommend-all                             # all 40 shipments → reports/recommendations.csv
```

Example output for a **trap case**. The cheapest route looks attractive under a budget profile, but it misses the hard deadline:

```
Recommended: R-S004-1 via SHA > RTM (sea, 0 transshipments).
Cost $2,924 | expected transit 28.8d (P50 28.2d, P90 31.3d) | 1% chance of missing the 36.0d hard deadline | route risk 0.45.
Weighting profile: budget (cost 0.60, time 0.10, reliability 0.15, risk 0.15) — budget priority, general cargo.

vs cheapest (R-S004-2): we pay $1,310 more (+81.1%); in exchange deadline-miss risk falls from 100% to 1%,
  it arrives 17.4d sooner on average, route risk falls from 0.67 to 0.45.
  The cheapest option is not allowed: P90 transit 49.5d exceeds hard deadline 36.0d by 13.5d.

Eliminated options:
  - R-S004-2: P90 transit 49.5d exceeds hard deadline 36.0d by 13.5d
  ...
Confidence: High — the same route wins in 100% of 500 runs with every weight randomly perturbed by up to +/-20%.
```

## How it works

| Stage | Module | What it does |
|---|---|---|
| Data | `generate_data.py` | Builds a reproducible synthetic dataset: 14 hubs, 34 legs, 10 lanes, 40 shipments, 163 routes. See [`data/data_card.md`](data/data_card.md). |
| Validate | `validate.py`, `schemas.py` | Uses pydantic v2 schemas, then checks ranges, endpoints and route counts. |
| Constraints | `constraints.py` | Checks P90 transit ≤ hard deadline, cost ≤ budget cap, and geopolitical risk ≤ the cargo limit. If no route passes, it falls back to the *least-violating* route and flags it NON-COMPLIANT. It always returns an answer. |
| Pareto | `pareto.py` | Removes dominated routes and records which route dominated each one, so the explanation can name it. |
| Uncertainty | `simulate.py` | Runs 1,000 Monte Carlo draws per route. Delays are Bernoulli × lognormal, which gives a right-skewed tail. Outputs are P50, P90, P(miss deadline) and expected days late. |
| Scoring | `scoring.py` | Min-max normalises within the shipment's own options. Criteria spread by less than 2% are treated as flat. Each profile applies its own weighted sum. |
| Explanation | `explain.py` | Fills deterministic templates. **No LLM is used**, so every number traces back to data. It covers why the route won, the trade-off accepted, comparisons with the cheapest and fastest routes, eliminated routes with reasons, confidence, and warnings. |
| Sensitivity | `sensitivity.py` | Runs 500 trials with weights perturbed by ±20% and measures input noise. It also sweeps one weight at a time to find **switching points**, then assigns a High, Medium or Low confidence label. |
| Evaluation | `evaluate.py`, `plots.py` | Compares 5 baselines and 1 ablation. Includes regret, deadline-miss rate, expected landed cost, Pareto share, violation rate and stability, across experiments E1–E5. |

### Weight profiles (`config/weights.yaml`)

| Profile | Cost | Time | Reliability | Risk | Used when |
|---|---:|---:|---:|---:|---|
| urgent | 0.15 | 0.40 | 0.30 | 0.15 | priority = urgent |
| standard | 0.35 | 0.25 | 0.25 | 0.15 | priority = standard |
| budget | 0.60 | 0.10 | 0.15 | 0.15 | priority = budget |
| critical | 0.20 | 0.15 | 0.30 | 0.35 | perishable, fragile or high-value cargo (overrides priority) |

These weights are **transparent, editable defaults**. They were not learned. The sensitivity analysis checks that the conclusions do not depend on their exact values. All thresholds live in YAML, not in code.

## Assumptions

1. All costs are in USD per shipment. Transit time is in days, door to door.
2. Route attributes are estimates available at decision time, not guarantees.
3. Leg delays are independent, so route delay probability = 1 − ∏(1 − leg delay probability).
4. When a delay happens, its duration is lognormal (right-skewed), with the mean pinned to the route's delay mean.
5. Geopolitical risk uses the **worst** link on the route, because one chokepoint can dominate. Congestion uses the **mean** over hubs.
6. Shipment priority and cargo type are known inputs.
7. Capacity is not modelled. Every listed route can take the shipment.
8. The synthetic data is plausible but not calibrated to real freight rates.
9. Hard deadlines are checked against **P90** transit, not the mean. A route that makes the deadline on average but misses it 40% of the time does not count as compliant.
10. Cargo geopolitical-risk limits are a safety policy and are never relaxed. Where every route on a lane breaches the limit, the shipment is booked under a recorded waiver (`note=geo_waiver`). It is not silently treated as compliant.

## Testing

The suite has **92 tests** with **92% line coverage** of `src/`. CI runs on every push.

| Group | Examples |
|---|---|
| Data | Schema valid; no nulls; probabilities in [0,1]; 3–5 routes per shipment; **seed 42 reproduces the committed CSVs byte-for-byte** |
| Data realism | Cost–time correlation within each shipment's options is negative (−0.37); ≥10% of routes dominated (11%); trap, near-tie and infeasible cases present |
| Constraints / Pareto / Scoring / Simulation | Hand-made cases; weights sum to 1; 2% spread rule; P50 ≤ P90; delay_prob = 0 gives zero variance |
| **Property tests (hypothesis, 200 cases each)** | Raising a route's cost never improves its rank; adding a dominated route never changes the winner; scaling all costs never changes the winner; permuting route order never changes the winner; score is monotone |
| Golden cases | Urgent picks fast; budget picks cheap; fragile cargo avoids the chokepoint; trap; fallback |
| Explanation | Names the chosen route and the runner-up; **every $ figure traces to data**; lists eliminated routes with reasons |
| Sensitivity | Stability in [0,1]; zero noise gives 100%; **injected near-ties never receive High confidence** |
| PRD success criteria | 100% Pareto-optimal; lowest worst-case regret; deadline-miss rate below cheapest; urgent faster than budget |
| Integration / NFR | CLI runs on all 40 shipments; one recommendation with 1,000 MC runs takes < 1 s; output is deterministic |

## Repository layout

```
config/weights.yaml          profiles, thresholds, simulation + sensitivity settings
data/                        hubs.csv legs.csv shipments.csv routes.csv data_card.md
src/routewise/               schemas, generate_data, validate, constraints, pareto, simulate,
                             scoring, recommend, explain, sensitivity, evaluate, plots, cli
app/streamlit_app.py         demo UI
tests/                       92 tests (unit, property, golden, integration)
reports/                     generated tables (e1–e5) and figures
report/                      technical report (PDF)
```

## Out of scope

The following are out of scope, by design: real carrier APIs or live data; learned ML models (there are no real outcome labels to learn from); vessel scheduling; consolidating multiple shipments; and customs or regulatory modelling. The routes are pre-enumerated (3–5 per shipment), so scoring every option exhaustively gives the exact answer. An ILP or OR-Tools solver would add complexity without improving the result.
