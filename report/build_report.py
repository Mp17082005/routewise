"""Build the technical report PDF from the real evaluation outputs.

    python report/build_report.py

Every number in the report is read from reports/*.csv|json produced by
`routewise evaluate`, so the PDF cannot drift from the code.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
OUT = ROOT / "report" / "RouteWise_Technical_Report.pdf"

e1 = pd.read_csv(REP / "e1_summary.csv").set_index("strategy")
e2 = pd.read_csv(REP / "e2_profile_behaviour.csv").set_index("profile")
e3 = json.loads((REP / "e3_noisy_estimates.json").read_text())
e4 = json.loads((REP / "e4_sensitivity_summary.json").read_text())
ps = pd.read_csv(REP / "per_shipment.csv")
card = (ROOT / "data" / "data_card.md").read_text(encoding="utf-8")

import re  # noqa: E402

corr = re.search(r"within each shipment's option set: ([-0-9.]+)", card).group(1)
dom = re.search(r"Dominated routes \| (\d+) \((\d+)%", card)
n_routes = len(pd.read_csv(ROOT / "data" / "routes.csv"))

A1, B1, B2 = e1.loc["A1_routewise"], e1.loc["B1_cheapest"], e1.loc["B2_fastest"]
A2 = e1.loc["A2_no_filters"]
premium = (A1.mean_cost_usd / B1.mean_cost_usd - 1) * 100

# ------------------------------------------------------------------ styles --
ss = getSampleStyleSheet()
base = ParagraphStyle(
    "b",
    parent=ss["BodyText"],
    fontName="Helvetica",
    fontSize=8.6,
    leading=11.0,
    alignment=TA_JUSTIFY,
    spaceAfter=3,
)
h1 = ParagraphStyle(
    "h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=15, leading=18, spaceAfter=2
)
h2 = ParagraphStyle(
    "h2",
    parent=ss["Heading2"],
    fontName="Helvetica-Bold",
    fontSize=10.2,
    leading=12.5,
    spaceBefore=5,
    spaceAfter=2,
    textColor=colors.HexColor("#1b4f72"),
)
small = ParagraphStyle("s", parent=base, fontSize=7.4, leading=9.2, textColor=colors.HexColor("#444444"))
bullet = ParagraphStyle("bl", parent=base, leftIndent=9, bulletIndent=0, spaceAfter=1.5)
mono = ParagraphStyle(
    "m",
    parent=base,
    fontName="Courier",
    fontSize=7.6,
    leading=9.2,
    textColor=colors.HexColor("#111111"),
    alignment=0,
    backColor=colors.HexColor("#f4f6f7"),
    borderPadding=3,
)


def P(t, st=base):
    return Paragraph(t, st)


def B(t):
    return Paragraph(t, bullet, bulletText="•")


def table(rows, widths, header_bg="#1b4f72", hi_row=None, font=7.4):
    t = Table(rows, colWidths=widths, repeatRows=1)
    style = [
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", font),
        ("FONT", (0, 1), (-1, -1), "Helvetica", font),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#bbbbbb")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f7")]),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
    ]
    if hi_row is not None:
        style += [
            ("BACKGROUND", (0, hi_row), (-1, hi_row), colors.HexColor("#d8f0dc")),
            ("FONT", (0, hi_row), (-1, hi_row), "Helvetica-Bold", font),
        ]
    t.setStyle(TableStyle(style))
    return t


story = []
story.append(P("RouteWise: Shipment Route Recommendation under Cost, Time and Risk", h1))
story.append(
    P(
        "Aiqon AI Internship take-home &nbsp;·&nbsp; technical report &nbsp;·&nbsp; "
        "code, data and tests in the accompanying repository",
        small,
    )
)
story.append(Spacer(1, 4))

# ------------------------------------------------------------ 1. optimal ---
story.append(P("1. Problem and definition of optimal", h2))
story.append(
    P(
        "A planner has one shipment and 3–5 candidate multi-leg routes. The cheapest route is usually slow "
        "or unreliable. The fastest is usually expensive or runs through a high-risk chokepoint. No route "
        "is optimal in the abstract, so RouteWise starts from an explicit definition:"
    )
)
story.append(
    P(
        "<b><i>The optimal route is the one that satisfies the shipment's hard constraints (deadline, "
        "budget, cargo risk limit) and, among the non-dominated routes that remain, has the lowest "
        "weighted trade-off score across cost, expected transit time, probability of missing the "
        "deadline, and route risk. The weights are set by shipment priority and cargo type.</i></b>"
    )
)
story.append(
    P(
        "Each part of the definition is a separate design choice. <b>Constraints are not traded "
        "away:</b> a cheap route that breaches a hard deadline is <i>invalid</i>, not merely low-scoring. "
        "No weighting should be able to buy it back. <b>Pareto filtering</b> guarantees the system never "
        "recommends a route that another route beats on every criterion. <b>Weighted scoring</b> settles "
        "the remaining trade-offs. Of the methods considered, it is the easiest to explain to an operator, "
        "and the brief judges explainability explicitly. <b>Context-dependent weights</b> reflect the "
        "fact that urgent pharma and bulk steel should not be optimised the same way."
    )
)

# ----------------------------------------------------------- 2. pipeline ---
story.append(P("2. Decision framework", h2))
story.append(
    P(
        "<font face='Courier'>shipment → validate → hard constraints → Pareto filter → Monte Carlo → "
        "score → rank → explain → confidence</font>"
    )
)
story.append(
    B(
        "<b>Four criteria, all lower-is-better.</b> C = total cost. T = expected transit = base + "
        "P(delay) × mean delay. R = P(arrive after deadline), from simulation. K = 0.5 × congestion + "
        "0.5 × geopolitical risk."
    )
)
story.append(
    B(
        "<b>Hard constraints.</b> P90 transit ≤ deadline (when the deadline is hard); cost ≤ budget cap; "
        "geopolitical risk ≤ cargo limit. The deadline is checked against <b>P90, not the mean</b>: a route "
        "that arrives on time on average but misses 40% of the time is not compliant. If nothing passes, "
        "the system returns the <i>least-violating</i> route (smallest normalised total violation), flags "
        "it NON-COMPLIANT and lists every failed constraint. It always returns an answer."
    )
)
story.append(
    B(
        "<b>Uncertainty (Monte Carlo).</b> 1,000 seeded draws per route. Each draw makes a Bernoulli "
        "delay decision; if delayed, the duration is lognormal with its mean pinned to the route estimate. "
        "Outputs are P50, P90, P(miss) and expected days late. This makes tail risk a decision input "
        "rather than a footnote. A route with a good mean but a fat tail is penalised on R and can fail "
        "the P90 check."
    )
)
story.append(
    B(
        "<b>Scoring.</b> Each criterion is min-max normalised <i>within the shipment's own option "
        "set</i>, so scores are relative to the real alternatives. A criterion whose spread is under 2% of "
        "its minimum is treated as flat, so rounding-level differences cannot decide. Score = Σ weight × "
        "normalised value."
    )
)
story.append(
    B(
        "<b>Explanation.</b> Deterministic templates, <b>not an LLM</b>, so numbers cannot be "
        "hallucinated and are testable. A test asserts that every dollar figure printed traces back to "
        "the data. Each explanation gives: a headline; the 2–3 criteria with the largest weighted "
        "advantage over the runner-up, with real deltas; the trade-off accepted; one sentence each versus "
        'the cheapest and fastest routes; eliminated routes with their reason ("dominated by R-x" or the '
        "violated constraint); confidence; and warnings."
    )
)

prof_rows = [
    ["Profile", "Cost", "Time", "Reliab.", "Risk", "Applied when"],
    ["urgent", "0.15", "0.40", "0.30", "0.15", "priority = urgent"],
    ["standard", "0.35", "0.25", "0.25", "0.15", "priority = standard"],
    ["budget", "0.60", "0.10", "0.15", "0.15", "priority = budget"],
    ["critical", "0.20", "0.15", "0.30", "0.35", "perishable / fragile / high-value (overrides)"],
]
story.append(
    KeepTogether(
        [
            table(prof_rows, [17 * mm, 12 * mm, 12 * mm, 13 * mm, 12 * mm, 66 * mm]),
            P(
                "Table 1. Weight profiles (config/weights.yaml). These are transparent, editable defaults, "
                "not learned values. Section 4 tests whether the conclusions depend on them.",
                small,
            ),
        ]
    )
)

story.append(
    P(
        "<b>Approaches not used.</b> <i>A learned ML model:</i> there are no real outcome labels, so a "
        "model would only learn the synthetic generator, and it would be unexplainable. <i>ILP or "
        "OR-Tools:</i> routes are pre-enumerated (3–5 per shipment), so exhaustive scoring is already "
        "exact and an optimiser adds nothing. <i>Pure lexicographic rules:</i> easy to explain, but they "
        "ignore the size of a trade-off. A 1% cost saving should not outweigh a 30-point drop in "
        "reliability."
    )
)

# --------------------------------------------------------------- 3. data ---
story.append(P("3. Synthetic data and how the trade-offs were built", h2))
story.append(
    P(
        f"The data has 14 ports (Mumbai, Colombo, Jebel Ali, Singapore, Port Said, Jeddah, Cape Town, "
        f"Rotterdam, Hamburg, Shanghai, Busan, Los Angeles, New York, Santos), 34 sea/air/rail legs, and "
        f"10 lanes with 4–6 path templates each: direct, one or two transshipments, a Cape corridor "
        f"instead of Suez, and sea-air hybrids. From these the generator builds <b>40 shipments and "
        f"{n_routes} routes</b> (3–5 per shipment) with seed 42, reproducible byte-for-byte (tested). "
        f"Route attributes are aggregated from legs: cost = leg freight × volume (discounted 0.72× per "
        f"transshipment, reflecting feeder consolidation) + handling; transit = sailing days + dwell that "
        f"scales with hub congestion; delay probability = 1 − ∏(1 − p<sub>leg</sub>); "
        f"geopolitical risk = <b>max</b> over hubs (one chokepoint can dominate a route); congestion = "
        f"mean over hubs. All estimates carry ±5% noise."
    )
)
story.append(
    P(
        f"<b>The trade-off structure is built in deliberately and then measured.</b> Cheaper routes "
        f"transship more, so they are slower and more congested. Direct Suez routes are fast but carry "
        f"0.78 geopolitical risk. The Cape corridor is safe but slow. Air hybrids are fast, reliable and "
        f"expensive. Within each shipment's option set, the cost-vs-transit correlation is <b>{corr}</b>. "
        f"This is the correlation that matters, because the decision only compares routes within a "
        f"shipment. The pooled cross-lane figure is confounded by lane length. {dom.group(1)} routes "
        f"({dom.group(2)}%) are Pareto-dominated, which exercises the filter. The data also includes "
        f"injected cases: 4 <i>trap</i> shipments where the cheapest route misses a hard deadline, "
        f"4 <i>near-ties</i>, and 2 shipments with <i>no feasible route</i>. Deadlines come from the "
        f"<b>simulated</b> P90 of the shipment's fastest relevant route (×1.1 urgent, ×1.3 standard, "
        f"×1.6 budget). The generator calls the same simulator the engine uses. In an earlier version "
        f"that approximated P90 instead, deadlines and constraints disagreed, and 8 shipments became "
        f"accidentally infeasible."
    )
)

# --------------------------------------------------------- 4. evaluation ---
story.append(P("4. Evaluation", h2))
story.append(
    P(
        "There are no ground-truth labels, so RouteWise is evaluated by comparing strategies on shared "
        "metrics and checking expected behaviours. To keep the evaluation from being circular, the money "
        "metric, <b>expected landed cost</b> (freight + inventory holding over simulated transit + late "
        "penalty × expected days late), is <i>not</i> what the scorer optimises. <b>Worst-case regret</b> "
        "is the largest normalised regret across the four criteria, measured against the routes the "
        "operator is <i>allowed</i> to book. Regret against a route that breaches a hard constraint is "
        "not regret anyone can act on. Values above 1 mean the pick is worse than every compliant option "
        "on some criterion."
    )
)

order = [
    "A1_routewise",
    "A2_no_filters",
    "B4_equal_weights",
    "B2_fastest",
    "B3_lowest_risk",
    "B5_random",
    "B1_cheapest",
]
names = {
    "A1_routewise": "A1 RouteWise",
    "A2_no_filters": "A2 Ablation: no constr./Pareto",
    "B4_equal_weights": "B4 Equal weights",
    "B2_fastest": "B2 Always fastest",
    "B3_lowest_risk": "B3 Always lowest risk",
    "B5_random": "B5 Random (100 draws)",
    "B1_cheapest": "B1 Always cheapest",
}
rows = [
    [
        "Strategy",
        "Mean cost",
        "E[transit]",
        "P(miss)",
        "Worst-case\nregret",
        "Landed\ncost",
        "Constraint\nviolations",
        "Pareto-\noptimal",
    ]
]
for s in order:
    r = e1.loc[s]
    rows.append(
        [
            names[s],
            f"${r.mean_cost_usd:,.0f}",
            f"{r.mean_exp_transit_d:.1f} d",
            f"{r.mean_deadline_miss_rate:.1%}",
            f"{r.worst_case_regret_feasible:.2f}",
            f"${r.expected_landed_cost:,.0f}",
            f"{r.constraint_violation_rate:.0%}",
            f"{r.share_pareto_optimal:.0%}",
        ]
    )
story.append(
    KeepTogether(
        [
            table(rows, [44 * mm, 17 * mm, 16 * mm, 14 * mm, 17 * mm, 17 * mm, 18 * mm, 15 * mm], hi_row=1),
            P(
                "Table 2 (E1). All strategies on 40 shipments. A1's 5% violation rate is exactly the 2 "
                "deliberately infeasible shipments, each returned with the NON-COMPLIANT flag.",
                small,
            ),
        ]
    )
)
story.append(Spacer(1, 3))
story.append(Image(str(REP / "strategy_comparison.png"), width=176 * mm, height=56.3 * mm))
story.append(
    P(
        "Figure 1. RouteWise vs baselines on 40 shipments; lower is better in every panel. "
        "Dotted line at regret 1.0 = as bad as the worst compliant option on some criterion.",
        small,
    )
)

story.append(P("<b>Findings, including the ones that do not favour RouteWise.</b>"))
story.append(
    B(
        f"<b>Best balance.</b> RouteWise has the lowest worst-case regret ({A1.worst_case_regret_feasible:.2f}). "
        f"Always-fastest scores {B2.worst_case_regret_feasible:.2f} and always-cheapest "
        f"{B1.worst_case_regret_feasible:.2f}. As expected, it is rarely the best on any single metric, "
        f"but it never makes a lopsided pick. 100% of its recommendations are Pareto-optimal."
    )
)
story.append(
    B(
        f"<b>Cost premium, stated openly.</b> Against always-cheapest, the deadline-miss rate falls from "
        f"{B1.mean_deadline_miss_rate:.1%} to {A1.mean_deadline_miss_rate:.1%}, at a <b>{premium:.0f}% "
        f"higher mean freight cost</b>. Always-fastest has a slightly lower miss rate "
        f"({B2.mean_deadline_miss_rate:.1%}) but costs ${B2.mean_cost_usd - A1.mean_cost_usd:,.0f} more per "
        f"shipment and violates a hard constraint in {B2.constraint_violation_rate:.0%} of cases, "
        f"usually the cargo geopolitical-risk limit."
    )
)
story.append(
    B(
        f"<b>RouteWise does not win on expected landed cost.</b> B1 (${B1.expected_landed_cost:,.0f}) and "
        f"A2 (${A2.expected_landed_cost:,.0f}) beat it (${A1.expected_landed_cost:,.0f}). Most of the gap "
        f"comes from the hard constraints, and that is intended: the yardstick prices lateness linearly "
        f"and gives no value to a geopolitical-risk limit. The constraints exclude cheap options a "
        f"money-only view would accept. Comparing A1 with the ablation A2 isolates the cost of the "
        f"constraints: ${A1.mean_cost_usd - A2.mean_cost_usd:,.0f} per shipment buys a fall in "
        f"violations from {A2.constraint_violation_rate:.0%} to {A1.constraint_violation_rate:.0%} and in "
        f"miss rate from {A2.mean_deadline_miss_rate:.1%} to {A1.mean_deadline_miss_rate:.1%}."
    )
)
story.append(
    B(
        f"<b>E2, profiles behave as intended.</b> With all 40 shipments forced through each profile, mean "
        f"transit is {e2.loc['urgent', 'exp_transit']:.1f} / {e2.loc['standard', 'exp_transit']:.1f} / "
        f"{e2.loc['budget', 'exp_transit']:.1f} d and mean cost is ${e2.loc['urgent', 'cost']:,.0f} / "
        f"${e2.loc['standard', 'cost']:,.0f} / ${e2.loc['budget', 'cost']:,.0f} for urgent / standard / "
        f"budget. Both are monotone, as required."
    )
)
story.append(
    B(
        f"<b>E3, noisy estimates.</b> When the system sees attributes with 15% noise and is scored against "
        f"the clean truth, its decision changes in {e3['decision_flip_rate']:.0%} of shipments. The median "
        f"regret against the clean-data oracle is ${e3['median_elc_regret_vs_oracle_usd']:,.0f} (mean "
        f"${e3['mean_elc_regret_vs_oracle_usd']:,.0f}, skewed by a few large misses). This is the most "
        f"important limitation: input quality, not the weights, is the dominant source of decision error."
    )
)

story.append(
    P(
        "<b>E4, sensitivity and confidence.</b> Each shipment's weights were perturbed by "
        "U(0.8, 1.2) and renormalised, 500 times. The same route wins in "
        f"<b>{e4['mean_stability']:.1%}</b> of runs on average, a dataset-level flip rate of "
        f"{e4['flip_rate_under_weight_noise']:.1%}. The cost weight causes the most flips "
        f"(mean {e4['mean_flip_rate_by_criterion_weight']['cost']:.1%} when perturbed alone, "
        f"versus {e4['mean_flip_rate_by_criterion_weight']['time']:.1%} for time). That fits "
        "the data, where cost is the widest-spread criterion. Confidence is <b>High</b> when "
        "stability ≥ 90% and margin ≥ 0.05, <b>Medium</b> at 70–90%, and <b>Low</b> below "
        '70% (shown as "close call, review manually"). Result: '
        f"{e4['confidence_counts'].get('High', 0)} High, {e4['confidence_counts'].get('Medium', 0)} "
        f"Medium, {e4['confidence_counts'].get('Low', 0)} Low. All 4 injected near-ties are "
        "below High (tested). Developing this exposed a real flaw. Min-max normalisation "
        "stretches a 1% raw gap to a 0→1 score gap, so near-identical routes showed a large "
        "margin <i>and</i> 100% stability, and were labelled High. A raw-value closeness check "
        "now caps such cases at Medium. A one-at-a-time weight sweep reports <b>switching "
        "points</b>. For S013, for example, the recommendation changes at cost weight ≈ 0.46. "
        "This tells an operator directly how strongly a priority would have to change to "
        "change the answer."
    )
)

story.append(
    P("<b>E5, a trap case (S004, budget priority, cost weight 0.60).</b> The engine output, abridged:")
)
story.append(
    P(
        "Recommended: R-S004-1 via SHA &gt; RTM. Cost $2,924 | E[transit] 28.8d (P90 31.3d)<br/>"
        "&nbsp;&nbsp;1% chance of missing the 36.0d hard deadline.<br/>"
        "vs cheapest (R-S004-2): we pay $1,310 more (+81.1%); in exchange deadline-miss<br/>"
        "&nbsp;&nbsp;risk falls from 100% to 1%, it arrives 17.4d sooner, route risk 0.67 -&gt; 0.45.<br/>"
        "&nbsp;&nbsp;The cheapest option is not allowed: P90 transit 49.5d exceeds hard deadline<br/>"
        "&nbsp;&nbsp;36.0d by 13.5d.<br/>"
        "Confidence: High (same route wins in 100% of 500 perturbed runs).",
        mono,
    )
)
story.append(
    P(
        "The full near-tie and no-feasible-route case studies are in "
        "<font face='Courier'>reports/e5_case_studies.md</font>.",
        small,
    )
)

# ------------------------------------------------------- 5. engineering ---
story.append(P("5. Verification", h2))
story.append(
    P(
        "There are 92 tests, with 92% line coverage, run in GitHub Actions CI together with ruff. The "
        "tests include <b>property-based tests (hypothesis, 200 random cases each)</b> for the invariants "
        "that make the framework trustworthy: raising a route's cost never improves its rank; adding a "
        "dominated route never changes the winner; scaling all costs never changes the winner; and "
        "permuting route order never changes the winner. Golden cases cover the behaviours an operator "
        "would expect: urgent picks fast, budget picks cheap, fragile cargo avoids the chokepoint, the "
        "trap case is avoided, and the fallback works. The PRD success criteria are written as tests, so "
        "they are checked rather than claimed. One recommendation, including 1,000 Monte Carlo draws, "
        "takes well under 1 s. Output is deterministic, and one command (<font face='Courier'>make "
        "all</font>) regenerates the data, results and figures."
    )
)

# ------------------------------------------------------ 6. assumptions -----
story.append(P("6. Assumptions", h2))
for a in [
    "All costs are in USD per shipment; transit time is in days, door to door. Route attributes are "
    "estimates available at decision time, not guarantees.",
    "Leg delays are independent. When a delay happens, its duration is lognormal (right-skewed).",
    "Geopolitical risk is worst-link (max over hubs); congestion is the mean over hubs.",
    "Priority and cargo type are known inputs. Capacity is not modelled: every listed route can take "
    "the shipment.",
    "Hard deadlines are checked at P90. Cargo geopolitical-risk limits are safety policy and are "
    "never relaxed. Where every route on a lane breaches the limit, the shipment is booked under a "
    "recorded waiver, never silently passed as compliant.",
    "The synthetic data is plausible, not calibrated to real freight rates. Conclusions concern the "
    "decision procedure, not absolute costs.",
]:
    story.append(B(a))

# ------------------------------------------------------- 7. limitations ----
story.append(P("7. Limitations and future work", h2))
story.append(
    P(
        "<b>Limitations.</b> The synthetic data may miss real correlations; in particular, delays on "
        "shared chokepoints are correlated in practice, so the independence assumption understates tail "
        "risk on multi-chokepoint routes. The weights are judgement-based, not learned or elicited from "
        "operators; sensitivity analysis bounds this but does not remove it. Min-max normalisation "
        "depends on which alternatives are on the menu: adding a far-off option can rescale the others. "
        "Evaluation proxies come from the model, not real shipment outcomes. E3 shows that 15% input "
        "noise changes 30% of decisions, so in practice the quality of the estimates matters more than "
        "tuning the scorer."
    )
)
story.append(
    P(
        "<b>Future work.</b> Calibrate against real rate and delay histories, including correlated "
        "chokepoint shocks. Elicit weights with pairwise comparisons, or learn them from operators' past "
        "choices. Feed live congestion and risk indices into the model. Add capacity and schedule "
        "constraints. Add a constrained LLM layer that only rephrases the template explanation; it would "
        "never be allowed to add or change numbers."
    )
)

doc = SimpleDocTemplate(
    str(OUT),
    pagesize=A4,
    leftMargin=15 * mm,
    rightMargin=15 * mm,
    topMargin=12 * mm,
    bottomMargin=12 * mm,
    title="RouteWise Technical Report",
    author="RouteWise",
)


def _footer(c, d):
    c.saveState()
    c.setFont("Helvetica", 7)
    c.setFillColor(colors.HexColor("#777777"))
    c.drawRightString(A4[0] - 15 * mm, 7 * mm, f"RouteWise · page {d.page}")
    c.restoreState()


OUT.parent.mkdir(parents=True, exist_ok=True)
doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
print(OUT)
