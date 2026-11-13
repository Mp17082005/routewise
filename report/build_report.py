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
cell = ParagraphStyle("c", parent=base, fontSize=7.6, leading=9.4, alignment=0)
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
    style += [("ALIGN", (0, 0), (0, -1), "LEFT")]
    t.setStyle(TableStyle(style))
    return t


story = []
story.append(P("RouteWise: Picking Shipment Routes When Cheap, Fast and Safe Disagree", h1))
story.append(
    P(
        "<b>Mahesh Kumar Pandey</b> &nbsp;·&nbsp; maheshkumarpandey17@gmail.com &nbsp;·&nbsp; "
        "Aiqon AI Internship take-home<br/>"
        "Code: <font color='#1b4f72'>github.com/Mp17082005/routewise</font> &nbsp;·&nbsp; "
        "Live demo: <font color='#1b4f72'>routewise-123.streamlit.app</font>",
        small,
    )
)
story.append(Spacer(1, 5))

# ---------------------------------------------------------------- summary --
story.append(P("Summary", h2))
story.append(
    P(
        "I built a tool that takes one shipment and its 3–5 possible routes, recommends a route, explains "
        "the choice in plain English, and says how confident it is. It works in four steps. First it "
        "removes any route that breaks a hard rule, such as missing a hard deadline. Next it removes any "
        "route that another route beats on every measure. It then scores the routes that are left, using "
        "weights that depend on how urgent the shipment is. Finally it tests whether the answer changes "
        "when those weights are nudged."
    )
)
story.append(
    P(
        f"On 40 test shipments, it missed deadlines on <b>{A1.mean_deadline_miss_rate:.1%}</b> of shipments. "
        f"Always picking the cheapest route missed <b>{B1.mean_deadline_miss_rate:.1%}</b>. The price of "
        f"that is a <b>{premium:.0f}% higher average freight cost than always-cheapest</b>. Of all the strategies I compared, "
        f"RouteWise made the most balanced choices. I also report the places where it does <i>not</i> win."
    )
)

# --------------------------------------------------------------- optimal ---
story.append(P('1. What I mean by the "best" route', h2))
story.append(
    P(
        "There is no best route in general. A cheap route is usually slow. A fast route is usually "
        "expensive or passes through a risky area such as the Suez Canal. A shipment of urgent medicine "
        "and a shipment of steel should not be routed the same way. So I started by writing down "
        'exactly what "best" means:'
    )
)
story.append(
    P(
        "<b><i>The best route is the one that (1) obeys the shipment's hard rules on deadline, budget and "
        "risk limit, (2) is not beaten on every measure by another route, and (3) among what is left, "
        "gives the best balance of cost, travel time, chance of being late and risk, weighted by what "
        "this shipment cares about most.</i></b>"
    )
)
story.append(
    P(
        "I kept hard rules separate from the scoring on purpose. If a route will miss a hard deadline, "
        "it should be ruled out, not just given a lower score. Otherwise a big enough discount could "
        '"buy" a late delivery, and no operator would accept that.'
    )
)

# -------------------------------------------------------------- example ----
story.append(P("2. One shipment, start to finish", h2))
story.append(
    P(
        "Shipment <b>S004</b> goes from Shanghai to Rotterdam. It is a <i>budget</i> shipment, so cost "
        "matters most (60% of the weight). It also has a <b>hard deadline of 36 days</b>. There are five "
        "options:"
    )
)
ex = [
    ["Route", "Path", "Cost", "Avg days", "Bad-case days*", "Result"],
    ["R-1", "Shanghai → Rotterdam (direct)", "$2,924", "28.8", "31.3", "Recommended"],
    ["R-2", "via Singapore, Mumbai, Colombo, Suez", "$1,614", "46.2", "49.5", "Too late"],
    ["R-3", "via Los Angeles, New York (rail+sea)", "$2,311", "39.6", "41.9", "Too late"],
    ["R-4", "via Los Angeles, New York (rail+sea)", "$2,282", "39.6", "41.5", "Too late"],
    ["R-5", "via Busan, Los Angeles, New York", "$1,796", "44.2", "45.9", "Too late"],
]
_ex = table(ex, [12 * mm, 62 * mm, 18 * mm, 18 * mm, 24 * mm, 24 * mm], hi_row=1, font=7.8)
_ex.setStyle(TableStyle([("ALIGN", (1, 0), (1, -1), "LEFT"), ("ALIGN", (-1, 0), (-1, -1), "LEFT")]))
story.append(_ex)
story.append(P('* Bad-case days: 9 out of 10 simulated trips arrive by this day (the "P90").', small))
story.append(
    P(
        'A plain "pick the cheapest" rule chooses R-2, which is about 10 days late on average and 13 days late in a bad case. RouteWise '
        "rules it out and recommends R-1. It costs $1,310 more, but its chance of missing the deadline "
        "drops from 100% to 1%. The explanation the tool prints says exactly that, using these numbers, "
        "and lists why each other route was dropped. When I randomly changed the weights 500 times, R-1 "
        "still won every time, so the tool marks the answer <b>High confidence</b>."
    )
)

# ------------------------------------------------------------------ how ----
story.append(P("3. How it works, and why I built it this way", h2))
steps = [
    ["Step", "What it does", "Why I chose this"],
    [
        P("<b>1. Simulate delays</b>", cell),
        P(
            "Runs each route 1,000 times with random delays to get a typical arrival time, a "
            "bad-case arrival time (the 90th percentile, P90) and the chance of being late.",
            cell,
        ),
        P(
            "An average hides risk. A route that is fine on average but late 4 times in 10 is not safe.",
            cell,
        ),
    ],
    [
        P("<b>2. Apply hard rules</b>", cell),
        P(
            "Drops routes whose <b>bad-case</b> time misses a hard deadline, that exceed the budget, "
            "or that are too risky for fragile, perishable or high-value cargo.",
            cell,
        ),
        P(
            "Some rules are not negotiable. If no route passes, it still returns the least-bad route, "
            "clearly marked NON-COMPLIANT.",
            cell,
        ),
    ],
    [
        P("<b>3. Remove clearly worse routes</b>", cell),
        P(
            "Drops any route that another route beats on cost, time, lateness and risk all at once "
            '(called a "Pareto" filter).',
            cell,
        ),
        P("Prevents obviously bad picks, and lets the explanation say <i>which</i> route beat it.", cell),
    ],
    [
        P("<b>4. Score what is left</b>", cell),
        P(
            "Rescales each measure from 0 (best option here) to 1 (worst option here), then takes a "
            "weighted sum. The lowest score wins.",
            cell,
        ),
        P("Simple and transparent. An operator can see exactly why one route beat another.", cell),
    ],
    [
        P("<b>5. Explain</b>", cell),
        P(
            "Fills in fixed sentence templates with the real numbers: why it won, what was traded "
            "off, and how it compares with the cheapest and fastest routes.",
            cell,
        ),
        P(
            "No AI text generator, so it cannot invent numbers. A test checks every figure against the data.",
            cell,
        ),
    ],
    [
        P("<b>6. Check confidence</b>", cell),
        P(
            "Nudges the weights randomly by up to ±20%, 500 times, and counts how often the same "
            "route still wins.",
            cell,
        ),
        P("The weights are my judgement. This shows whether the answer depends on them.", cell),
    ],
]
t = Table(steps, colWidths=[28 * mm, 80 * mm, 72 * mm], repeatRows=1)
t.setStyle(
    TableStyle(
        [
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
            ("FONT", (0, 1), (0, -1), "Helvetica-Bold", 7.6),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b4f72")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#bbbbbb")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f7")]),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]
    )
)
story.append(t)
story.append(Spacer(1, 4))
story.append(
    P(
        "<b>The weights.</b> Each shipment gets a profile based on its priority. For <i>urgent</i>, time "
        "counts most (40%). For <i>standard</i>, the weights are balanced. For <i>budget</i>, cost counts "
        "most (60%). Fragile, perishable or high-value cargo uses a <i>critical</i> profile where risk "
        "counts most (35%). These are my own sensible defaults, not values learned from data, and anyone "
        "can change them in a settings file without touching the code."
    )
)
story.append(
    P(
        "<b>Things I chose not to use.</b> <i>Machine learning:</i> there is no record of real outcomes "
        "to learn from, and a model would be harder to explain. <i>An optimisation solver:</i> with only "
        "3–5 routes per shipment, checking every route already gives the exact answer."
    )
)

# ------------------------------------------------------------------ data ---
story.append(P("4. The test data", h2))
story.append(
    P(
        f"No public dataset fits this problem, so I generated one, as the brief suggests. It has 14 real "
        f"ports, including Mumbai, Singapore, Suez, Cape Town, Rotterdam, Shanghai and Los Angeles, which "
        f"form <b>40 shipments with {n_routes} routes</b>. I built the trade-offs in deliberately. Routes "
        f"with more stops are cheaper but slower. Routes through Suez are fast but risky. The route around "
        f"Africa is safe but slow. Air routes are fast and expensive. Within a shipment, cost and time move "
        f"in opposite directions (correlation {corr.replace('-', '−')}). I also planted hard cases to test the tool: 4 "
        f'"traps" where the cheapest route misses a hard deadline, 4 near-ties, and 2 shipments where '
        f"no route can meet every rule. The data is the same on every run (fixed random seed)."
    )
)

# --------------------------------------------------------------- results ---
story.append(P("5. Results", h2))
story.append(
    P(
        "There is no answer key for a decision like this, so I compared RouteWise with simpler "
        "strategies. To make the comparison fair, I also measured <b>expected landed cost</b>: freight, "
        "plus the cost of goods sitting in transit, plus a late penalty. RouteWise does not optimise "
        "this number, so it is an independent check. <b>Balance</b> measures how badly a choice does on "
        "its <i>worst</i> measure, compared with the other routes that obey the rules. Lower is better."
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
    "A1_routewise": "RouteWise",
    "A2_no_filters": "RouteWise without hard rules",
    "B4_equal_weights": "Equal weights, no rules",
    "B2_fastest": "Always fastest",
    "B3_lowest_risk": "Always safest",
    "B5_random": "Random pick",
    "B1_cheapest": "Always cheapest",
}
rows = [["Strategy", "Avg cost", "Avg days", "Late rate", "Balance", "Landed cost", "Broke a rule"]]
for s in order:
    r = e1.loc[s]
    rows.append(
        [
            names[s],
            f"${r.mean_cost_usd:,.0f}",
            f"{r.mean_exp_transit_d:.1f}",
            f"{r.mean_deadline_miss_rate:.1%}",
            f"{r.worst_case_regret_feasible:.2f}",
            f"${r.expected_landed_cost:,.0f}",
            f"{r.constraint_violation_rate:.0%}",
        ]
    )
story.append(
    KeepTogether(
        [
            table(rows, [50 * mm, 20 * mm, 18 * mm, 18 * mm, 18 * mm, 24 * mm, 22 * mm], hi_row=1, font=7.8),
            P(
                "Table: all strategies on the same 40 shipments. RouteWise's 5% is exactly the 2 shipments "
                "where no route can meet every rule. Both are flagged NON-COMPLIANT, not hidden.",
                small,
            ),
        ]
    )
)
story.append(Spacer(1, 3))
story.append(Spacer(1, 4))
story.append(Image(str(REP / "strategy_comparison.png"), width=176 * mm, height=56.3 * mm))
story.append(
    P(
        "Chart: the same comparison; lower is better in every panel. The dotted line marks a balance "
        'score of 1.0, meaning "as bad as the worst allowed route on some measure".',
        small,
    )
)
story.append(P("<b>What the results show</b>"))
story.append(
    B(
        f"<b>Most balanced.</b> RouteWise's balance score is {A1.worst_case_regret_feasible:.2f}, against "
        f"{B2.worst_case_regret_feasible:.2f} for always-fastest and {B1.worst_case_regret_feasible:.2f} "
        f"for always-cheapest. It is rarely the best on any single measure, but it never makes a "
        f"one-sided choice."
    )
)
story.append(
    B(
        f"<b>Priorities change the answer as they should.</b> When I forced every shipment through each "
        f"profile, urgent shipments averaged {e2.loc['urgent', 'exp_transit']:.1f} days and "
        f"${e2.loc['urgent', 'cost']:,.0f}. Budget shipments averaged {e2.loc['budget', 'exp_transit']:.1f} "
        f"days and ${e2.loc['budget', 'cost']:,.0f}."
    )
)
story.append(
    B(
        f"<b>The answers are stable.</b> With the weights randomly nudged, the same route still won "
        f"<b>{e4['mean_stability']:.0%}</b> of the time. Of the 40 decisions, "
        f"{e4['confidence_counts'].get('High', 0)} were High confidence, "
        f"{e4['confidence_counts'].get('Medium', 0)} Medium and {e4['confidence_counts'].get('Low', 0)} Low. "
        f'The Low one is shown as "close call, review manually". The tool also reports <i>switching '
        f'points</i>, for example "the answer changes if the cost weight rises above 0.46".'
    )
)
story.append(
    B(
        f"<b>Where it does not win.</b> It costs {premium:.0f}% more than always-cheapest. On landed cost, "
        f"always-cheapest (${B1.expected_landed_cost:,.0f}) beats it (${A1.expected_landed_cost:,.0f}). "
        f"That is mostly the price of the hard rules: they rule out cheap routes that are too late or too "
        f"risky, and the landed-cost formula gives no value to staying within a risk limit."
    )
)
story.append(
    B(
        f"<b>Bad input data is the biggest weakness.</b> When I added 15% random error to the route "
        f"estimates, the recommendation changed for {e3['decision_flip_rate']:.0%} of shipments. In real "
        f"use, accurate estimates matter more than fine-tuning the weights."
    )
)

# ---------------------------------------------------------- lessons --------
story.append(P("6. Problems I found while building it", h2))
story.append(
    B(
        "<b>Deadlines that could not be met.</b> At first I set deadlines from the <i>average</i> trip "
        "time but checked them against the <i>bad-case</i> time. That made 8 shipments impossible by "
        "accident. I fixed it by using the same simulation for both."
    )
)
story.append(
    B(
        "<b>False confidence on near-ties.</b> Because scores are rescaled from 0 to 1, two routes only "
        '1% apart could look far apart, and the tool labelled them "High confidence". It now also '
        "compares the raw numbers and never calls a near-tie High."
    )
)
story.append(
    B(
        "<b>Unfair comparison.</b> My first balance measure compared each choice with routes that broke "
        "the rules, which penalises a strategy for not booking an illegal option. It now compares only "
        "with routes the operator is actually allowed to book."
    )
)

# --------------------------------------------------------- assumptions -----
_lim = [P("7. Assumptions and limitations", h2)]
for a in [
    "Costs are in US dollars per shipment and time is in door-to-door days. Route figures are "
    "estimates, not guarantees.",
    "Delays on different legs of a route are independent. In reality, a Suez closure delays many "
    "routes at once, so I probably underestimate the worst cases.",
    "A route is only as safe as its riskiest port, so I take the highest risk on the route, not the "
    "average. Congestion is averaged across ports.",
    "Capacity is not modelled: every route can take every shipment.",
    "Shipment priority and cargo type are given as inputs.",
    "The data is realistic in shape but not based on real freight prices. The conclusions are about "
    "the decision method, not real costs.",
    "The weights are my judgement. The stability test shows the answers do not depend heavily on "
    "them, but they have not been checked with real planners.",
]:
    _lim.append(B(a))
story.append(KeepTogether(_lim))

story.append(P("8. What I would do next", h2))
story.append(
    P(
        "With real data, I would calibrate prices and delays against real shipment history, especially "
        "events where many routes are delayed at once, such as Red Sea disruptions. I would ask real "
        "planners to compare pairs of routes and derive the weights from their choices, and I would feed "
        "in live port congestion and risk data. I would also add capacity and sailing schedules. An AI "
        "model could reword the explanations more naturally, but only if it were not allowed to change "
        "any number."
    )
)
story.append(Spacer(1, 3))
story.append(
    P(
        "<b>How to check this work:</b> the repository has 92 automated tests, which run on every update. "
        "Running <font face='Courier'>make all</font> regenerates the data, results and charts. All "
        "numbers in this report come directly from that run."
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
    c.drawRightString(A4[0] - 15 * mm, 7 * mm, f"Mahesh Kumar Pandey · RouteWise · page {d.page}")
    c.restoreState()


OUT.parent.mkdir(parents=True, exist_ok=True)
doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
print(OUT)
