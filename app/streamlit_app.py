"""RouteWise demo UI (FR-11).

Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from routewise.config import CRITERIA, load_config, profile_for, weights_for  # noqa: E402
from routewise.explain import comparison_table  # noqa: E402
from routewise.generate_data import load_dataset  # noqa: E402
from routewise.recommend import recommend  # noqa: E402
from routewise.sensitivity import switching_points, weight_sweep  # noqa: E402

st.set_page_config(page_title="RouteWise", layout="wide")


@st.cache_data
def _data():
    return load_dataset()


cfg = load_config()
shipments, routes_by = _data()
by_id = {s.shipment_id: s for s in shipments}

st.title("RouteWise — shipment route recommendation")
st.caption(
    "Hard constraints → Pareto filter → Monte Carlo reliability → profile-weighted score → "
    "explanation + confidence. Optimal = lowest weighted trade-off score among non-dominated "
    "routes that satisfy the shipment's hard constraints."
)

# ------------------------------------------------------------------ sidebar --
with st.sidebar:

    def _label(sid):
        s = by_id[sid]
        tag = f" [{s.note}]" if s.note else ""
        return f"{sid}  {s.origin}→{s.destination}  {s.priority.value}/{s.cargo_type.value}{tag}"

    sid = st.selectbox("Shipment", list(by_id), format_func=_label)
    s = by_id[sid]
    default_prof = profile_for(s.priority.value, s.cargo_type.value, cfg)
    profiles = [p for p in cfg["profiles"] if p != "equal"] + ["equal"]
    prof = st.selectbox("Weight profile", profiles, index=profiles.index(default_prof))

    st.markdown("**Weights** (renormalised to sum to 1)")
    base_w = weights_for(prof, cfg)
    raw = {c: st.slider(c, 0.0, 1.0, float(base_w[c]), 0.05, key=f"{prof}-{c}") for c in CRITERIA}
    total = sum(raw.values()) or 1.0
    weights = {c: v / total for c, v in raw.items()}
    st.caption(" · ".join(f"{c} {w:.2f}" for c, w in weights.items()))

    st.markdown("---")
    st.markdown(
        f"**Deadline** {s.deadline_days:.1f}d ({'hard' if s.hard_deadline else 'soft'})  \n"
        f"**Budget cap** {'$' + format(s.budget_cap, ',.0f') if s.budget_cap else '—'}  \n"
        f"**Max geo risk** {s.max_geo_risk if s.max_geo_risk is not None else '—'}  \n"
        f"**Volume** {s.volume_teu} TEU · **Value** ${s.value_usd:,.0f}"
    )

rec = recommend(s, routes_by[sid], profile=prof, weights=weights, cfg=cfg)

# --------------------------------------------------------------- headline ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Recommended", rec.chosen_route_id)
c2.metric("Confidence", rec.confidence, f"{rec.stability:.0%} stable under ±20% weights", delta_color="off")
c3.metric("Score margin", f"{rec.margin:.3f}" if rec.margin is not None else "—")
c4.metric("Compliant", "Yes" if rec.compliant else "NO")
for w in rec.warnings:
    st.warning(w)

# ------------------------------------------------------------------ table ---
st.subheader("All route options")
df = pd.DataFrame(comparison_table(rec))
show = df[
    [
        "route_id",
        "status",
        "path",
        "modes",
        "cost_usd",
        "exp_transit_d",
        "p50_d",
        "p90_d",
        "p_miss_deadline",
        "risk",
        "geo_risk",
        "score",
        "rank",
        "reason",
    ]
]
st.dataframe(
    show.style.apply(
        lambda r: ["background-color: #d8f0dc" if r["status"] == "chosen" else ""] * len(r), axis=1
    ).format(
        {
            "cost_usd": "${:,.0f}",
            "exp_transit_d": "{:.1f}",
            "p50_d": "{:.1f}",
            "p90_d": "{:.1f}",
            "p_miss_deadline": "{:.0%}",
            "risk": "{:.2f}",
            "geo_risk": "{:.2f}",
            "score": lambda v: "—" if pd.isna(v) else f"{v:.3f}",
        }
    ),
    width="stretch",
    hide_index=True,
)

left, right = st.columns([1.1, 1])

# ----------------------------------------------------------------- pareto ---
with left:
    st.subheader("Trade-off map")
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    colours = {"chosen": "#1b7f3b", "scored": "#2c6fbb", "infeasible": "#c23b22"}
    for _, r in df.iterrows():
        col = colours.get(r["status"], "#999999")
        ax.scatter(
            r["cost_usd"], r["exp_transit_d"], s=120 + 900 * r["risk"], c=col, alpha=0.75, edgecolors="black"
        )
        ax.annotate(
            r["route_id"].split("-")[-1],
            (r["cost_usd"], r["exp_transit_d"]),
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=9,
        )
    ax.axhline(s.deadline_days, ls="--", c="#c23b22", lw=1)
    ax.set_xlabel("cost (USD)")
    ax.set_ylabel("expected transit (days)")
    ax.set_title("green=chosen · blue=scored · red=infeasible · grey=dominated\nbubble = risk", fontsize=9)
    ax.grid(alpha=0.25)
    st.pyplot(fig, width="stretch")

# ------------------------------------------------------------ explanation ---
with right:
    st.subheader("Why this route")
    st.code(rec.explanation, language=None)

# ------------------------------------------------------------ sensitivity ---
st.subheader("Switching points — how much would a priority have to change?")
crit = st.radio("Vary the weight on", CRITERIA, horizontal=True)
sweep = weight_sweep(s, routes_by[sid], crit, cfg, steps=41)
pts = switching_points(sweep)
if pts:
    for w, a, b in pts:
        st.write(f"- at **{crit} weight ≈ {w:.2f}** the recommendation switches **{a} → {b}**")
else:
    st.write(
        f"- **{sweep[0][1]}** wins for every {crit} weight from 0 to 1 — the decision does not hinge on it."
    )
st.line_chart(
    pd.DataFrame({"weight": [x for x, _ in sweep], "winner": [w for _, w in sweep]})
    .assign(winner_idx=lambda d: d["winner"].astype("category").cat.codes)
    .set_index("weight")[["winner_idx"]]
)
st.caption(
    "Profile weights here use the shipment's default profile; sliders above affect the headline recommendation."
)
