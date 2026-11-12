"""Reproducible synthetic dataset generator (FR-1).

Design goals, each verified by a test in tests/test_data.py:
  * 40 shipments, 3-5 route options each, every route multi-leg capable
  * a negative cost-vs-time correlation (cheap routes are slow)
  * >= 10% of routes dominated (exercises the Pareto filter)
  * injected trap / near-tie / infeasible shipments
  * fully deterministic given a seed
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR
from .schemas import CargoType, Hub, Leg, Priority, Route, Shipment

SEED = 42

# ------------------------------------------------------------------ hubs ----
# congestion, geopolitical_risk, handling_cost (USD/TEU), chokepoint
HUB_SPECS: list[tuple[str, str, str, float, float, float, bool]] = [
    ("MUM", "Mumbai (Nhava Sheva)", "IN", 0.62, 0.22, 140, False),
    ("CMB", "Colombo", "LK", 0.45, 0.25, 110, False),
    ("JEA", "Jebel Ali", "AE", 0.30, 0.30, 160, False),
    ("SIN", "Singapore", "SG", 0.35, 0.08, 150, False),
    ("PSD", "Port Said (Suez)", "EG", 0.68, 0.78, 190, True),
    ("JED", "Jeddah (Red Sea)", "SA", 0.55, 0.72, 150, True),
    ("CPT", "Cape Town", "ZA", 0.40, 0.28, 120, False),
    ("RTM", "Rotterdam", "NL", 0.48, 0.07, 170, False),
    ("HAM", "Hamburg", "DE", 0.52, 0.07, 175, False),
    ("SHA", "Shanghai", "CN", 0.72, 0.30, 130, False),
    ("PUS", "Busan", "KR", 0.38, 0.18, 135, False),
    ("LAX", "Los Angeles", "US", 0.75, 0.10, 210, False),
    ("NYC", "New York", "US", 0.58, 0.09, 205, False),
    ("SAN", "Santos", "BR", 0.50, 0.26, 125, False),
]

# from, to, mode, distance_km, base_days, cost_per_teu, delay_prob, delay_mean_days
LEG_SPECS: list[tuple[str, str, str, float, float, float, float, float]] = [
    ("MUM", "CMB", "sea", 1550, 2.0, 320, 0.12, 1.2),
    ("MUM", "JEA", "sea", 1900, 2.5, 380, 0.14, 1.5),
    ("MUM", "SIN", "sea", 3900, 6.0, 700, 0.16, 2.0),
    ("CMB", "JEA", "sea", 3000, 4.0, 520, 0.15, 1.8),
    ("CMB", "SIN", "sea", 2700, 3.5, 480, 0.13, 1.5),
    ("CMB", "PSD", "sea", 5200, 7.5, 880, 0.30, 4.0),
    ("JEA", "PSD", "sea", 3400, 5.0, 640, 0.28, 3.6),
    ("JEA", "JED", "sea", 1900, 3.0, 420, 0.26, 3.2),
    ("JED", "PSD", "sea", 1300, 2.0, 300, 0.34, 4.5),
    ("PSD", "RTM", "sea", 5600, 8.0, 900, 0.18, 2.4),
    ("PSD", "HAM", "sea", 6100, 8.8, 960, 0.19, 2.5),
    ("CMB", "CPT", "sea", 7300, 11.0, 1050, 0.11, 1.6),
    ("MUM", "CPT", "sea", 7600, 11.5, 1090, 0.11, 1.6),
    ("CPT", "RTM", "sea", 11000, 15.0, 1420, 0.10, 1.8),
    ("CPT", "HAM", "sea", 11400, 15.6, 1480, 0.10, 1.8),
    ("CPT", "SAN", "sea", 6300, 9.5, 980, 0.12, 1.9),
    ("SIN", "SHA", "sea", 3900, 5.0, 620, 0.17, 2.1),
    ("SIN", "PUS", "sea", 4600, 6.0, 700, 0.14, 1.8),
    ("SIN", "LAX", "sea", 14000, 19.0, 1750, 0.16, 2.6),
    ("SHA", "PUS", "sea", 900, 1.5, 240, 0.12, 1.0),
    ("SHA", "LAX", "sea", 10400, 14.0, 1380, 0.18, 2.8),
    ("PUS", "LAX", "sea", 9600, 13.0, 1300, 0.15, 2.3),
    ("SHA", "RTM", "sea", 19500, 28.0, 2300, 0.20, 3.4),
    ("LAX", "NYC", "rail", 4500, 5.0, 620, 0.13, 1.2),
    ("RTM", "HAM", "rail", 500, 1.0, 180, 0.08, 0.6),
    ("NYC", "RTM", "sea", 6000, 8.5, 880, 0.12, 1.7),
    ("SAN", "RTM", "sea", 9800, 13.5, 1250, 0.13, 2.0),
    ("MUM", "SAN", "sea", 13000, 18.0, 1700, 0.15, 2.6),
    # air legs: expensive, fast, reliable (used by sea-air hybrids)
    ("JEA", "HAM", "air", 5000, 1.6, 3600, 0.08, 0.7),
    ("JEA", "RTM", "air", 5200, 1.7, 3700, 0.08, 0.7),
    ("SIN", "LAX", "air", 14000, 2.0, 5200, 0.09, 0.8),
    ("CMB", "RTM", "air", 8200, 2.0, 4600, 0.09, 0.8),
    ("SHA", "LAX", "air", 10400, 1.8, 4800, 0.09, 0.8),
    ("MUM", "RTM", "air", 7000, 1.8, 4300, 0.08, 0.7),
]


@dataclass(frozen=True)
class LaneTemplate:
    name: str
    hubs: tuple[str, ...]


# 10 lanes, each with 4-6 path templates of deliberately different character.
LANES: dict[str, list[LaneTemplate]] = {
    "MUM-RTM": [
        LaneTemplate("direct_suez", ("MUM", "JEA", "PSD", "RTM")),
        LaneTemplate("transship_colombo_suez", ("MUM", "CMB", "PSD", "RTM")),
        LaneTemplate("double_transship_redsea", ("MUM", "JEA", "JED", "PSD", "RTM")),
        LaneTemplate("cape_corridor", ("MUM", "CPT", "RTM")),
        LaneTemplate("cape_via_colombo", ("MUM", "CMB", "CPT", "RTM")),
        LaneTemplate("sea_air_hybrid", ("MUM", "JEA", "RTM")),  # JEA-RTM air
    ],
    "MUM-HAM": [
        LaneTemplate("direct_suez", ("MUM", "JEA", "PSD", "HAM")),
        LaneTemplate("transship_colombo_suez", ("MUM", "CMB", "PSD", "HAM")),
        LaneTemplate("cape_corridor", ("MUM", "CPT", "HAM")),
        LaneTemplate("double_transship_redsea", ("MUM", "JEA", "JED", "PSD", "HAM")),
        LaneTemplate("sea_air_hybrid", ("MUM", "JEA", "HAM")),  # JEA-HAM air
        LaneTemplate("suez_rail_last_mile", ("MUM", "CMB", "PSD", "RTM", "HAM")),
    ],
    "CMB-RTM": [
        LaneTemplate("direct_suez", ("CMB", "PSD", "RTM")),
        LaneTemplate("via_jebel_ali", ("CMB", "JEA", "PSD", "RTM")),
        LaneTemplate("cape_corridor", ("CMB", "CPT", "RTM")),
        LaneTemplate("air_direct", ("CMB", "RTM")),
        LaneTemplate("redsea_corridor", ("CMB", "JEA", "JED", "PSD", "RTM")),
    ],
    "SHA-RTM": [
        LaneTemplate("direct_longhaul", ("SHA", "RTM")),
        LaneTemplate("via_singapore_suez", ("SHA", "SIN", "MUM", "CMB", "PSD", "RTM")),
        LaneTemplate("transpacific_rail_atlantic", ("SHA", "LAX", "NYC", "RTM")),
        LaneTemplate("sea_air_via_lax", ("SHA", "LAX", "NYC", "RTM")),
        LaneTemplate("busan_transship", ("SHA", "PUS", "LAX", "NYC", "RTM")),
    ],
    "SHA-LAX": [
        LaneTemplate("direct_sea", ("SHA", "LAX")),
        LaneTemplate("busan_transship", ("SHA", "PUS", "LAX")),
        LaneTemplate("singapore_transship", ("SHA", "SIN", "LAX")),
        LaneTemplate("air_direct", ("SHA", "LAX")),  # air variant
    ],
    "SIN-RTM": [
        LaneTemplate("suez_via_colombo", ("SIN", "CMB", "PSD", "RTM")),
        LaneTemplate("suez_via_jebel_ali", ("SIN", "CMB", "JEA", "PSD", "RTM")),
        LaneTemplate("cape_corridor", ("SIN", "CMB", "CPT", "RTM")),
        LaneTemplate("transpacific", ("SIN", "LAX", "NYC", "RTM")),
        LaneTemplate("air_via_colombo", ("SIN", "CMB", "RTM")),
    ],
    "MUM-LAX": [
        LaneTemplate("via_singapore", ("MUM", "SIN", "LAX")),
        LaneTemplate("via_colombo_singapore", ("MUM", "CMB", "SIN", "LAX")),
        LaneTemplate("via_shanghai", ("MUM", "SIN", "SHA", "LAX")),
        LaneTemplate("sea_air_singapore", ("MUM", "SIN", "LAX")),  # SIN-LAX air
    ],
    "CMB-HAM": [
        LaneTemplate("direct_suez", ("CMB", "PSD", "HAM")),
        LaneTemplate("suez_rail", ("CMB", "PSD", "RTM", "HAM")),
        LaneTemplate("cape_corridor", ("CMB", "CPT", "HAM")),
        LaneTemplate("via_jebel_ali", ("CMB", "JEA", "PSD", "HAM")),
    ],
    "MUM-NYC": [
        LaneTemplate("suez_atlantic", ("MUM", "JEA", "PSD", "RTM", "NYC")),
        LaneTemplate("cape_atlantic", ("MUM", "CPT", "RTM", "NYC")),
        LaneTemplate("pacific_rail", ("MUM", "SIN", "LAX", "NYC")),
        LaneTemplate("brazil_transship", ("MUM", "SAN", "RTM", "NYC")),
    ],
    "SIN-HAM": [
        LaneTemplate("suez_via_colombo", ("SIN", "CMB", "PSD", "HAM")),
        LaneTemplate("cape_corridor", ("SIN", "CMB", "CPT", "HAM")),
        LaneTemplate("suez_rail", ("SIN", "CMB", "PSD", "RTM", "HAM")),
        LaneTemplate("transpacific", ("SIN", "LAX", "NYC", "RTM", "HAM")),
    ],
}

# Templates whose final leg should be flown when an air leg exists.
AIR_TEMPLATES = {"sea_air_hybrid", "air_direct", "sea_air_via_lax", "air_via_colombo", "sea_air_singapore"}

DEADLINE_MULTIPLIER = {"urgent": 1.10, "standard": 1.30, "budget": 1.60}


def _hubs() -> dict[str, Hub]:
    return {
        h[0]: Hub(
            hub_id=h[0],
            name=h[1],
            country=h[2],
            congestion=h[3],
            geopolitical_risk=h[4],
            handling_cost=h[5],
            is_chokepoint=h[6],
        )
        for h in HUB_SPECS
    }


def _legs() -> dict[tuple[str, str, str], Leg]:
    out = {}
    for frm, to, mode, dist, days, cost, dp, dm in LEG_SPECS:
        leg = Leg(
            leg_id=f"L-{frm}-{to}-{mode}",
            from_hub=frm,
            to_hub=to,
            mode=mode,
            distance_km=dist,
            base_days=days,
            cost_per_teu=cost,
            delay_prob=dp,
            delay_mean_days=dm,
        )
        out[(frm, to, mode)] = out[(to, frm, mode)] = leg  # legs are bidirectional
    return out


def _pick_leg(legs, frm: str, to: str, prefer_air: bool) -> Leg | None:
    if prefer_air and (frm, to, "air") in legs:
        return legs[(frm, to, "air")]
    for mode in ("sea", "rail", "road", "air"):
        if (frm, to, mode) in legs:
            return legs[(frm, to, mode)]
    return None


def _build_route(
    rng: np.random.Generator,
    route_id: str,
    shipment: Shipment,
    tpl: LaneTemplate,
    hubs: dict[str, Hub],
    legs: dict,
) -> Route | None:
    """Aggregate leg attributes into route attributes (PRD 3.3 step 4)."""
    prefer_air = tpl.name in AIR_TEMPLATES
    chain: list[Leg] = []
    for a, b in zip(tpl.hubs[:-1], tpl.hubs[1:], strict=False):
        # air only on the final leg of a hybrid, so it stays a realistic sea-air
        leg = _pick_leg(legs, a, b, prefer_air and b == tpl.hubs[-1])
        if leg is None:
            return None
        chain.append(leg)
    if not chain:
        return None

    vol = shipment.volume_teu
    n_transship = max(0, len(tpl.hubs) - 2)
    # Economics of transshipment: feeder/consolidated services are CHEAPER per
    # TEU (slow steaming, better load factors) but add dwell days at each
    # intermediate hub. This is the core cost-vs-time trade-off in the data.
    freight = sum(leg.cost_per_teu * vol for leg in chain) * (0.72**n_transship)
    handling = sum(hubs[h].handling_cost * vol for h in tpl.hubs) * 0.5
    total_cost = freight + handling

    # transit = sailing days + dwell that scales with hub congestion
    sail = sum(leg.base_days for leg in chain)
    dwell = sum(1.0 + 6.0 * hubs[h].congestion for h in tpl.hubs[1:-1])
    transit = sail + dwell + 0.5  # 0.5d origin cut-off

    # independent leg delays -> route delay probability (assumption 3)
    p_ok = 1.0
    for leg in chain:
        p_ok *= 1.0 - leg.delay_prob
    delay_prob = 1.0 - p_ok
    delay_mean = (
        sum(leg.delay_prob * leg.delay_mean_days for leg in chain) / delay_prob if delay_prob > 1e-9 else 0.0
    )

    geo_risk = max(hubs[h].geopolitical_risk for h in tpl.hubs)  # worst link
    congestion = float(np.mean([hubs[h].congestion for h in tpl.hubs]))  # mean

    # +/-5% measurement noise on the estimates available at decision time
    n = lambda: float(rng.uniform(0.95, 1.05))  # noqa: E731
    return Route(
        route_id=route_id,
        shipment_id=shipment.shipment_id,
        template=tpl.name,
        leg_ids=[leg.leg_id for leg in chain],
        hub_path=list(tpl.hubs),
        mode_mix="+".join(sorted({leg.mode.value for leg in chain})),
        total_cost=round(total_cost * n(), 2),
        transit_days=round(transit * n(), 2),
        delay_prob=round(min(0.98, delay_prob * n()), 4),
        delay_mean_days=round(delay_mean * n(), 3),
        congestion=round(min(1.0, congestion), 4),
        geo_risk=round(min(1.0, geo_risk), 4),
        n_transshipments=max(0, len(tpl.hubs) - 2),
        uses_chokepoint=any(hubs[h].is_chokepoint for h in tpl.hubs),
    )


def route_p90(route: Route) -> float:
    """P90 transit from the real Monte Carlo simulator (same code the decision
    engine uses), so generated deadlines and evaluated constraints agree."""
    from .simulate import simulate_route

    return simulate_route(route, deadline_days=1e9).p90_transit


def _dominates(a: Route, b: Route) -> bool:
    av = (a.total_cost, a.expected_transit_days, a.delay_prob, a.risk_score)
    bv = (b.total_cost, b.expected_transit_days, b.delay_prob, b.risk_score)
    return all(x <= y for x, y in zip(av, bv, strict=True)) and any(
        x < y for x, y in zip(av, bv, strict=True)
    )


def generate(seed: int = SEED, n_shipments: int = 40, out_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    from .config import load_config

    cfg_max_geo = load_config()["constraints"]["default_max_geo_risk"]
    rng = np.random.default_rng(seed)
    hubs, legs = _hubs(), _legs()
    lane_ids = list(LANES)

    shipments: list[Shipment] = []
    routes: list[Route] = []

    # Injected test cases (PRD 3.3 step 7), assigned to fixed indices so the
    # dataset always contains the hard cases the test plan looks for.
    trap_idx = {3, 11, 19, 27, 34}  # cheapest route misses a hard deadline
    near_tie_idx = {5, 14, 23, 31}  # two routes within ~2% score
    infeasible_idx = {9, 29}  # no route satisfies the constraints
    dominated_idx = {2, 7, 13, 17, 21, 25, 28, 30, 33, 35, 36, 37, 38, 39}  # add an explicitly worse route

    for i in range(n_shipments):
        lane = lane_ids[i % len(lane_ids)]
        origin, destination = lane.split("-")
        priority = [Priority.URGENT, Priority.STANDARD, Priority.BUDGET][
            int(rng.choice(3, p=[0.25, 0.50, 0.25]))
        ]
        cargo = [CargoType.GENERAL, CargoType.PERISHABLE, CargoType.FRAGILE, CargoType.HIGH_VALUE][
            int(rng.choice(4, p=[0.55, 0.15, 0.15, 0.15]))
        ]
        volume = float(np.round(rng.uniform(1, 12), 1))
        value = float(np.round(rng.uniform(40_000, 900_000), -2))

        sid = f"S{i + 1:03d}"
        waiver = False
        note = (
            "trap"
            if i in trap_idx
            else "near_tie"
            if i in near_tie_idx
            else "infeasible"
            if i in infeasible_idx
            else None
        )

        # provisional shipment (deadline/caps filled after routes are priced)
        ship = Shipment(
            shipment_id=sid,
            origin=origin,
            destination=destination,
            cargo_type=cargo,
            priority=priority,
            volume_teu=volume,
            value_usd=value,
            deadline_days=99.0,
            hard_deadline=bool(rng.random() < 0.65),
            lane_id=lane,
            note=note,
        )

        templates = LANES[lane]
        # Reserve a slot on shipments where a dominated or near-tie route will
        # be injected, so the injection is never silently skipped.
        reserve = 1 if (i in dominated_idx or i in near_tie_idx) else 0
        hi = min(5 - reserve, len(templates))
        k = int(rng.integers(3, hi + 1))
        chosen_idx = rng.choice(len(templates), size=k, replace=False)
        chosen = [templates[j] for j in sorted(chosen_idx)]

        s_routes: list[Route] = []
        for j, tpl in enumerate(chosen):
            r = _build_route(rng, f"R-{sid}-{j + 1}", ship, tpl, hubs, legs)
            if r is not None:
                s_routes.append(r)
        if len(s_routes) < 3:  # pad with any remaining template
            for tpl in templates:
                if len(s_routes) >= 3:
                    break
                if tpl.name in {r.template for r in s_routes}:
                    continue
                r = _build_route(rng, f"R-{sid}-{len(s_routes) + 1}", ship, tpl, hubs, legs)
                if r is not None:
                    s_routes.append(r)
        if len(s_routes) < 3:
            continue

        # -- inject a strictly dominated route (worse on every criterion) ----
        if i in dominated_idx and len(s_routes) < 5:
            base = min(s_routes, key=lambda r: r.total_cost)
            s_routes.append(
                base.model_copy(
                    update={
                        "route_id": f"R-{sid}-{len(s_routes) + 1}",
                        "template": base.template + "_slow_feeder",
                        "total_cost": round(base.total_cost * 1.12, 2),
                        "transit_days": round(base.transit_days * 1.18, 2),
                        "delay_prob": round(min(0.98, base.delay_prob * 1.25 + 0.02), 4),
                        "delay_mean_days": round(base.delay_mean_days * 1.15 + 0.1, 3),
                        "congestion": round(min(1.0, base.congestion + 0.05), 4),
                        "geo_risk": round(min(1.0, base.geo_risk + 0.03), 4),
                    }
                )
            )

        # -- deadline from the fastest route (PRD 3.3 step 5) ----------------
        # The hard-deadline constraint is checked against *simulated* P90
        # transit, so the deadline is derived from exactly the same quantity --
        # the generator calls the real simulator rather than approximating it.
        # Without this, deadlines and constraints disagree and shipments become
        # accidentally infeasible.
        def approx_p90(r: Route) -> float:
            return route_p90(r)

        # Anchor: urgent shipments are quoted against the fastest option of
        # any mode; others against the fastest *surface* (non-air) option,
        # because a customer shipping by sea does not set a 2-day deadline just
        # because an air leg exists on the lane.
        surface = [r for r in s_routes if "air" not in r.mode_mix]
        anchor = s_routes if (priority == Priority.URGENT or not surface) else surface
        fastest_p90 = min(approx_p90(r) for r in anchor)
        mult = DEADLINE_MULTIPLIER[priority.value]
        deadline = round(fastest_p90 * mult, 1)

        cheapest = min(s_routes, key=lambda r: r.total_cost)
        if note == "trap":
            # A trap must leave a *valid alternative*: the deadline is set so
            # the cheapest option cannot make it, but a compliant alternative
            # comfortably can. Uses the same simulated P90 the constraint
            # check uses.
            #
            # The alternative is chosen from routes that pass the OTHER
            # constraints (the cargo geo-risk cap), because a deadline
            # calibrated against a geo-blocked route would leave the shipment
            # with no valid option at all.
            geo_cap = (
                cfg_max_geo
                if cargo in (CargoType.PERISHABLE, CargoType.FRAGILE, CargoType.HIGH_VALUE)
                else None
            )
            eligible = [r for r in s_routes if geo_cap is None or r.geo_risk <= geo_cap]
            if not eligible:  # all-chokepoint lane: waiver applies later
                eligible = list(s_routes)
            fastest_route = min(eligible, key=approx_p90)
            lo, hi = approx_p90(fastest_route), approx_p90(cheapest)
            if cheapest.route_id != fastest_route.route_id and hi > lo * 1.08:
                # Place the deadline strictly between the two tails: above the
                # alternative's P90 (so a valid choice exists) and below the
                # cheapest route's P90 (so the cheap option really fails).
                deadline = round(min(hi * 0.97, lo * 1.15), 1)
                ship = ship.model_copy(update={"hard_deadline": True})
            else:
                note = None  # cannot build a meaningful trap on this lane
                ship = ship.model_copy(update={"note": None})
        if note == "near_tie":
            # clone the best-value route with ~1% perturbation -> score near-tie
            ref = min(s_routes, key=lambda r: r.expected_transit_days)
            if len(s_routes) < 5:
                s_routes.append(
                    ref.model_copy(
                        update={
                            "route_id": f"R-{sid}-{len(s_routes) + 1}",
                            "template": ref.template + "_alt_carrier",
                            "total_cost": round(ref.total_cost * 1.008, 2),
                            "transit_days": round(ref.transit_days * 0.995, 2),
                            "delay_prob": round(min(0.98, ref.delay_prob * 1.01), 4),
                        }
                    )
                )

        budget_cap = None
        max_geo = None
        costs = sorted(r.total_cost for r in s_routes)
        if note == "trap":
            budget_cap = None  # a trap tests the deadline, not the budget
        elif priority == Priority.BUDGET:
            budget_cap = round(costs[len(costs) // 2] * 1.05, 2)
        elif rng.random() < 0.5:
            budget_cap = round(costs[-1] * 1.10, 2)
        if cargo in (CargoType.PERISHABLE, CargoType.FRAGILE, CargoType.HIGH_VALUE):
            max_geo = cfg_max_geo
            if all(r.geo_risk > max_geo for r in s_routes):
                # Every option on this lane transits a chokepoint (e.g. all
                # Suez routings). A risk cap no route can meet is not a
                # decision input, it is an unserviceable lane -- in practice
                # the planner books under a documented waiver. We record the
                # waiver instead of silently creating a fake infeasible case.
                max_geo = None
                waiver = True

        # -- guarantee solvability except for the injected infeasible cases --
        # Constraints must bind (they have to eliminate weak options), but a
        # shipment where *nothing* passes is only interesting when we built it
        # that way on purpose. Relaxation order is deliberate:
        #   1. deadline  - a planning target, the softest knob
        #   2. budget    - commercial, negotiable
        #   3. geo risk  - NEVER relaxed: it is a cargo safety policy, and
        #                  bending it to manufacture a feasible route would be
        #                  exactly the modelling dishonesty this system exists
        #                  to prevent.
        if note != "infeasible":

            def passes(r: Route, dl: float, bud: float | None, geo: float | None) -> bool:
                return (
                    approx_p90(r) <= dl
                    and (bud is None or r.total_cost <= bud)
                    and (geo is None or r.geo_risk <= geo)
                )

            if note != "trap" and not any(passes(r, deadline, budget_cap, max_geo) for r in s_routes):
                # 1. push the deadline just past the best policy-compliant option.
                # Skipped for trap cases: their deadline was set deliberately,
                # and relaxing it would defuse the very trap under test.
                eligible = [r for r in s_routes if (max_geo is None or r.geo_risk <= max_geo)]
                if eligible:
                    deadline = round(min(approx_p90(r) for r in eligible) * 1.08, 1)
            if not any(passes(r, deadline, budget_cap, max_geo) for r in s_routes):
                # 2. lift the budget cap to the cheapest option that now qualifies
                ok_cost = [r.total_cost for r in s_routes if passes(r, deadline, None, max_geo)]
                budget_cap = round(min(ok_cost) * 1.02, 2) if ok_cost else None

        if note == "infeasible":
            # budget below the cheapest AND deadline below the fastest
            budget_cap = round(costs[0] * 0.80, 2)
            deadline = round(fastest_p90 * 0.70, 1)
            ship = ship.model_copy(update={"hard_deadline": True})
            max_geo = 0.20

        if waiver and note is None:
            note = "geo_waiver"
        ship = ship.model_copy(
            update={
                "note": note,
                "deadline_days": max(deadline, 0.5),
                "budget_cap": budget_cap,
                "max_geo_risk": max_geo,
            }
        )
        shipments.append(ship)
        routes.extend(s_routes)

    df_hubs = pd.DataFrame([h.model_dump() for h in hubs.values()])
    uniq_legs = {leg.leg_id: leg for leg in legs.values()}
    df_legs = pd.DataFrame([legs_.model_dump(mode="json") for legs_ in uniq_legs.values()])
    df_ship = pd.DataFrame([s.model_dump(mode="json") for s in shipments])
    df_routes = pd.DataFrame([r.model_dump(mode="json") for r in routes])
    df_routes["leg_ids"] = df_routes["leg_ids"].apply("|".join)
    df_routes["hub_path"] = df_routes["hub_path"].apply("|".join)

    # ---- observable data properties, written into the data card ------------
    n_dominated = 0
    for sid, _grp in df_routes.groupby("shipment_id"):
        rs = [r for r in routes if r.shipment_id == sid]
        for a in rs:
            if any(_dominates(b, a) for b in rs if b.route_id != a.route_id):
                n_dominated += 1
    corr = float(df_routes[["total_cost", "transit_days"]].corr().iloc[0, 1])

    out_dir = Path(out_dir) if out_dir else DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    df_hubs.to_csv(out_dir / "hubs.csv", index=False, lineterminator="\n")
    df_legs.to_csv(out_dir / "legs.csv", index=False, lineterminator="\n")
    df_ship.to_csv(out_dir / "shipments.csv", index=False, lineterminator="\n")
    df_routes.to_csv(out_dir / "routes.csv", index=False, lineterminator="\n")

    stats = {
        "seed": seed,
        "n_shipments": len(df_ship),
        "n_routes": len(df_routes),
        "routes_per_shipment_min": int(df_routes.groupby("shipment_id").size().min()),
        "routes_per_shipment_max": int(df_routes.groupby("shipment_id").size().max()),
        "cost_time_corr_within_lane": _within_lane_corr(df_routes, df_ship),
        "cost_time_corr_global": round(corr, 3),
        "dominated_routes": n_dominated,
        "dominated_share": round(n_dominated / len(df_routes), 3),
        "trap_cases": int((df_ship["note"] == "trap").sum()),
        "near_tie_cases": int((df_ship["note"] == "near_tie").sum()),
        "infeasible_cases": int((df_ship["note"] == "infeasible").sum()),
    }
    _write_data_card(out_dir, stats)
    return {"hubs": df_hubs, "legs": df_legs, "shipments": df_ship, "routes": df_routes, "stats": stats}


def _within_lane_corr(df_routes: pd.DataFrame, df_ship: pd.DataFrame) -> float:
    """Cost-vs-time correlation computed *within* each shipment's option set.

    The global correlation is dominated by lane length (a long lane is both
    expensive and slow), which hides the trade-off. The decision only ever
    compares routes inside one shipment, so that is where the trade-off must
    live: we z-score within shipment, then pool.
    """
    d = df_routes.copy()
    d["exp_transit"] = d["transit_days"] + d["delay_prob"] * d["delay_mean_days"]
    zs = []
    for _, grp in d.groupby("shipment_id"):
        if len(grp) < 3 or grp["total_cost"].std() == 0 or grp["exp_transit"].std() == 0:
            continue
        zc = (grp["total_cost"] - grp["total_cost"].mean()) / grp["total_cost"].std()
        zt = (grp["exp_transit"] - grp["exp_transit"].mean()) / grp["exp_transit"].std()
        zs.append(pd.DataFrame({"c": zc, "t": zt}))
    if not zs:
        return 0.0
    pooled = pd.concat(zs)
    return round(float(pooled["c"].corr(pooled["t"])), 3)


def _write_data_card(out_dir: Path, stats: dict) -> None:
    card = f"""# RouteWise synthetic data card

Generated by `src/routewise/generate_data.py` with seed **{stats["seed"]}**.
Re-running `routewise generate-data --seed {stats["seed"]}` reproduces these files byte-for-byte.

## Contents

| File | Rows | Description |
|---|---|---|
| `hubs.csv` | {len(HUB_SPECS)} | Ports/hubs with congestion, geopolitical risk, handling cost, chokepoint flag |
| `legs.csv` | {len(LEG_SPECS)} | Point-to-point legs (sea/air/rail) with distance, base days, cost/TEU, delay prob, delay mean |
| `shipments.csv` | {stats["n_shipments"]} | Shipments with cargo type, priority, volume, value, deadline and hard constraints |
| `routes.csv` | {stats["n_routes"]} | Aggregated route options ({stats["routes_per_shipment_min"]}-{stats["routes_per_shipment_max"]} per shipment) |

## How routes were constructed

1. **Hub graph** — 14 real-sounding ports. Port Said (Suez) and Jeddah (Red Sea)
   are marked chokepoints with high geopolitical risk (0.78 / 0.72).
2. **Lanes and templates** — 10 origin-destination lanes, each with 4-6 path
   templates: direct, one transshipment, two transshipments, an alternate
   corridor (Cape of Good Hope instead of Suez), and a sea-air hybrid.
3. **Aggregation per route**
   - cost = [sum(leg cost/TEU) x 0.72^(transshipments) + 0.5 x sum(hub handling)] x volume.
     The 0.72 factor is the feeder/consolidation discount: routing via more
     transshipment hubs is **cheaper** per TEU but adds dwell days. This is the
     mechanism that creates the cost-vs-time trade-off.
   - transit = sum(leg base days) + dwell (1.0 + 6.0 x congestion) per intermediate hub + 0.5d cut-off
   - delay probability = 1 - prod(1 - leg delay prob)   *(legs assumed independent)*
   - delay mean = probability-weighted mean of leg delay durations
   - geopolitical risk = **max** over hubs (one chokepoint can dominate a route)
   - congestion = **mean** over hubs
   - all four attributes carry +/-5% estimation noise
4. **Deadlines** are derived from the **simulated P90 transit** of the
   shipment's own fastest option: urgent x1.10, standard x1.30, budget x1.60.
   Urgent shipments anchor on the fastest option of any mode; others anchor on
   the fastest *surface* option (a sea shipper does not set a 2-day deadline
   just because an air leg exists on the lane).
   The generator calls the same Monte Carlo simulator the decision engine uses,
   so deadlines and the P90 hard-deadline check always agree — otherwise
   shipments become accidentally infeasible.
5. **Constraints** — budget caps on all budget-priority shipments (median route
   cost x1.05) and on ~50% of others; `max_geo_risk = 0.65` on perishable,
   fragile and high-value cargo.
6. **Solvability guard** — apart from the deliberately infeasible cases, each
   shipment is checked to have at least one fully compliant route. If not, the
   **deadline** is relaxed first, then the **budget cap**. The geopolitical risk
   cap is *never* relaxed: it is a cargo safety policy. Where every route on a
   lane breaches it (all-Suez routings), the shipment is marked
   `note=geo_waiver` — an unserviceable lane booked under a documented waiver,
   rather than a silently fabricated infeasible case.

## Injected trade-off structure

Cheaper routes use more transshipments, so they are slower and more congested;
the fast direct route goes through the Suez chokepoint and carries high
geopolitical risk; the safe Cape corridor is slower and dearer; the sea-air
hybrid is fastest and most reliable but far costlier.

- **Cost vs expected-transit correlation, within each shipment's option set: {stats["cost_time_corr_within_lane"]}**
  (this is the number that matters — it is the trade-off the decision actually faces)
- Global pooled correlation across all lanes: {stats["cost_time_corr_global"]}
  (positive, because long lanes are both slow and expensive — lane length is a confounder)

## Injected test cases

| Case | Count | Purpose |
|---|---|---|
| Dominated routes | {stats["dominated_routes"]} ({stats["dominated_share"]:.0%} of all routes) | Exercise the Pareto filter |
| Trap shipments (`note=trap`) | {stats["trap_cases"]} | Cheapest route misses a hard deadline |
| Near-tie shipments (`note=near_tie`) | {stats["near_tie_cases"]} | Two routes within ~1% — tests confidence labelling |
| Infeasible shipments (`note=infeasible`) | {stats["infeasible_cases"]} | No route satisfies the constraints — tests the fallback |

## Honesty note

This data is **plausible, not calibrated**. Freight rates, dwell times and
delay probabilities are hand-set to produce realistic trade-off *structure*,
not to match any real carrier tariff. No conclusion about real-world absolute
costs should be drawn from it; the conclusions in the report are about the
*decision procedure*, which is what the data was built to stress.
"""
    (out_dir / "data_card.md").write_text(card, encoding="utf-8", newline="\n")


def load_dataset(data_dir: Path | None = None) -> tuple[list[Shipment], dict[str, list[Route]]]:
    """Load generated CSVs into validated pydantic objects."""
    d = Path(data_dir) if data_dir else DATA_DIR
    df_s = pd.read_csv(d / "shipments.csv")
    df_r = pd.read_csv(d / "routes.csv")

    def clean(row: dict) -> dict:
        # CSV round-trip turns missing optional fields into float NaN;
        # pydantic wants None.
        return {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in row.items()}

    shipments = [Shipment(**clean(row)) for row in df_s.to_dict("records")]
    routes: dict[str, list[Route]] = {}
    for row in df_r.to_dict("records"):
        r = Route(**clean(row))
        routes.setdefault(r.shipment_id, []).append(r)
    return shipments, routes


if __name__ == "__main__":  # pragma: no cover
    out = generate()
    print(out["stats"])
