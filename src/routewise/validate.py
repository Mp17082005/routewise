"""Data validation beyond schema types."""

from __future__ import annotations

from collections import Counter

from .schemas import Route, Shipment


class ValidationReport:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    @property
    def ok(self) -> bool:
        return not self.errors

    def __str__(self) -> str:
        lines = [f"errors={len(self.errors)} warnings={len(self.warnings)}"]
        lines += [f"  ERROR   {e}" for e in self.errors]
        lines += [f"  WARNING {w}" for w in self.warnings]
        return "\n".join(lines)


def validate_dataset(
    shipments: list[Shipment],
    routes: dict[str, list[Route]],
    min_routes: int = 3,
    max_routes: int = 5,
) -> ValidationReport:
    rep = ValidationReport()

    ids = [s.shipment_id for s in shipments]
    dupes = [k for k, v in Counter(ids).items() if v > 1]
    if dupes:
        rep.errors.append(f"duplicate shipment ids: {dupes}")

    all_route_ids: list[str] = []
    for s in shipments:
        rs = routes.get(s.shipment_id, [])
        if not rs:
            rep.errors.append(f"{s.shipment_id}: no route options")
            continue
        if not (min_routes <= len(rs) <= max_routes):
            rep.errors.append(f"{s.shipment_id}: {len(rs)} routes, expected {min_routes}-{max_routes}")
        for r in rs:
            all_route_ids.append(r.route_id)
            if r.shipment_id != s.shipment_id:
                rep.errors.append(f"{r.route_id}: shipment_id mismatch")
            if r.total_cost <= 0 or r.transit_days <= 0:
                rep.errors.append(f"{r.route_id}: non-positive cost or transit")
            if not 0.0 <= r.delay_prob <= 1.0:
                rep.errors.append(f"{r.route_id}: delay_prob out of [0,1]")
            if not 0.0 <= r.geo_risk <= 1.0 or not 0.0 <= r.congestion <= 1.0:
                rep.errors.append(f"{r.route_id}: risk value out of [0,1]")
            if r.hub_path[0] != s.origin or r.hub_path[-1] != s.destination:
                rep.errors.append(
                    f"{r.route_id}: path {r.hub_path[0]}->{r.hub_path[-1]} "
                    f"does not match shipment {s.origin}->{s.destination}"
                )
        if s.budget_cap is not None and all(r.total_cost > s.budget_cap for r in rs):
            rep.warnings.append(f"{s.shipment_id}: no route within budget_cap")
        if all(r.expected_transit_days > s.deadline_days for r in rs):
            rep.warnings.append(f"{s.shipment_id}: no route meets the deadline in expectation")

    dup_routes = [k for k, v in Counter(all_route_ids).items() if v > 1]
    if dup_routes:
        rep.errors.append(f"duplicate route ids: {dup_routes[:5]}")

    return rep
