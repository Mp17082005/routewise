"""Shared fixtures and hand-built route factories."""

from __future__ import annotations

import pytest

from routewise.config import load_config
from routewise.generate_data import load_dataset
from routewise.schemas import Route, Shipment


def make_route(
    route_id: str = "R1",
    shipment_id: str = "S1",
    cost: float = 10_000,
    transit: float = 20.0,
    delay_prob: float = 0.2,
    delay_mean: float = 2.0,
    congestion: float = 0.4,
    geo_risk: float = 0.3,
    hub_path: tuple[str, ...] = ("MUM", "CMB", "RTM"),
    **kw,
) -> Route:
    return Route(
        route_id=route_id,
        shipment_id=shipment_id,
        template=kw.pop("template", "test"),
        leg_ids=kw.pop("leg_ids", ["L1", "L2"]),
        hub_path=list(hub_path),
        mode_mix=kw.pop("mode_mix", "sea"),
        total_cost=cost,
        transit_days=transit,
        delay_prob=delay_prob,
        delay_mean_days=delay_mean,
        congestion=congestion,
        geo_risk=geo_risk,
        n_transshipments=max(0, len(hub_path) - 2),
        uses_chokepoint=kw.pop("uses_chokepoint", False),
        **kw,
    )


def make_shipment(
    shipment_id: str = "S1",
    origin: str = "MUM",
    destination: str = "RTM",
    priority: str = "standard",
    cargo_type: str = "general",
    deadline: float = 30.0,
    hard: bool = True,
    budget: float | None = None,
    max_geo: float | None = None,
    **kw,
) -> Shipment:
    return Shipment(
        shipment_id=shipment_id,
        origin=origin,
        destination=destination,
        cargo_type=cargo_type,
        priority=priority,
        volume_teu=kw.pop("volume_teu", 5.0),
        value_usd=kw.pop("value_usd", 200_000.0),
        deadline_days=deadline,
        hard_deadline=hard,
        budget_cap=budget,
        max_geo_risk=max_geo,
        **kw,
    )


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def dataset():
    """The generated dataset. Generates it on demand if absent."""
    try:
        return load_dataset()
    except FileNotFoundError:  # pragma: no cover
        from routewise.generate_data import generate

        generate()
        return load_dataset()
