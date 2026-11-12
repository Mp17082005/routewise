"""Typed schemas for RouteWise (pydantic v2).

Every object that crosses a module boundary is validated here, so a bad
generator or a hand-edited CSV fails loudly instead of silently producing a
wrong recommendation.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Mode(StrEnum):
    SEA = "sea"
    AIR = "air"
    RAIL = "rail"
    ROAD = "road"


class Priority(StrEnum):
    URGENT = "urgent"
    STANDARD = "standard"
    BUDGET = "budget"


class CargoType(StrEnum):
    GENERAL = "general"
    PERISHABLE = "perishable"
    FRAGILE = "fragile"
    HIGH_VALUE = "high_value"


class Hub(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hub_id: str
    name: str
    country: str
    congestion: float = Field(ge=0.0, le=1.0)
    geopolitical_risk: float = Field(ge=0.0, le=1.0)
    handling_cost: float = Field(ge=0.0)
    is_chokepoint: bool = False


class Leg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    leg_id: str
    from_hub: str
    to_hub: str
    mode: Mode
    distance_km: float = Field(gt=0)
    base_days: float = Field(gt=0)
    cost_per_teu: float = Field(gt=0)
    delay_prob: float = Field(ge=0.0, le=1.0)
    delay_mean_days: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _distinct_endpoints(self) -> Leg:
        if self.from_hub == self.to_hub:
            raise ValueError(f"leg {self.leg_id}: from_hub == to_hub")
        return self


class Shipment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shipment_id: str
    origin: str
    destination: str
    cargo_type: CargoType
    priority: Priority
    volume_teu: float = Field(gt=0)
    value_usd: float = Field(gt=0)
    deadline_days: float = Field(gt=0)
    hard_deadline: bool
    budget_cap: float | None = Field(default=None, gt=0)
    max_geo_risk: float | None = Field(default=None, ge=0.0, le=1.0)
    lane_id: str | None = None
    note: str | None = None  # e.g. "trap", "near_tie", "infeasible"

    @model_validator(mode="after")
    def _distinct_endpoints(self) -> Shipment:
        if self.origin == self.destination:
            raise ValueError(f"shipment {self.shipment_id}: origin == destination")
        return self


class Route(BaseModel):
    """A fully-aggregated route option for one shipment."""

    model_config = ConfigDict(extra="forbid")

    route_id: str
    shipment_id: str
    template: str
    leg_ids: list[str] = Field(min_length=1)
    hub_path: list[str] = Field(min_length=2)
    mode_mix: str
    total_cost: float = Field(gt=0)
    transit_days: float = Field(gt=0)
    delay_prob: float = Field(ge=0.0, le=1.0)
    delay_mean_days: float = Field(ge=0.0)
    congestion: float = Field(ge=0.0, le=1.0)
    geo_risk: float = Field(ge=0.0, le=1.0)
    n_transshipments: int = Field(ge=0)
    uses_chokepoint: bool = False

    @field_validator("hub_path", "leg_ids", mode="before")
    @classmethod
    def _split_strings(cls, v):
        if isinstance(v, str):
            return [p for p in v.split("|") if p]
        return v

    # ---- derived, deterministic, no simulation needed ----------------------
    @property
    def expected_transit_days(self) -> float:
        return self.transit_days + self.delay_prob * self.delay_mean_days

    @property
    def risk_score(self) -> float:
        return 0.5 * self.congestion + 0.5 * self.geo_risk


class SimResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route_id: str
    p50_transit: float
    p90_transit: float
    mean_transit: float
    p_miss_deadline: float
    expected_days_late: float


class RouteEvaluation(BaseModel):
    """Everything RouteWise computed about one route option."""

    model_config = ConfigDict(extra="forbid")

    route: Route
    sim: SimResult
    feasible: bool
    violations: list[str] = Field(default_factory=list)
    violation_magnitude: float = 0.0
    dominated_by: str | None = None
    criteria: dict[str, float] = Field(default_factory=dict)  # raw values
    normalised: dict[str, float] = Field(default_factory=dict)  # 0..1
    contributions: dict[str, float] = Field(default_factory=dict)  # weight*norm
    score: float | None = None
    rank: int | None = None


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shipment: Shipment
    profile: str
    weights: dict[str, float]
    chosen_route_id: str
    evaluations: list[RouteEvaluation]
    compliant: bool
    fallback_used: bool
    margin: float | None = None
    raw_near_tie: bool = False
    stability: float | None = None
    confidence: str | None = None
    warnings: list[str] = Field(default_factory=list)
    explanation: str = ""

    def by_id(self, route_id: str) -> RouteEvaluation:
        for e in self.evaluations:
            if e.route.route_id == route_id:
                return e
        raise KeyError(route_id)

    @property
    def chosen(self) -> RouteEvaluation:
        return self.by_id(self.chosen_route_id)

    @property
    def scored(self) -> list[RouteEvaluation]:
        """Scored routes, best first."""
        return sorted(
            [e for e in self.evaluations if e.score is not None],
            key=lambda e: (e.score, e.route.route_id),
        )
