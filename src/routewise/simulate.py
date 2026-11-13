"""Monte Carlo transit simulation.

A route's delay is a distribution, not a number. The mean hides the tail that
actually causes missed deadlines, so reliability enters the score as
P(arrive after deadline) estimated here rather than as a point estimate.
"""

from __future__ import annotations

import zlib

import numpy as np

from .schemas import Route, SimResult


def _stable_hash(s: str) -> int:
    """CRC32, not builtin hash(): str hashing is randomised per process."""
    return zlib.crc32(s.encode("utf-8"))


def _lognormal_with_mean(rng: np.random.Generator, mean: float, sigma: float, size: int) -> np.ndarray:
    """Lognormal samples whose *arithmetic mean* equals `mean` (right-skewed)."""
    if mean <= 0:
        return np.zeros(size)
    mu = np.log(mean) - 0.5 * sigma**2
    return rng.lognormal(mean=mu, sigma=sigma, size=size)


def simulate_route(
    route: Route,
    deadline_days: float,
    n_runs: int = 1000,
    seed: int = 42,
    sigma: float = 0.6,
) -> SimResult:
    """Deterministic given (route_id, seed): the stream is seeded per route."""
    # Per-route seed derived from the global seed + route id keeps results
    # stable regardless of the order routes are simulated in.
    rs = np.random.default_rng([seed, _stable_hash(route.route_id)])

    occurs = rs.random(n_runs) < route.delay_prob
    durations = _lognormal_with_mean(rs, route.delay_mean_days, sigma, n_runs)
    transit = route.transit_days + np.where(occurs, durations, 0.0)

    late = np.maximum(0.0, transit - deadline_days)
    return SimResult(
        route_id=route.route_id,
        p50_transit=float(np.percentile(transit, 50)),
        p90_transit=float(np.percentile(transit, 90)),
        mean_transit=float(transit.mean()),
        p_miss_deadline=float((transit > deadline_days).mean()),
        expected_days_late=float(late.mean()),
    )


def simulate_all(
    routes: list[Route], deadline_days: float, n_runs: int = 1000, seed: int = 42, sigma: float = 0.6
) -> dict[str, SimResult]:
    return {
        r.route_id: simulate_route(r, deadline_days, n_runs=n_runs, seed=seed, sigma=sigma) for r in routes
    }
