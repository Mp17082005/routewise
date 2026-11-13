"""Stage 3: normalisation and weighted scoring.

All four criteria are lower-is-better and are min-max normalised *within the
shipment's own option set*, because the decision is always a comparison among
the real alternatives for this shipment — not against a global scale.

Known weakness, stated in the report: min-max exaggerates differences when
options are nearly identical. Mitigated by the flat-spread rule below (a
criterion whose spread is under 2% of its minimum is treated as flat) and by
reporting the winner's score margin.
"""

from __future__ import annotations

from .config import CRITERIA


def normalise(
    values_by_id: dict[str, dict[str, float]],
    flat_spread_ratio: float = 0.02,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {rid: {} for rid in values_by_id}
    for crit in CRITERIA:
        vals = [v[crit] for v in values_by_id.values()]
        lo, hi = min(vals), max(vals)
        spread = hi - lo
        # Flat criterion: everything equal, or differences too small to trust.
        flat = spread <= 1e-12 or spread < flat_spread_ratio * max(abs(lo), 1e-12)
        for rid, v in values_by_id.items():
            out[rid][crit] = 0.0 if flat else (v[crit] - lo) / spread
    return out


def score(
    normalised: dict[str, dict[str, float]], weights: dict[str, float]
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Return (score_by_id, contribution_by_id) — lowest score wins."""
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"weights must sum to 1.0, got {total}")

    scores: dict[str, float] = {}
    contributions: dict[str, dict[str, float]] = {}
    for rid, norms in normalised.items():
        contrib = {c: weights[c] * norms[c] for c in CRITERIA}
        contributions[rid] = contrib
        scores[rid] = sum(contrib.values())
    return scores, contributions


def rank(scores: dict[str, float]) -> list[str]:
    """Best first. Route id breaks ties, so ranking is deterministic and
    independent of the order routes were supplied in."""
    return sorted(scores, key=lambda rid: (scores[rid], rid))
