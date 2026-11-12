"""Stage 2: Pareto dominance filter (FR-4).

Route A dominates B if A is no worse than B on all four criteria and strictly
better on at least one. Dominated routes are never recommended, but they are
kept (with the dominator recorded) so the explanation can say *why* a route
was dropped.
"""

from __future__ import annotations

from collections.abc import Sequence

CRITERIA_ORDER = ("cost", "time", "reliability", "risk")


def dominates(a: dict[str, float], b: dict[str, float], tol: float = 1e-9) -> bool:
    """True if criteria dict `a` dominates `b` (all criteria lower-is-better)."""
    no_worse = all(a[k] <= b[k] + tol for k in CRITERIA_ORDER)
    strictly_better = any(a[k] < b[k] - tol for k in CRITERIA_ORDER)
    return no_worse and strictly_better


def pareto_filter(
    criteria_by_id: dict[str, dict[str, float]],
    candidate_ids: Sequence[str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Return (non_dominated_ids, {dominated_id: dominator_id}).

    Ties never eliminate each other (identical routes both survive), so a
    single route always returns itself.
    """
    ids = list(candidate_ids) if candidate_ids is not None else list(criteria_by_id)
    non_dominated: list[str] = []
    dominated_by: dict[str, str] = {}

    for rid in ids:
        dominator = None
        for other in ids:
            if other == rid:
                continue
            if dominates(criteria_by_id[other], criteria_by_id[rid]):
                dominator = other
                break
        if dominator is None:
            non_dominated.append(rid)
        else:
            dominated_by[rid] = dominator

    return non_dominated, dominated_by
