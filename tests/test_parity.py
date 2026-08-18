"""Parity check: verify our N counts match the upstream leaderboard.

Marked slow because it fetches live data from GitHub.
"""

from __future__ import annotations

import pytest

from forecastbench_parity.constants import MARKET_SOURCES
from forecastbench_parity.questions import (
    fetch_all_question_sets,
    fetch_all_resolutions,
    fetch_leaderboard,
    filter_question_sets_by_age,
    join_resolved_questions,
    refresh_cache,
)

REFERENCE_MODELS = ["Always 0.5", "Always 0", "Always 1", "Naive Forecaster", "Imputed Forecaster"]
TOLERANCE = 0.05


def _find_reference_row(rows: list[dict[str, str]]) -> dict[str, str]:
    candidates = [r for r in rows if r.get("Model", "") in REFERENCE_MODELS]
    assert candidates, f"No reference model found. Looked for: {REFERENCE_MODELS}"
    return max(candidates, key=lambda r: int(r.get("N", "0").replace(",", "")))


def _is_market(source: str) -> bool:
    return any(s in source.lower() for s in MARKET_SOURCES)


def _count_unique_scoring_keys(resolved: list) -> tuple[int, int]:
    dataset_keys: set[str] = set()
    market_keys: set[str] = set()
    for q in resolved:
        if _is_market(q.source):
            key = f"{q.forecast_due_date}_{q.source}_{q.id}"
        else:
            if q.resolution_date and q.resolution_date != "N/A":
                key = f"{q.forecast_due_date}_{q.source}_{q.id}_{q.resolution_date}"
            else:
                key = f"{q.forecast_due_date}_{q.source}_{q.id}"
        if _is_market(q.source):
            market_keys.add(key)
        else:
            dataset_keys.add(key)
    return len(dataset_keys), len(market_keys)


@pytest.mark.slow
def test_parity_n_counts():
    refresh_cache()

    rows = fetch_leaderboard("baseline")
    ref = _find_reference_row(rows)
    upstream_n_dataset = int(ref["N dataset"].replace(",", ""))
    upstream_n_market = int(ref["N market"].replace(",", ""))

    question_sets = fetch_all_question_sets()
    filtered = filter_question_sets_by_age(question_sets, max_age_days=365)
    resolutions = fetch_all_resolutions()
    resolved = join_resolved_questions(filtered, resolutions)
    our_n_dataset, our_n_market = _count_unique_scoring_keys(resolved)

    dataset_delta = abs(our_n_dataset - upstream_n_dataset) / upstream_n_dataset if upstream_n_dataset else 0
    market_delta = abs(our_n_market - upstream_n_market) / upstream_n_market if upstream_n_market else 0

    assert dataset_delta <= TOLERANCE, (
        f"Dataset N mismatch: ours={our_n_dataset} upstream={upstream_n_dataset} delta={dataset_delta:.1%}"
    )
    assert market_delta <= TOLERANCE, (
        f"Market N mismatch: ours={our_n_market} upstream={upstream_n_market} delta={market_delta:.1%}"
    )
