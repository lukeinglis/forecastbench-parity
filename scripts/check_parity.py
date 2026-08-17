"""Check our pipeline's N counts against the upstream ForecastBench leaderboard.

Usage:
    uv run python scripts/check_parity.py
"""

from __future__ import annotations

import sys

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
    """Find the full-coverage reference forecaster with the highest N."""
    candidates = [r for r in rows if r.get("Model", "") in REFERENCE_MODELS]
    if not candidates:
        raise RuntimeError(
            f"No reference model found in leaderboard. Looked for: {REFERENCE_MODELS}"
        )
    return max(candidates, key=lambda r: int(r.get("N", "0").replace(",", "")))


def _is_market(source: str) -> bool:
    return any(s in source.lower() for s in MARKET_SOURCES)


def _count_unique_scoring_keys(
    resolved: list,
) -> tuple[int, int]:
    """Count unique scoring keys by type (dataset, market)."""
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


def main() -> int:
    refresh_cache()

    print("Fetching upstream leaderboard...")
    rows = fetch_leaderboard("baseline")
    ref = _find_reference_row(rows)
    upstream_n_dataset = int(ref["N dataset"].replace(",", ""))
    upstream_n_market = int(ref["N market"].replace(",", ""))
    print(f"Reference model: {ref['Model']}")
    print(f"Upstream:  N_dataset={upstream_n_dataset}  N_market={upstream_n_market}")

    print("\nRunning our pipeline...")
    question_sets = fetch_all_question_sets()
    filtered = filter_question_sets_by_age(question_sets, max_age_days=365)
    resolutions = fetch_all_resolutions()
    resolved = join_resolved_questions(filtered, resolutions)
    our_n_dataset, our_n_market = _count_unique_scoring_keys(resolved)
    print(f"Ours:      N_dataset={our_n_dataset}  N_market={our_n_market}")

    dataset_delta = abs(our_n_dataset - upstream_n_dataset) / upstream_n_dataset if upstream_n_dataset else 0
    market_delta = abs(our_n_market - upstream_n_market) / upstream_n_market if upstream_n_market else 0

    print(f"\nDataset delta: {dataset_delta:.1%}  ({'PASS' if dataset_delta <= TOLERANCE else 'FAIL'})")
    print(f"Market delta:  {market_delta:.1%}  ({'PASS' if market_delta <= TOLERANCE else 'FAIL'})")

    if dataset_delta <= TOLERANCE and market_delta <= TOLERANCE:
        print("\nPARITY CHECK PASSED")
        return 0
    else:
        print("\nPARITY CHECK FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
