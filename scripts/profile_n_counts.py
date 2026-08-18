"""Profile N counts at each pipeline stage and compare against upstream targets.

Usage:
    python scripts/profile_n_counts.py

Requires cached data (run any scoring operation first to populate .cache/).
"""

from __future__ import annotations

import logging

from forecastbench_parity.questions import (
    fetch_all_question_sets,
    fetch_all_resolutions,
    filter_question_sets_by_age,
    join_resolved_questions,
)
from forecastbench_parity.score import _is_market_question, _scoring_key, score_forecasts

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

TARGET_N = 35_148
TARGET_INDEX = 58.4


def _count_raw_resolutions() -> int:
    """Count total resolution entries across all files (before dedup)."""
    from forecastbench_parity.questions import fetch_resolution, list_resolution_files

    total = 0
    for f in list_resolution_files():
        try:
            total += len(fetch_resolution(f))
        except Exception:
            continue
    return total


def _profile_resolved(resolved: list, label: str) -> None:
    """Print N count profile for a resolved question list."""
    n_dataset = sum(1 for q in resolved if not _is_market_question(q))
    n_market = sum(1 for q in resolved if _is_market_question(q))
    n_total = len(resolved)

    log.info("  Resolved questions: dataset=%d  market=%d  total=%d", n_dataset, n_market, n_total)

    scoring_keys = {_scoring_key(q) for q in resolved}
    dataset_keys = {_scoring_key(q) for q in resolved if not _is_market_question(q)}
    market_keys = {_scoring_key(q) for q in resolved if _is_market_question(q)}

    log.info("  Unique scoring keys: dataset=%d  market=%d  total=%d", len(dataset_keys), len(market_keys), len(scoring_keys))

    if resolved:
        result = score_forecasts({}, resolved, difficulty_adjusted=False)
        log.info("  Scored N: dataset=%d  market=%d  total=%d", result.n_dataset, result.n_market, result.n_dataset + result.n_market)
        log.info("  Overall Index: %.1f", result.overall_index)
        n_actual = result.n_dataset + result.n_market
        log.info("  vs target: N=%d (delta %+d)  Index=%.1f (delta %+.1f)",
                 n_actual, n_actual - TARGET_N, result.overall_index, result.overall_index - TARGET_INDEX)


def main() -> None:
    log.info("=== ForecastBench Parity — N Count Profile ===\n")

    raw_resolution_count = _count_raw_resolutions()
    log.info("Raw resolution entries (all files): %d", raw_resolution_count)

    question_sets = fetch_all_question_sets()
    resolutions = fetch_all_resolutions()

    deduped_resolution_count = sum(len(v) for v in resolutions.values())
    log.info("Deduplicated resolution entries:    %d", deduped_resolution_count)
    log.info(
        "Duplicates dropped:                 %d",
        raw_resolution_count - deduped_resolution_count,
    )

    log.info("\n--- All Question Sets (no date cutoff) ---")
    log.info("Question sets: %d", len(question_sets))
    resolved_all = join_resolved_questions(question_sets, resolutions)
    _profile_resolved(resolved_all, "all")

    log.info("\n--- With 365-day Date Cutoff (upstream parity) ---")
    filtered_sets = filter_question_sets_by_age(question_sets, max_age_days=365)
    log.info("Question sets after cutoff: %d (excluded %d)", len(filtered_sets), len(question_sets) - len(filtered_sets))
    if filtered_sets:
        log.info("Date range: %s to %s",
                 min(qs.forecast_due_date for qs in filtered_sets),
                 max(qs.forecast_due_date for qs in filtered_sets))
    resolved_filtered = join_resolved_questions(filtered_sets, resolutions)
    _profile_resolved(resolved_filtered, "filtered")

    log.info("\n--- Summary ---")
    log.info("Date cutoff impact: %d -> %d resolved questions (-%d)",
             len(resolved_all), len(resolved_filtered),
             len(resolved_all) - len(resolved_filtered))


if __name__ == "__main__":
    main()
