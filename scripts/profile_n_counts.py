#!/usr/bin/env python3
"""Profile N counts at each pipeline stage and compare against upstream targets.

Usage:
    python scripts/profile_n_counts.py

Requires cached data (run any scoring operation first to populate .cache/).
"""

from __future__ import annotations

import logging
import sys

from forecastbench_parity.constants import MARKET_SOURCES
from forecastbench_parity.questions import (
    fetch_all_question_sets,
    fetch_all_resolutions,
    join_resolved_questions,
)
from forecastbench_parity.score import _is_market_question, _scoring_key, score_forecasts

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

TARGET_N = 35_148
TARGET_INDEX = 58.4


def _count_raw_resolutions() -> tuple[int, int]:
    """Count total resolution entries across all files (before dedup)."""
    from forecastbench_parity.questions import list_resolution_files, fetch_resolution

    total = 0
    for f in list_resolution_files():
        try:
            total += len(fetch_resolution(f))
        except Exception:
            continue
    return total


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

    resolved = join_resolved_questions(question_sets, resolutions)

    n_dataset = sum(1 for q in resolved if not _is_market_question(q))
    n_market = sum(1 for q in resolved if _is_market_question(q))
    n_total = len(resolved)

    log.info("\n--- Resolved Questions by Type ---")
    log.info("Dataset:  %d", n_dataset)
    log.info("Market:   %d", n_market)
    log.info("Total:    %d", n_total)

    scoring_keys = {_scoring_key(q) for q in resolved}
    dataset_keys = {_scoring_key(q) for q in resolved if not _is_market_question(q)}
    market_keys = {_scoring_key(q) for q in resolved if _is_market_question(q)}

    log.info("\n--- Unique Scoring Keys ---")
    log.info("Dataset keys:  %d", len(dataset_keys))
    log.info("Market keys:   %d", len(market_keys))
    log.info("Total keys:    %d", len(scoring_keys))

    log.info("\n--- Naive Forecaster (all 0.5) ---")
    result = score_forecasts({}, resolved, difficulty_adjusted=False)
    log.info("N_dataset:     %d", result.n_dataset)
    log.info("N_market:      %d", result.n_market)
    log.info("N_total:       %d", result.n_dataset + result.n_market)
    log.info("Overall Index: %.1f", result.overall_index)

    log.info("\n--- Comparison with Targets ---")
    n_actual = result.n_dataset + result.n_market
    log.info("N_total:       %d (target: %d, delta: %+d)", n_actual, TARGET_N, n_actual - TARGET_N)
    log.info(
        "Overall Index: %.1f (target: %.1f, delta: %+.1f)",
        result.overall_index,
        TARGET_INDEX,
        result.overall_index - TARGET_INDEX,
    )

    if n_actual == TARGET_N:
        log.info("\n✓ N count matches target")
    else:
        log.info("\n✗ N count does NOT match target")

    if abs(result.overall_index - TARGET_INDEX) < 0.5:
        log.info("✓ Overall index within 0.5 of target")
    else:
        log.info("✗ Overall index NOT within 0.5 of target")


if __name__ == "__main__":
    main()
