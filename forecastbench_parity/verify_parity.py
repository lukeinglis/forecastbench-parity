"""Verify forecastbench-parity matches official ForecastBench competition rules."""

from __future__ import annotations

import math
import sys


def verify_constants() -> list[str]:
    """Verify competition constants match official values."""
    from forecastbench_parity.constants import (
        COVERAGE_THRESHOLD,
        MARKET_SOURCES,
        MISSING_FORECAST_DEFAULT,
    )

    errors: list[str] = []

    expected_sources = frozenset({"metaculus", "polymarket", "manifold", "infer"})
    if MARKET_SOURCES != expected_sources:
        errors.append(f"MARKET_SOURCES mismatch: {MARKET_SOURCES} != {expected_sources}")

    if MISSING_FORECAST_DEFAULT != 0.5:
        errors.append(f"MISSING_FORECAST_DEFAULT={MISSING_FORECAST_DEFAULT}, expected 0.5")

    if COVERAGE_THRESHOLD != 0.95:
        errors.append(f"COVERAGE_THRESHOLD={COVERAGE_THRESHOLD}, expected 0.95")

    return errors


def verify_brier_index_formula() -> list[str]:
    """Verify Brier Index = (1 - sqrt(mean_brier)) * 100."""
    from forecastbench_parity.score import brier_index

    errors: list[str] = []

    test_cases = [
        (0.0, 100.0),
        (0.25, (1.0 - math.sqrt(0.25)) * 100.0),
        (1.0, 0.0),
        (0.04, (1.0 - math.sqrt(0.04)) * 100.0),
    ]

    for mean_bs, expected_bi in test_cases:
        result = brier_index(mean_bs)
        if not math.isclose(result, expected_bi, rel_tol=1e-9):
            errors.append(f"brier_index({mean_bs})={result}, expected {expected_bi}")

    return errors


def verify_scoring() -> list[str]:
    """Verify scoring produces correct results for known test cases."""
    from forecastbench_parity.score import brier_index, brier_score, mean_brier_score

    errors: list[str] = []

    bs = brier_score(0.9, 1)
    expected = (0.9 - 1) ** 2
    if not math.isclose(bs, expected, rel_tol=1e-9):
        errors.append(f"brier_score(0.9, 1)={bs}, expected {expected}")

    bs_wrong = brier_score(0.1, 1)
    expected_wrong = (0.1 - 1) ** 2
    if not math.isclose(bs_wrong, expected_wrong, rel_tol=1e-9):
        errors.append(f"brier_score(0.1, 1)={bs_wrong}, expected {expected_wrong}")

    pairs = [(0.5, 0), (0.5, 1)]
    mbs = mean_brier_score(pairs)
    if not math.isclose(mbs, 0.25, rel_tol=1e-9):
        errors.append(f"mean_brier_score(always-0.5)={mbs}, expected 0.25")

    bi = brier_index(mbs)
    expected_bi = (1.0 - math.sqrt(0.25)) * 100.0
    if not math.isclose(bi, expected_bi, rel_tol=1e-9):
        errors.append(f"brier_index after mean={bi}, expected {expected_bi}")

    perfect_pairs = [(1.0, 1), (0.0, 0)]
    perfect_mbs = mean_brier_score(perfect_pairs)
    if not math.isclose(perfect_mbs, 0.0, abs_tol=1e-12):
        errors.append(f"Perfect forecaster mean_brier={perfect_mbs}, expected 0.0")

    worst_pairs = [(0.0, 1), (1.0, 0)]
    worst_mbs = mean_brier_score(worst_pairs)
    if not math.isclose(worst_mbs, 1.0, rel_tol=1e-9):
        errors.append(f"Worst forecaster mean_brier={worst_mbs}, expected 1.0")

    return errors


def verify_missing_forecast_default() -> list[str]:
    """Verify score_forecasts defaults missing forecasts to 0.5."""
    from forecastbench_parity.questions import ResolvedQuestion
    from forecastbench_parity.score import score_forecasts

    errors: list[str] = []

    resolved = [
        ResolvedQuestion(
            id="q1", source="fred", question="test", outcome=1,
            resolution_date="2024-01-01",
        ),
    ]

    result = score_forecasts({}, resolved, difficulty_adjusted=False)
    expected_bs = (0.5 - 1) ** 2
    if not math.isclose(result.overall_brier, expected_bs, rel_tol=1e-9):
        errors.append(
            f"Missing forecast brier={result.overall_brier}, expected {expected_bs} (default=0.5)"
        )
    if result.n_missing != 1:
        errors.append(f"n_missing={result.n_missing}, expected 1")

    return errors


def run_all() -> bool:
    """Run all verification checks. Returns True if all pass."""
    checks = [
        ("Constants", verify_constants),
        ("Brier Index formula", verify_brier_index_formula),
        ("Scoring correctness", verify_scoring),
        ("Missing forecast default", verify_missing_forecast_default),
    ]

    all_pass = True
    for name, check_fn in checks:
        errors = check_fn()
        if errors:
            print(f"FAIL: {name}")
            for e in errors:
                print(f"  - {e}")
            all_pass = False
        else:
            print(f"PASS: {name}")

    return all_pass


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
