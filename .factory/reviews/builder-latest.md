# Builder Report: Scoring Bugs, Lint, CI & Hygiene

## Bug Fixes

### 1a. Overall formula (score.py)
- **Before:** Count-weighted pooling `(ds * n_ds + mk * n_mk) / total`
- **After:** Equal-weight average `(ds + mk) / 2.0`, matching upstream ForecastBench

### 1b. Per-question clamp (score.py)
- **Before:** `max(0.0, min(1.0, val + shift))` — clamped adjusted scores to [0, 1]
- **After:** `val + shift` — no clamp, matching upstream

### 1c. Single global shift (score.py)
- **Before:** One global shift computed across all questions
- **After:** Per-column shifts — separate shifts for dataset and market questions, matching upstream which computes independent calibration per column

## Lint Fixes

1. `questions.py` — replaced bare `except Exception: pass` with specific exception types and `_logger.warning()`
2. `score.py:179` — changed unused `forecaster_id` loop variable to `.values()`
3. `verify_parity.py:55,94` — sorted imports alphabetically
4. `tests/test_score.py:8` — removed unused `settings` import

## Tests Added

- `test_overall_diverges_on_unbalanced_split` — 3 dataset + 1 market; asserts equal-weight and verifies it differs from count-weighted
- `test_difficulty_adjustment_no_clamp` — verifies adjusted scores are not clamped to [0, 1]
- `test_per_column_shifts` — verifies dataset and market questions get independent shifts

All existing tests preserved. `test_overall_is_average_of_components` now passes with the fixed formula.

## Project Hygiene

- `pyproject.toml`: added `[build-system]` (hatchling), bumped version to 0.1.2
- `LICENSE`: MIT, Copyright 2026 Luke Inglis
- `.github/workflows/ci.yml`: Python 3.11/3.12/3.13 matrix, ruff + pytest

## Verification

- `uv run ruff check .` — clean (0 errors)
- `uv run pytest -v` — 72 passed
