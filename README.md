# forecastbench-parity

> **Warning:** Versions prior to v0.1.2 (including v0.1.1) produce incorrect scores due to three bugs: count-weighted overall pooling instead of equal-weight, clamped difficulty adjustments, and a single global calibration shift instead of per-column shifts. Upgrade to v0.1.2+ for correct results.

Frozen competition contract for [ForecastBench](https://www.forecastbench.org/): scoring, submission format, and question handling.

This package contains the immutable parts of the ForecastBench competition interface -- the code that MUST match the official specification regardless of forecasting methodology. It is extracted from the backtester to enforce a clean boundary between "what the competition requires" (this package) and "how we forecast" (the backtester).

## What's included

- **Scoring** (`forecastbench_parity.score`): Brier score, Brier Index, difficulty adjustment, bootstrap CI, Murphy decomposition
- **Submission** (`forecastbench_parity.submission`): Validation, assembly, coverage checking, GCS upload
- **Questions** (`forecastbench_parity.questions`): Pydantic models, field validators, GitHub data fetching, resolution matching
- **Constants** (`forecastbench_parity.constants`): MARKET_SOURCES, thresholds, repo URLs

## What's NOT included

Prompts, temperature, thinking mode, parsing, RAG, ensemble, calibration, model selection -- these are all methodology choices that each team customizes. The competition rules are maximally permissive on methodology.

## Install

```bash
pip install forecastbench-parity@git+https://github.com/lukeinglis/forecastbench-parity.git
```

## Usage

```python
from forecastbench_parity import (
    brier_score, brier_index, score_forecasts,
    Question, ResolvedQuestion, load_data,
    MARKET_SOURCES,
)

# Score forecasts
result = score_forecasts(forecasts, resolved_questions)
print(f"Overall Brier Index: {result.overall_index:.1f}%")
```

## Key formulas

- **Brier Score**: `(forecast - outcome)^2`
- **Brier Index**: `(1 - sqrt(mean_brier_score)) * 100`, applied AFTER averaging
- **Missing forecasts**: Default to 0.5 per ForecastBench rules
- **Binary outcomes**: {0, 1} only

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Parity Validation

This project aims to match the official [ForecastBench tournament](https://www.forecastbench.org/) scoring pipeline exactly.

- **Quick check** — compare N counts against the live upstream leaderboard:
  ```bash
  uv run python scripts/check_parity.py
  ```
- **Slow parity test** — runs the full pipeline and asserts against upstream:
  ```bash
  uv run pytest -m slow
  ```
- **Current status:** dataset at parity (~1.7% delta), market has a known gap (see [#10](https://github.com/lukeinglis/forecastbench-parity/issues/10))

## Upstream Alignment

Key structural choices that match the upstream scoring pipeline:

- **Resolution deduplication** by `(question_id, resolution_date)` — drops duplicate resolution entries before scoring
- **Scoring key format:** `{forecast_due_date}_{source}_{id}[_{resolution_date}]` — mirrors upstream's `question_pk`
- **8 standard forecast horizons** for dataset questions: 7, 30, 90, 180, 365, 1095, 1825, 3650 days (`FORECAST_HORIZONS_IN_DAYS`)
- **365-day question set date cutoff** — matches upstream's `MODEL_RELEASE_DAYS_CUTOFF`
- **`filter_question_sets_by_age()`** applies the date cutoff to exclude old question sets
- **Market questions are NOT horizon-filtered** — they use their native resolution dates

## Data Profiling

- **Run:** `uv run python scripts/profile_n_counts.py`
- **Reports:** N counts at each pipeline stage with/without the 365-day date cutoff
