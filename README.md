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
pip install forecastbench-parity@git+https://github.com/lukeinglis/forecastbench-parity.git@v0.1.2
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
