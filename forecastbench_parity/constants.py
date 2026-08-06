"""Competition constants for ForecastBench."""

from __future__ import annotations

MARKET_SOURCES: frozenset[str] = frozenset({"metaculus", "polymarket", "manifold", "infer"})

MISSING_FORECAST_DEFAULT: float = 0.5

COVERAGE_THRESHOLD: float = 0.95

FORECAST_RANGE: tuple[float, float] = (0.0, 1.0)

MAX_SUBMISSIONS_PER_ROUND: int = 3

REPO_OWNER: str = "forecastingresearch"
REPO_NAME: str = "forecastbench-datasets"
RAW_BASE: str = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/main/datasets"
API_BASE: str = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/datasets"
LEADERBOARD_BASE: str = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/main/leaderboards/csv"
LEADERBOARD_NAMES: frozenset[str] = frozenset({"baseline", "tournament", "dataset", "preliminary"})
