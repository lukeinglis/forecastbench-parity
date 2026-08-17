"""ForecastBench question sets, resolutions, and data fetching."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel, field_validator

from forecastbench_parity.constants import (
    API_BASE,
    LEADERBOARD_BASE,
    LEADERBOARD_NAMES,
    RAW_BASE,
)

_logger = logging.getLogger(__name__)

CACHE_DIR = Path(".cache")


class Question(BaseModel):
    id: str
    source: str
    question: str
    background: str = ""
    resolution_criteria: str = ""
    freeze_datetime: str | None = None
    freeze_datetime_value: float | None = None
    resolution_dates: Any = None
    url: str | None = None
    combination_of: list[str] | None = None
    source_intro: str | None = None
    freeze_datetime_value_explanation: str | None = None
    market_info_open_datetime: str | None = None
    market_info_close_datetime: str | None = None
    market_info_resolution_criteria: str | None = None
    forecast_due_date: str | None = None

    @field_validator("id", mode="before")
    @classmethod
    def _coerce_id(cls, v: Any) -> str:
        if isinstance(v, list):
            return "|".join(str(x) for x in v)
        return str(v)

    @field_validator("freeze_datetime_value", mode="before")
    @classmethod
    def _coerce_freeze_value(cls, v: Any) -> float | None:
        if v is None:
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    @field_validator("combination_of", mode="before")
    @classmethod
    def _coerce_combination_of(cls, v: Any) -> list[str] | None:
        if v is None or (isinstance(v, str) and v.upper() == "N/A"):
            return None
        if isinstance(v, list):
            return [x["id"] if isinstance(x, dict) else str(x) for x in v]
        return list(v) if isinstance(v, (list, tuple)) else None


class QuestionSet(BaseModel):
    forecast_due_date: str
    question_set: str = ""
    questions: list[Question]


class Resolution(BaseModel):
    id: str
    outcome: int | None = None
    resolution_date: str | None = None
    resolved: bool | None = None

    @field_validator("id", mode="before")
    @classmethod
    def _coerce_id(cls, v: Any) -> str:
        if isinstance(v, list):
            return "|".join(str(x) for x in v)
        return str(v)

    @classmethod
    def model_validate(cls, obj: Any, **kwargs: Any) -> Resolution:
        if isinstance(obj, dict) and "outcome" not in obj and "resolved_to" in obj:
            obj = dict(obj)
            val = obj.pop("resolved_to", None)
            obj["outcome"] = round(val) if val is not None else None
        return super().model_validate(obj, **kwargs)


class ResolvedQuestion(BaseModel):
    id: str
    source: str
    question: str
    background: str = ""
    resolution_criteria: str = ""
    freeze_datetime: str | None = None
    freeze_datetime_value: float | None = None
    resolution_dates: Any = None
    url: str | None = None
    combination_of: list[str] | None = None
    source_intro: str | None = None
    freeze_datetime_value_explanation: str | None = None
    market_info_open_datetime: str | None = None
    market_info_close_datetime: str | None = None
    market_info_resolution_criteria: str | None = None
    outcome: int
    resolution_date: str | None = None
    forecast_due_date: str = ""
    question_set: str = ""


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(filename: str) -> Path:
    return CACHE_DIR / filename.replace("/", "_")


def _fetch_json(url: str, cache_key: str) -> Any:
    _ensure_cache_dir()
    cached = _cache_path(cache_key)
    if cached.exists():
        return json.loads(cached.read_text())
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    cached.write_text(json.dumps(data))
    return data


def _fetch_text(url: str, cache_key: str) -> str:
    _ensure_cache_dir()
    cached = _cache_path(cache_key)
    if cached.exists():
        return cached.read_text()
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    text = resp.text
    cached.write_text(text)
    return text


def list_question_set_files() -> list[str]:
    """List available question set JSON filenames from the GitHub repo."""
    data = _fetch_json(f"{API_BASE}/question_sets", "question_sets_listing.json")
    return [
        item["name"]
        for item in data
        if item["name"].endswith(".json") and item["name"] != "latest-llm.json"
    ]


def get_latest_round() -> str:
    """Get the name of the current/latest round from the ForecastBench repo."""
    url = f"{RAW_BASE}/question_sets/latest-llm.json"
    text = _fetch_text(url, "latest_round.txt")
    return text.strip().replace(".json", "")


def list_resolution_files() -> list[str]:
    """List available resolution JSON filenames from the GitHub repo."""
    data = _fetch_json(f"{API_BASE}/resolution_sets", "resolution_sets_listing.json")
    return [item["name"] for item in data if item["name"].endswith(".json")]


def fetch_question_set(filename: str) -> QuestionSet:
    """Fetch and parse a single question set file."""
    url = f"{RAW_BASE}/question_sets/{filename}"
    data = _fetch_json(url, f"qs_{filename}")
    return QuestionSet.model_validate(data)


def fetch_resolution(filename: str) -> list[Resolution]:
    """Fetch and parse a single resolution file."""
    url = f"{RAW_BASE}/resolution_sets/{filename}"
    data = _fetch_json(url, f"res_{filename}")
    if isinstance(data, list):
        return [Resolution.model_validate(r) for r in data]
    if isinstance(data, dict) and "resolutions" in data:
        return [Resolution.model_validate(r) for r in data["resolutions"]]
    return [Resolution.model_validate(data)]


def fetch_all_question_sets() -> list[QuestionSet]:
    """Fetch all available question sets."""
    filenames = list_question_set_files()
    result = []
    for f in filenames:
        try:
            qs = fetch_question_set(f)
            result.append(qs)
        except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError) as e:
            _logger.warning("Failed to fetch %s: %s", f, e)
            continue
    return result


def fetch_all_resolutions() -> dict[str, list[Resolution]]:
    """Fetch all resolutions, returning a dict keyed by question id.

    Each question id maps to a list of Resolution entries so that
    multi-horizon questions (which appear once per resolution_date)
    are all preserved.
    """
    filenames = list_resolution_files()
    resolutions: dict[str, list[Resolution]] = {}
    for f in filenames:
        try:
            res_list = fetch_resolution(f)
            for r in res_list:
                resolutions.setdefault(r.id, []).append(r)
        except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError) as e:
            _logger.warning("Failed to fetch %s: %s", f, e)
            continue
    return resolutions


def join_resolved_questions(
    question_sets: list[QuestionSet],
    resolutions: dict[str, list[Resolution]],
) -> list[ResolvedQuestion]:
    """Join questions with their resolutions, returning only resolved questions.

    Each question is expanded across all of its Resolution entries so that
    multi-horizon questions produce one ResolvedQuestion per resolution_date.
    """
    resolved = []
    total_seen = 0
    skipped_null_date = 0
    skipped_invalid_date = 0
    skipped_no_outcome = 0
    skipped_unresolved = 0
    for qs in question_sets:
        for q in qs.questions:
            for r in resolutions.get(q.id, []):
                total_seen += 1
                if isinstance(q.resolution_dates, list):
                    if r.resolution_date is None or r.resolution_date == "N/A":
                        skipped_null_date += 1
                        continue
                    if r.resolution_date not in q.resolution_dates:
                        skipped_invalid_date += 1
                        continue
                if r.outcome is None:
                    skipped_no_outcome += 1
                    continue
                if getattr(r, "resolved", None) is False:
                    skipped_unresolved += 1
                    continue
                resolved.append(
                    ResolvedQuestion(
                        id=q.id,
                        source=q.source,
                        question=q.question,
                        background=q.background,
                        resolution_criteria=q.resolution_criteria,
                        freeze_datetime=q.freeze_datetime,
                        freeze_datetime_value=q.freeze_datetime_value,
                        resolution_dates=q.resolution_dates,
                        url=q.url,
                        combination_of=q.combination_of,
                        source_intro=q.source_intro,
                        freeze_datetime_value_explanation=q.freeze_datetime_value_explanation,
                        market_info_open_datetime=q.market_info_open_datetime,
                        market_info_close_datetime=q.market_info_close_datetime,
                        market_info_resolution_criteria=q.market_info_resolution_criteria,
                        outcome=r.outcome,
                        resolution_date=r.resolution_date,
                        forecast_due_date=qs.forecast_due_date,
                        question_set=qs.question_set,
                    )
                )
    _logger.debug(
        "join_resolved_questions: total_resolutions_seen=%d skipped_null_date=%d "
        "skipped_invalid_date=%d skipped_no_outcome=%d skipped_unresolved=%d kept_count=%d",
        total_seen,
        skipped_null_date,
        skipped_invalid_date,
        skipped_no_outcome,
        skipped_unresolved,
        len(resolved),
    )
    return resolved


def fetch_leaderboard(name: str = "baseline") -> list[dict[str, str]]:
    """Fetch a leaderboard CSV and return as list of dicts."""
    import csv
    import io

    if name not in LEADERBOARD_NAMES:
        raise ValueError(
            f"Unknown leaderboard {name!r}, expected one of {sorted(LEADERBOARD_NAMES)}"
        )
    url = f"{LEADERBOARD_BASE}/leaderboard_{name}.csv"
    text = _fetch_text(url, f"lb_{name}.csv")
    reader = csv.DictReader(io.StringIO(text))
    return [dict(row) for row in reader]


def fetch_superforecaster_forecasts() -> list[dict[str, object]]:
    """Fetch individual superforecaster forecasts from the July 2024 round."""
    url = (
        "https://media.githubusercontent.com/media/forecastingresearch/"
        "forecastbench-datasets/main/datasets/forecast_sets/"
        "2024-07-21/2024-07-21.ForecastBench.human_super_individual.json"
    )
    data = _fetch_json(url, "superforecaster_individual.json")
    result: list[dict[str, object]] = data.get("forecasts", [])
    return result


def superforecaster_medians(forecasts: list[dict[str, object]]) -> dict[str, float]:
    """Compute median forecast per question from individual superforecaster entries."""
    from statistics import median

    by_question: dict[str, list[float]] = {}
    for entry in forecasts:
        qid = str(entry["id"])
        prob: Any = entry.get("forecast")
        if prob is not None:
            by_question.setdefault(qid, []).append(float(prob))
    return {qid: median(probs) for qid, probs in by_question.items() if probs}


def refresh_cache() -> None:
    """Delete volatile cache files so next fetch pulls fresh data."""
    if not CACHE_DIR.exists():
        return
    patterns = ["question_sets_listing.json", "resolution_sets_listing.json", "latest_round.txt"]
    for name in patterns:
        path = CACHE_DIR / name
        if path.exists():
            path.unlink()
    for path in CACHE_DIR.glob("res_*"):
        path.unlink()
    for path in CACHE_DIR.glob("lb_*"):
        path.unlink()


def load_data() -> tuple[list[QuestionSet], list[ResolvedQuestion]]:
    """Main entry point: fetch all data and return question sets + resolved questions."""
    question_sets = fetch_all_question_sets()
    resolutions = fetch_all_resolutions()
    resolved = join_resolved_questions(question_sets, resolutions)
    return question_sets, resolved
