"""ForecastBench submission assembly, validation, and upload."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forecastbench_parity.constants import MARKET_SOURCES, MAX_SUBMISSIONS_PER_ROUND
from forecastbench_parity.questions import Question, ResolvedQuestion


@dataclass
class SubmissionMetadata:
    organization: str
    model: str
    model_organization: str
    question_set: str
    sequence_number: int = 1


@dataclass
class CoverageResult:
    market_coverage: float
    dataset_coverage: float
    market_total: int
    market_covered: int
    dataset_total: int
    dataset_covered: int
    passes: bool


def validate_forecasts(entries: list[dict[str, Any]]) -> None:
    """Validate forecast entries for submission compliance."""
    for entry in entries:
        prob = entry.get("forecast")
        if not isinstance(prob, (int, float)) or not math.isfinite(prob) or prob < 0 or prob > 1:
            raise ValueError(
                f"Forecast for {entry.get('id', '?')} is out of range [0, 1]: {prob}"
            )
        source = entry.get("source", "")
        if source.lower() in MARKET_SOURCES and entry.get("resolution_date") is not None:
            raise ValueError(
                f"Market question {entry.get('id', '?')} must not have resolution_date"
            )


def assemble_submission(
    forecasts: dict[str, float],
    questions: list[ResolvedQuestion],
    metadata: SubmissionMetadata,
    reasoning: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build ForecastBench-format submission JSON."""
    entries = []
    for q in questions:
        prob = forecasts.get(q.id, 0.5)
        entry: dict[str, Any] = {
            "id": q.id,
            "source": q.source,
            "forecast": prob,
        }
        if q.resolution_date and q.source.lower() not in MARKET_SOURCES:
            entry["resolution_date"] = q.resolution_date
        if reasoning and q.id in reasoning:
            entry["reasoning"] = reasoning[q.id]
        entries.append(entry)

    validate_forecasts(entries)

    return {
        "organization": metadata.organization,
        "model": metadata.model,
        "model_organization": metadata.model_organization,
        "question_set": metadata.question_set,
        "forecasts": entries,
    }


def validate_coverage(
    submission: dict[str, Any],
    questions: list[Question] | list[ResolvedQuestion],
    threshold: float = 0.95,
) -> CoverageResult:
    """Check that submission covers 95%+ of market and dataset questions."""
    forecast_ids = {f["id"] for f in submission["forecasts"]}
    market_qs = [q for q in questions if q.source.lower() in MARKET_SOURCES]
    dataset_qs = [q for q in questions if q.source.lower() not in MARKET_SOURCES]

    mkt_covered = sum(1 for q in market_qs if q.id in forecast_ids)
    ds_covered = sum(1 for q in dataset_qs if q.id in forecast_ids)
    mkt_cov = mkt_covered / len(market_qs) if market_qs else 0.0
    ds_cov = ds_covered / len(dataset_qs) if dataset_qs else 0.0

    mkt_ok = mkt_cov >= threshold if market_qs else True
    ds_ok = ds_cov >= threshold if dataset_qs else True

    return CoverageResult(
        market_coverage=mkt_cov,
        dataset_coverage=ds_cov,
        market_total=len(market_qs),
        market_covered=mkt_covered,
        dataset_total=len(dataset_qs),
        dataset_covered=ds_covered,
        passes=mkt_ok and ds_ok and bool(questions),
    )


def save_submission(
    submission: dict[str, Any],
    output_dir: Path = Path("submissions"),
) -> Path:
    """Save submission JSON to local staging directory."""
    due_date = submission["question_set"]
    org = submission["organization"]
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob(f"{due_date}.{org}.*.json"))
    n = len(existing) + 1
    if n > MAX_SUBMISSIONS_PER_ROUND:
        raise ValueError(
            f"ForecastBench allows max {MAX_SUBMISSIONS_PER_ROUND} submissions per round, "
            f"found {len(existing)} existing"
        )
    filename = f"{due_date}.{org}.{n}.json"
    path = output_dir / filename
    path.write_text(json.dumps(submission, indent=2))
    return path


def upload_to_gcs(
    path: Path,
    bucket: str,
    folder: str = "",
) -> str:
    """Upload submission to GCS bucket. Requires google-cloud-storage."""
    from google.cloud import storage

    client = storage.Client()
    blob_name = f"{folder}/{path.name}" if folder else path.name
    blob = client.bucket(bucket).blob(blob_name)
    blob.upload_from_filename(str(path))
    return f"gs://{bucket}/{blob_name}"
