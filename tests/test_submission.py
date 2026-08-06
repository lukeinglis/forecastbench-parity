"""Tests for forecastbench_parity.submission."""

from __future__ import annotations

from pathlib import Path

import pytest

from forecastbench_parity.questions import Question, ResolvedQuestion
from forecastbench_parity.submission import (
    SubmissionMetadata,
    assemble_submission,
    save_submission,
    validate_coverage,
)


def _make_resolved(qid: str, source: str, outcome: int, due: str = "2024-01-01", resolution_date: str | None = None) -> ResolvedQuestion:
    return ResolvedQuestion(id=qid, source=source, question=f"Q {qid}", outcome=outcome, forecast_due_date=due, resolution_date=resolution_date)


def _make_question(qid: str, source: str) -> Question:
    return Question(id=qid, source=source, question=f"Q {qid}")


def _make_metadata(question_set: str = "2024-01-01") -> SubmissionMetadata:
    return SubmissionMetadata(organization="test-org", model="test-model", model_organization="test-model-org", question_set=question_set)


class TestAssembleSubmission:
    def test_assemble_submission(self) -> None:
        questions = [_make_resolved("q1", "acled", 1, resolution_date="2024-02-01"), _make_resolved("q2", "metaculus", 0)]
        forecasts = {"q1": 0.8, "q2": 0.3}
        result = assemble_submission(forecasts, questions, _make_metadata())
        assert result["organization"] == "test-org"
        assert len(result["forecasts"]) == 2
        entry_q1 = next(e for e in result["forecasts"] if e["id"] == "q1")
        assert entry_q1["forecast"] == 0.8
        assert entry_q1["resolution_date"] == "2024-02-01"
        entry_q2 = next(e for e in result["forecasts"] if e["id"] == "q2")
        assert "resolution_date" not in entry_q2

    def test_missing_forecasts_default_05(self) -> None:
        questions = [_make_resolved("q1", "acled", 1), _make_resolved("q2", "acled", 0)]
        result = assemble_submission({"q1": 0.9}, questions, _make_metadata())
        entry_q2 = next(e for e in result["forecasts"] if e["id"] == "q2")
        assert entry_q2["forecast"] == 0.5


class TestValidateCoverage:
    def test_full_coverage(self) -> None:
        questions = [_make_question("d1", "acled"), _make_question("m1", "metaculus")]
        submission = {"forecasts": [{"id": "d1", "forecast": 0.8}, {"id": "m1", "forecast": 0.7}]}
        result = validate_coverage(submission, questions)
        assert result.passes is True

    def test_below_threshold(self) -> None:
        market_qs = [_make_question(f"m{i}", "metaculus") for i in range(20)]
        submission = {"forecasts": [{"id": f"m{i}", "forecast": 0.5} for i in range(18)]}
        result = validate_coverage(submission, market_qs)
        assert result.passes is False

    def test_empty_questions(self) -> None:
        submission = {"forecasts": [{"id": "q1", "forecast": 0.5}]}
        result = validate_coverage(submission, [])
        assert result.passes is False


class TestSaveSubmission:
    def test_file_naming(self, tmp_path: Path) -> None:
        submission = {"organization": "test-org", "question_set": "2024-01-01", "forecasts": []}
        path = save_submission(submission, output_dir=tmp_path)
        assert path.name == "2024-01-01.test-org.1.json"

    def test_max_three(self, tmp_path: Path) -> None:
        submission = {"organization": "test-org", "question_set": "2024-01-01", "forecasts": []}
        save_submission(submission, output_dir=tmp_path)
        save_submission(submission, output_dir=tmp_path)
        save_submission(submission, output_dir=tmp_path)
        with pytest.raises(ValueError, match="max 3"):
            save_submission(submission, output_dir=tmp_path)
