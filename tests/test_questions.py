"""Tests for forecastbench_parity.questions."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from forecastbench_parity.questions import (
    Question,
    QuestionSet,
    Resolution,
    _cache_path,
    _fetch_json,
    fetch_question_set,
    fetch_resolution,
    join_resolved_questions,
    refresh_cache,
)


class TestResolutionModelValidate:
    def test_resolved_to_rounds_up(self) -> None:
        r = Resolution.model_validate({"id": "q1", "resolved_to": 0.7})
        assert r.outcome == 1

    def test_resolved_to_rounds_down(self) -> None:
        r = Resolution.model_validate({"id": "q2", "resolved_to": 0.3})
        assert r.outcome == 0

    def test_resolved_to_exact_one(self) -> None:
        r = Resolution.model_validate({"id": "q3", "resolved_to": 1.0})
        assert r.outcome == 1

    def test_resolved_to_exact_zero(self) -> None:
        r = Resolution.model_validate({"id": "q4", "resolved_to": 0.0})
        assert r.outcome == 0

    def test_direct_outcome_field(self) -> None:
        r = Resolution.model_validate({"id": "q5", "outcome": 1})
        assert r.outcome == 1

    def test_resolved_to_none(self) -> None:
        r = Resolution.model_validate({"id": "q9", "resolved_to": None})
        assert r.outcome is None

    def test_no_outcome_and_no_resolved_to(self) -> None:
        r = Resolution.model_validate({"id": "q10"})
        assert r.outcome is None

    def test_outcome_takes_precedence_over_resolved_to(self) -> None:
        r = Resolution.model_validate({"id": "q7", "outcome": 0, "resolved_to": 0.9})
        assert r.outcome == 0


class TestResolutionCoerceId:
    def test_list_id_joined(self) -> None:
        r = Resolution.model_validate({"id": ["abc", "def"], "outcome": 1})
        assert r.id == "abc|def"

    def test_string_id_passthrough(self) -> None:
        r = Resolution.model_validate({"id": "simple", "outcome": 0})
        assert r.id == "simple"


class TestQuestionCoerceId:
    def test_list_id(self) -> None:
        q = Question(id=["abc", "def"], source="acled", question="Test")  # type: ignore[arg-type]
        assert q.id == "abc|def"

    def test_string_id(self) -> None:
        q = Question(id="simple_string", source="acled", question="Test")
        assert q.id == "simple_string"


class TestQuestionCoerceCombinationOf:
    def test_dict_in_list(self) -> None:
        q = Question(id="q1", source="acled", question="Test", combination_of=[{"id": "q1"}, {"id": "q2"}])  # type: ignore[list-item]
        assert q.combination_of == ["q1", "q2"]

    def test_none(self) -> None:
        q = Question(id="q1", source="acled", question="Test", combination_of=None)
        assert q.combination_of is None

    def test_na_string(self) -> None:
        q = Question(id="q1", source="acled", question="Test", combination_of="N/A")  # type: ignore[arg-type]
        assert q.combination_of is None


class TestFetchResolutionShapes:
    @patch("forecastbench_parity.questions._fetch_json")
    def test_list_shape(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = [{"id": "r1", "outcome": 1}, {"id": "r2", "outcome": 0}]
        result = fetch_resolution("test.json")
        assert len(result) == 2
        assert result[0].id == "r1"

    @patch("forecastbench_parity.questions._fetch_json")
    def test_dict_with_resolutions_key(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = {"resolutions": [{"id": "r1", "resolved_to": 0.8}, {"id": "r2", "resolved_to": 0.2}]}
        result = fetch_resolution("test.json")
        assert len(result) == 2
        assert result[0].outcome == 1

    @patch("forecastbench_parity.questions._fetch_json")
    def test_single_object(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = {"id": "r1", "outcome": 1}
        result = fetch_resolution("test.json")
        assert len(result) == 1


class TestFetchJsonCaching:
    def test_first_call_fetches_and_caches(self, tmp_path: Path) -> None:
        payload = {"key": "value"}
        mock_resp = MagicMock()
        mock_resp.json.return_value = payload
        mock_resp.raise_for_status = MagicMock()
        with patch("forecastbench_parity.questions.CACHE_DIR", tmp_path), patch("forecastbench_parity.questions.requests.get", return_value=mock_resp) as mock_get:
            result = _fetch_json("https://example.com/data.json", "test_cache.json")
        assert result == payload
        mock_get.assert_called_once()
        assert (tmp_path / "test_cache.json").exists()

    def test_second_call_uses_cache(self, tmp_path: Path) -> None:
        payload = {"cached": True}
        cache_file = tmp_path / "cached.json"
        cache_file.write_text(json.dumps(payload))
        with patch("forecastbench_parity.questions.CACHE_DIR", tmp_path), patch("forecastbench_parity.questions.requests.get") as mock_get:
            result = _fetch_json("https://example.com/data.json", "cached.json")
        assert result == payload
        mock_get.assert_not_called()

    def test_slash_in_cache_key_replaced(self) -> None:
        result = _cache_path("some/nested/key.json")
        assert "/" not in result.name


class TestNetworkFailures:
    @patch("forecastbench_parity.questions._fetch_json")
    def test_connection_error_propagates(self, mock_fetch: MagicMock) -> None:
        mock_fetch.side_effect = requests.exceptions.ConnectionError("Connection refused")
        with pytest.raises(requests.exceptions.ConnectionError):
            fetch_question_set("test.json")

    @patch("forecastbench_parity.questions._fetch_json")
    def test_timeout_propagates(self, mock_fetch: MagicMock) -> None:
        mock_fetch.side_effect = requests.exceptions.Timeout("Read timed out")
        with pytest.raises(requests.exceptions.Timeout):
            fetch_question_set("test.json")


class TestResolvedFieldFiltering:
    def _make_qs(self, question_id: str = "q1") -> QuestionSet:
        return QuestionSet(
            forecast_due_date="2024-06-01",
            question_set="round_1",
            questions=[Question(id=question_id, source="metaculus", question="Will X?")],
        )

    def test_resolved_false_excluded(self) -> None:
        qs = self._make_qs()
        resolutions: dict[str, list[Resolution]] = {
            "q1": [Resolution(id="q1", outcome=1, resolved=False)],
        }
        result = join_resolved_questions([qs], resolutions)
        assert len(result) == 0

    def test_resolved_true_included(self) -> None:
        qs = self._make_qs()
        resolutions: dict[str, list[Resolution]] = {
            "q1": [Resolution(id="q1", outcome=1, resolved=True)],
        }
        result = join_resolved_questions([qs], resolutions)
        assert len(result) == 1

    def test_resolved_none_included(self) -> None:
        qs = self._make_qs()
        resolutions: dict[str, list[Resolution]] = {
            "q1": [Resolution(id="q1", outcome=0, resolved=None)],
        }
        result = join_resolved_questions([qs], resolutions)
        assert len(result) == 1


class TestMultiHorizonResolutions:
    def _make_qs(self, question_id: str = "q1") -> QuestionSet:
        return QuestionSet(
            forecast_due_date="2024-06-01",
            question_set="round_1",
            questions=[Question(id=question_id, source="acled", question="Will X?")],
        )

    def test_multi_horizon_produces_multiple_resolved_questions(self) -> None:
        qs = self._make_qs()
        resolutions: dict[str, list[Resolution]] = {
            "q1": [
                Resolution(id="q1", outcome=1, resolution_date="2024-07-01", resolved=True),
                Resolution(id="q1", outcome=0, resolution_date="2024-08-01", resolved=True),
                Resolution(id="q1", outcome=1, resolution_date="2024-09-01", resolved=True),
            ],
        }
        result = join_resolved_questions([qs], resolutions)
        assert len(result) == 3
        dates = [r.resolution_date for r in result]
        assert dates == ["2024-07-01", "2024-08-01", "2024-09-01"]
        assert [r.outcome for r in result] == [1, 0, 1]

    def test_single_horizon_produces_one_resolved_question(self) -> None:
        qs = self._make_qs()
        resolutions: dict[str, list[Resolution]] = {
            "q1": [Resolution(id="q1", outcome=0, resolution_date="N/A", resolved=True)],
        }
        result = join_resolved_questions([qs], resolutions)
        assert len(result) == 1
        assert result[0].outcome == 0

    def test_resolved_false_excluded_in_multi_horizon(self) -> None:
        qs = self._make_qs()
        resolutions: dict[str, list[Resolution]] = {
            "q1": [
                Resolution(id="q1", outcome=1, resolution_date="2024-07-01", resolved=True),
                Resolution(id="q1", outcome=0, resolution_date="2024-08-01", resolved=False),
                Resolution(id="q1", outcome=1, resolution_date="2024-09-01", resolved=True),
            ],
        }
        result = join_resolved_questions([qs], resolutions)
        assert len(result) == 2
        dates = [r.resolution_date for r in result]
        assert "2024-08-01" not in dates

    def test_outcome_none_excluded_in_multi_horizon(self) -> None:
        qs = self._make_qs()
        resolutions: dict[str, list[Resolution]] = {
            "q1": [
                Resolution(id="q1", outcome=1, resolution_date="2024-07-01", resolved=True),
                Resolution(id="q1", outcome=None, resolution_date="2024-08-01", resolved=True),
            ],
        }
        result = join_resolved_questions([qs], resolutions)
        assert len(result) == 1
        assert result[0].resolution_date == "2024-07-01"

    def test_empty_resolution_list_produces_nothing(self) -> None:
        qs = self._make_qs()
        resolutions: dict[str, list[Resolution]] = {"q1": []}
        result = join_resolved_questions([qs], resolutions)
        assert len(result) == 0

    def test_missing_question_id_produces_nothing(self) -> None:
        qs = self._make_qs()
        resolutions: dict[str, list[Resolution]] = {}
        result = join_resolved_questions([qs], resolutions)
        assert len(result) == 0


class TestMultiHorizonResolutionDateFilter:
    """Tests for resolution_date filtering in join_resolved_questions (upstream parity)."""

    def test_multi_horizon_filters_by_resolution_dates(self) -> None:
        qs = QuestionSet(
            forecast_due_date="2024-06-01",
            question_set="round_1",
            questions=[Question(
                id="q1", source="acled", question="Will X?",
                resolution_dates=["2024-07-28", "2024-08-20"],
            )],
        )
        resolutions: dict[str, list[Resolution]] = {
            "q1": [
                Resolution(id="q1", outcome=1, resolution_date="2024-07-28", resolved=True),
                Resolution(id="q1", outcome=0, resolution_date="2024-08-20", resolved=True),
                Resolution(id="q1", outcome=1, resolution_date="2024-09-01", resolved=True),
            ],
        }
        result = join_resolved_questions([qs], resolutions)
        assert len(result) == 2
        dates = {r.resolution_date for r in result}
        assert dates == {"2024-07-28", "2024-08-20"}
        assert "2024-09-01" not in dates

    def test_single_horizon_na_matches_all_resolutions(self) -> None:
        qs = QuestionSet(
            forecast_due_date="2024-06-01",
            question_set="round_1",
            questions=[Question(
                id="q1", source="acled", question="Will X?",
                resolution_dates="N/A",
            )],
        )
        resolutions: dict[str, list[Resolution]] = {
            "q1": [
                Resolution(id="q1", outcome=1, resolution_date="2024-07-28", resolved=True),
                Resolution(id="q1", outcome=0, resolution_date="2024-08-20", resolved=True),
            ],
        }
        result = join_resolved_questions([qs], resolutions)
        assert len(result) == 2

    def test_none_resolution_dates_matches_all(self) -> None:
        qs = QuestionSet(
            forecast_due_date="2024-06-01",
            question_set="round_1",
            questions=[Question(id="q1", source="acled", question="Will X?")],
        )
        resolutions: dict[str, list[Resolution]] = {
            "q1": [
                Resolution(id="q1", outcome=1, resolution_date="2024-07-28", resolved=True),
                Resolution(id="q1", outcome=0, resolution_date="2024-08-20", resolved=True),
            ],
        }
        result = join_resolved_questions([qs], resolutions)
        assert len(result) == 2

    def test_resolution_with_none_date_filtered_for_multi_horizon(self) -> None:
        qs = QuestionSet(
            forecast_due_date="2024-06-01",
            question_set="round_1",
            questions=[Question(
                id="q1", source="acled", question="Will X?",
                resolution_dates=["2024-07-28"],
            )],
        )
        resolutions: dict[str, list[Resolution]] = {
            "q1": [
                Resolution(id="q1", outcome=1, resolution_date=None, resolved=True),
                Resolution(id="q1", outcome=0, resolution_date="2024-07-28", resolved=True),
            ],
        }
        result = join_resolved_questions([qs], resolutions)
        assert len(result) == 1
        assert result[0].resolution_date == "2024-07-28"


class TestMultiHorizonNullDateRejection:
    """Tests that null and N/A resolution_dates are rejected for multi-horizon questions."""

    def test_multi_horizon_rejects_null_resolution_date(self) -> None:
        qs = QuestionSet(
            forecast_due_date="2024-06-01",
            question_set="round_1",
            questions=[Question(
                id="q1", source="acled", question="Will X?",
                resolution_dates=["2024-07-28", "2024-08-20"],
            )],
        )
        resolutions: dict[str, list[Resolution]] = {
            "q1": [
                Resolution(id="q1", outcome=1, resolution_date=None, resolved=True),
                Resolution(id="q1", outcome=0, resolution_date="2024-07-28", resolved=True),
                Resolution(id="q1", outcome=1, resolution_date="2024-08-20", resolved=True),
            ],
        }
        result = join_resolved_questions([qs], resolutions)
        assert len(result) == 2
        dates = {r.resolution_date for r in result}
        assert dates == {"2024-07-28", "2024-08-20"}

    def test_multi_horizon_rejects_na_resolution_date(self) -> None:
        qs = QuestionSet(
            forecast_due_date="2024-06-01",
            question_set="round_1",
            questions=[Question(
                id="q1", source="acled", question="Will X?",
                resolution_dates=["2024-07-28", "2024-08-20"],
            )],
        )
        resolutions: dict[str, list[Resolution]] = {
            "q1": [
                Resolution(id="q1", outcome=1, resolution_date="N/A", resolved=True),
                Resolution(id="q1", outcome=0, resolution_date="2024-07-28", resolved=True),
                Resolution(id="q1", outcome=1, resolution_date="2024-08-20", resolved=True),
            ],
        }
        result = join_resolved_questions([qs], resolutions)
        assert len(result) == 2
        dates = {r.resolution_date for r in result}
        assert dates == {"2024-07-28", "2024-08-20"}


class TestRefreshCache:
    def test_deletes_listings_and_resolutions(self, tmp_path: Path) -> None:
        (tmp_path / "question_sets_listing.json").write_text("{}")
        (tmp_path / "resolution_sets_listing.json").write_text("{}")
        (tmp_path / "res_2024-01-01.json").write_text("{}")
        (tmp_path / "lb_baseline.csv").write_text("")
        (tmp_path / "qs_round1.json").write_text("{}")
        with patch("forecastbench_parity.questions.CACHE_DIR", tmp_path):
            refresh_cache()
        assert not (tmp_path / "question_sets_listing.json").exists()
        assert not (tmp_path / "resolution_sets_listing.json").exists()
        assert not (tmp_path / "res_2024-01-01.json").exists()
        assert not (tmp_path / "lb_baseline.csv").exists()
        assert (tmp_path / "qs_round1.json").exists()

    def test_noop_when_cache_dir_missing(self, tmp_path: Path) -> None:
        with patch("forecastbench_parity.questions.CACHE_DIR", tmp_path / "nonexistent"):
            refresh_cache()
