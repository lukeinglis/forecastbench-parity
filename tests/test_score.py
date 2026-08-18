"""Tests for forecastbench_parity.score."""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from forecastbench_parity.questions import ResolvedQuestion
from forecastbench_parity.score import (
    _build_market_effects,
    _estimate_difficulty_effects_ols,
    _scoring_key,
    adjust_for_difficulty,
    brier_index,
    brier_score,
    mean_brier_score,
    score_forecasts,
)


class TestBrierScore:
    def test_perfect_positive(self) -> None:
        assert brier_score(1.0, 1) == 0.0

    def test_perfect_negative(self) -> None:
        assert brier_score(0.0, 0) == 0.0

    def test_worst_positive(self) -> None:
        assert brier_score(0.0, 1) == 1.0

    def test_worst_negative(self) -> None:
        assert brier_score(1.0, 0) == 1.0

    def test_half(self) -> None:
        assert brier_score(0.5, 1) == 0.25
        assert brier_score(0.5, 0) == 0.25

    def test_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            brier_score(float("nan"), 1)

    def test_rejects_inf(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            brier_score(float("inf"), 1)

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            brier_score(1.5, 1)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            brier_score(-0.1, 0)

    def test_rejects_invalid_outcome(self) -> None:
        with pytest.raises(ValueError, match="0 or 1"):
            brier_score(0.5, 2)


class TestHandComputedFixture:
    def test_individual_scores(self, five_question_fixture: tuple) -> None:
        forecasts, outcomes, expected_bs, _, _ = five_question_fixture
        for f, o, expected in zip(forecasts, outcomes, expected_bs):
            actual = brier_score(f, o)
            assert abs(actual - expected) < 1e-10

    def test_mean_brier_score(self, five_question_fixture: tuple) -> None:
        forecasts, outcomes, _, expected_mean, _ = five_question_fixture
        pairs = list(zip(forecasts, outcomes))
        actual = mean_brier_score(pairs)
        assert abs(actual - expected_mean) < 1e-10

    def test_brier_index(self, five_question_fixture: tuple) -> None:
        _, _, _, expected_mean, expected_index = five_question_fixture
        actual = brier_index(expected_mean)
        assert abs(actual - expected_index) < 1e-6


class TestMeanBrierScore:
    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            mean_brier_score([])

    def test_single(self) -> None:
        assert mean_brier_score([(0.5, 1)]) == 0.25


class TestBrierIndex:
    def test_perfect(self) -> None:
        assert brier_index(0.0) == 100.0

    def test_worst(self) -> None:
        assert brier_index(1.0) == 0.0

    def test_half(self) -> None:
        expected = (1.0 - math.sqrt(0.25)) * 100.0
        assert abs(brier_index(0.25) - expected) < 1e-10


class TestScoreForecasts:
    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="No resolved"):
            score_forecasts({}, [])

    def test_dataset_and_market_separate(self, mixed_resolved_questions: list) -> None:
        forecasts = {"d1": 0.9, "d2": 0.1, "m1": 0.9, "m2": 0.1}
        result = score_forecasts(forecasts, mixed_resolved_questions)
        assert result.n_dataset == 2
        assert result.n_market == 2
        assert result.dataset_brier == result.market_brier
        assert result.n_missing == 0

    def test_missing_defaults_to_half(self, five_resolved_questions: list) -> None:
        result = score_forecasts({}, five_resolved_questions)
        assert result.n_missing == 5
        assert abs(result.dataset_brier - 0.25) < 1e-10

    def test_overall_is_average_of_components(self, mixed_resolved_questions: list) -> None:
        forecasts = {"d1": 0.8, "d2": 0.2, "m1": 0.6, "m2": 0.4}
        result = score_forecasts(forecasts, mixed_resolved_questions)
        expected_overall = (result.dataset_brier + result.market_brier) / 2.0
        assert abs(result.overall_brier - expected_overall) < 1e-10


class TestPropertyBased:
    @given(
        forecast=st.floats(min_value=0.0, max_value=1.0),
        outcome=st.sampled_from([0, 1]),
    )
    def test_brier_in_range(self, forecast: float, outcome: int) -> None:
        bs = brier_score(forecast, outcome)
        assert 0.0 <= bs <= 1.0

    @given(
        forecasts=st.lists(
            st.tuples(
                st.floats(min_value=0.0, max_value=1.0),
                st.sampled_from([0, 1]),
            ),
            min_size=1,
            max_size=20,
        ),
    )
    def test_mean_brier_in_range(self, forecasts: list[tuple[float, int]]) -> None:
        mbs = mean_brier_score(forecasts)
        assert 0.0 <= mbs <= 1.0

    @given(mean_bs=st.floats(min_value=0.0, max_value=1.0))
    def test_brier_index_in_range(self, mean_bs: float) -> None:
        bi = brier_index(mean_bs)
        assert 0.0 <= bi <= 100.0

    def test_perfect_forecasts(self) -> None:
        pairs = [(1.0, 1), (0.0, 0), (1.0, 1), (0.0, 0)]
        mbs = mean_brier_score(pairs)
        assert mbs == 0.0
        assert brier_index(mbs) == 100.0

    def test_worst_forecasts(self) -> None:
        pairs = [(0.0, 1), (1.0, 0), (0.0, 1), (1.0, 0)]
        mbs = mean_brier_score(pairs)
        assert mbs == 1.0
        assert brier_index(mbs) == 0.0


def _make_resolved(
    qid: str, source: str, outcome: int, due: str = "2024-01-01",
) -> ResolvedQuestion:
    return ResolvedQuestion(
        id=qid, source=source, question=f"Q {qid}",
        outcome=outcome, forecast_due_date=due,
    )


class TestDifficultyEffectsOLS:
    def test_symmetric_forecasters(self) -> None:
        qs = [_make_resolved("q1", "acled", 1), _make_resolved("q2", "acled", 0)]
        forecasts = {"A": {"q1": 0.9, "q2": 0.1}, "B": {"q1": 0.7, "q2": 0.3}}
        outcomes = {q.id: q.outcome for q in qs}
        effects = _estimate_difficulty_effects_ols(forecasts, outcomes, ["q1", "q2"])
        assert abs(effects["q1"] + effects["q2"]) < 1e-10

    def test_single_forecaster_effects_zero(self) -> None:
        qs = [_make_resolved("q1", "acled", 1), _make_resolved("q2", "acled", 0)]
        forecasts = {"A": {"q1": 0.8, "q2": 0.2}}
        outcomes = {q.id: q.outcome for q in qs}
        effects = _estimate_difficulty_effects_ols(forecasts, outcomes, ["q1", "q2"])
        assert abs(effects["q1"]) < 1e-10
        assert abs(effects["q2"]) < 1e-10


class TestAdjustForDifficulty:
    def test_constant_half_yields_025(self) -> None:
        qs = [_make_resolved("q1", "acled", 1), _make_resolved("q2", "acled", 0), _make_resolved("q3", "acled", 1)]
        forecasts = {
            "half": {"q1": 0.5, "q2": 0.5, "q3": 0.5},
            "good": {"q1": 0.9, "q2": 0.1, "q3": 0.8},
            "bad": {"q1": 0.2, "q2": 0.8, "q3": 0.3},
        }
        result = adjust_for_difficulty(forecasts, qs)
        half_scores = result.adjusted_scores["half"]
        mean_adj = sum(half_scores.values()) / len(half_scores)
        assert abs(mean_adj - 0.25) < 1e-10

    def test_empty_returns_empty(self) -> None:
        result = adjust_for_difficulty({}, [])
        assert result.adjusted_scores == {}
        assert result.question_effects == {}


class TestScoreForecastsDifficultyAdjusted:
    def test_no_peer_pool_falls_back_to_raw(self) -> None:
        qs = [_make_resolved(f"q{i}", "acled", o) for i, o in enumerate([1, 0, 1])]
        forecasts = {"q0": 0.5, "q1": 0.5, "q2": 0.5}
        result = score_forecasts(forecasts, qs, difficulty_adjusted=True)
        assert not result.difficulty_adjusted
        assert abs(result.dataset_brier - 0.25) < 1e-10

    def test_adjusted_constant_half_yields_025_mean(self) -> None:
        qs = [_make_resolved("q1", "acled", 1), _make_resolved("q2", "acled", 0), _make_resolved("q3", "acled", 1)]
        peer_pool = {"peer1": {"q1": 0.9, "q2": 0.1, "q3": 0.8}, "peer2": {"q1": 0.3, "q2": 0.7, "q3": 0.4}}
        result = score_forecasts({"q1": 0.5, "q2": 0.5, "q3": 0.5}, qs, difficulty_adjusted=True, all_forecasts=peer_pool)
        assert result.difficulty_adjusted
        assert abs(result.dataset_brier - 0.25) < 1e-10
        assert abs(result.dataset_index - brier_index(0.25)) < 1e-6

    def test_brier_index_applied_after_averaging(self) -> None:
        qs = [_make_resolved("q1", "acled", 1), _make_resolved("q2", "acled", 0)]
        peer_pool = {"p1": {"q1": 0.7, "q2": 0.3}, "p2": {"q1": 0.6, "q2": 0.4}}
        result = score_forecasts({"q1": 0.9, "q2": 0.1}, qs, difficulty_adjusted=True, all_forecasts=peer_pool)
        expected_index = (1.0 - math.sqrt(result.dataset_brier)) * 100.0
        assert abs(result.dataset_index - expected_index) < 1e-6


class TestOverallDiverges:
    def test_overall_diverges_on_unbalanced_split(
        self, unbalanced_resolved_questions: list,
    ) -> None:
        forecasts = {"d1": 0.9, "d2": 0.1, "d3": 0.8, "m1": 0.6}
        result = score_forecasts(forecasts, unbalanced_resolved_questions)
        expected_equal_weight = (result.dataset_brier + result.market_brier) / 2.0
        assert abs(result.overall_brier - expected_equal_weight) < 1e-10
        n_ds, n_mk = result.n_dataset, result.n_market
        count_weighted = (
            result.dataset_brier * n_ds + result.market_brier * n_mk
        ) / (n_ds + n_mk)
        assert abs(result.overall_brier - count_weighted) > 1e-6


class TestDifficultyNoClamp:
    def test_difficulty_adjustment_no_clamp(self) -> None:
        qs = [
            _make_resolved("q1", "acled", 1),
            _make_resolved("q2", "acled", 0),
        ]
        forecasts = {
            "A": {"q1": 1.0, "q2": 0.0},
            "B": {"q1": 1.0, "q2": 0.0},
        }
        result = adjust_for_difficulty(forecasts, qs)
        has_out_of_unit = any(
            v < 0.0 or v > 1.0
            for fscores in result.adjusted_scores.values()
            for v in fscores.values()
        )
        all_in_unit = all(
            0.0 <= v <= 1.0
            for fscores in result.adjusted_scores.values()
            for v in fscores.values()
        )
        assert has_out_of_unit or all_in_unit


class TestAdjustForDifficultyDedup:
    def test_duplicate_scoring_keys_not_inflated(self) -> None:
        resolved = [
            _make_resolved_horizon("m1", "metaculus", 1, "2024-07-28"),
            _make_resolved_horizon("m1", "metaculus", 1, "2024-07-28"),
            _make_resolved_horizon("m1", "metaculus", 1, "2024-07-28"),
            _make_resolved_horizon("d1", "acled", 0, "2024-07-28"),
        ]
        forecasts = {
            "A": {"m1_2024-07-28": 0.5, "d1_2024-07-28": 0.5},
            "B": {"m1_2024-07-28": 0.9, "d1_2024-07-28": 0.1},
        }
        result = adjust_for_difficulty(forecasts, resolved)
        half_scores = result.adjusted_scores["A"]
        market_vals = [v for k, v in half_scores.items() if k.startswith("m1")]
        dataset_vals = [v for k, v in half_scores.items() if k.startswith("d1")]
        assert len(market_vals) == 1
        assert len(dataset_vals) == 1
        assert abs(market_vals[0] - 0.25) < 1e-10
        assert abs(dataset_vals[0] - 0.25) < 1e-10


class TestPerColumnShifts:
    def test_per_column_shifts(self) -> None:
        qs = [
            _make_resolved("d1", "acled", 1),
            _make_resolved("d2", "acled", 1),
            _make_resolved("m1", "metaculus", 0),
            _make_resolved("m2", "polymarket", 0),
        ]
        forecasts = {
            "A": {"d1": 0.5, "d2": 0.5, "m1": 0.5, "m2": 0.5},
            "B": {"d1": 0.9, "d2": 0.9, "m1": 0.1, "m2": 0.1},
        }
        result = adjust_for_difficulty(forecasts, qs)
        half_scores = result.adjusted_scores["A"]
        ds_mean = (half_scores["d1"] + half_scores["d2"]) / 2
        mk_mean = (half_scores["m1"] + half_scores["m2"]) / 2
        assert abs(ds_mean - 0.25) < 1e-10
        assert abs(mk_mean - 0.25) < 1e-10


class TestMarketEffects:
    def test_market_effects_are_centered(self) -> None:
        qs = [_make_resolved("m1", "metaculus", 1), _make_resolved("m2", "polymarket", 0), _make_resolved("m3", "metaculus", 1)]
        forecasts = {"A": {"m1": 0.8, "m2": 0.2, "m3": 0.7}, "B": {"m1": 0.6, "m2": 0.4, "m3": 0.5}}
        outcomes = {q.id: q.outcome for q in qs}
        market_forecasts = {"m1": 0.75, "m2": 0.25, "m3": 0.65}
        effects = _build_market_effects(forecasts, outcomes, ["m1", "m2", "m3"], market_weight=1.0, market_forecasts=market_forecasts)
        assert len(effects) == 3
        assert abs(sum(effects.values())) < 1e-10


def _make_resolved_horizon(
    qid: str, source: str, outcome: int, resolution_date: str,
    due: str = "2024-01-01",
) -> ResolvedQuestion:
    return ResolvedQuestion(
        id=qid, source=source, question=f"Q {qid}",
        outcome=outcome, resolution_date=resolution_date,
        forecast_due_date=due,
    )


class TestScoringKey:
    def test_with_resolution_date(self) -> None:
        q = _make_resolved_horizon("q1", "acled", 1, "2024-07-28")
        assert _scoring_key(q) == "q1_2024-07-28"

    def test_with_na_resolution_date(self) -> None:
        q = _make_resolved("q1", "acled", 1)
        q2 = ResolvedQuestion(
            id="q1", source="acled", question="Q", outcome=1,
            resolution_date="N/A", forecast_due_date="2024-01-01",
        )
        assert _scoring_key(q) == "q1"
        assert _scoring_key(q2) == "q1"

    def test_with_none_resolution_date(self) -> None:
        q = _make_resolved("q1", "acled", 1)
        assert q.resolution_date is None
        assert _scoring_key(q) == "q1"


class TestMultiHorizonScoring:
    def test_each_horizon_gets_own_brier_score(self) -> None:
        resolved = [
            _make_resolved_horizon("q1", "acled", 1, "2024-07-28"),
            _make_resolved_horizon("q1", "acled", 0, "2024-08-20"),
        ]
        forecasts = {"q1": 0.9}
        result = score_forecasts(forecasts, resolved, difficulty_adjusted=False)
        bs_h1 = (0.9 - 1) ** 2  # 0.01
        bs_h2 = (0.9 - 0) ** 2  # 0.81
        expected_mean = (bs_h1 + bs_h2) / 2.0
        assert result.n_dataset == 2
        assert abs(result.dataset_brier - expected_mean) < 1e-10

    def test_missing_forecast_counted_per_horizon(self) -> None:
        resolved = [
            _make_resolved_horizon("q1", "acled", 1, "2024-07-28"),
            _make_resolved_horizon("q1", "acled", 0, "2024-08-20"),
            _make_resolved_horizon("q2", "acled", 1, "2024-07-28"),
        ]
        forecasts = {"q1": 0.8}
        result = score_forecasts(forecasts, resolved, difficulty_adjusted=False)
        assert result.n_missing == 1  # q2 missing, q1 covers both horizons

    def test_single_horizon_backward_compatible(self) -> None:
        resolved = [_make_resolved("q1", "acled", 1), _make_resolved("q2", "acled", 0)]
        forecasts = {"q1": 0.9, "q2": 0.1}
        result = score_forecasts(forecasts, resolved, difficulty_adjusted=False)
        expected = ((0.9 - 1) ** 2 + (0.1 - 0) ** 2) / 2.0
        assert abs(result.dataset_brier - expected) < 1e-10
        assert result.n_missing == 0


class TestMultiHorizonDifficultyAdjustment:
    def test_different_outcomes_per_horizon_produce_different_effects(self) -> None:
        resolved = [
            _make_resolved_horizon("q1", "acled", 1, "2024-07-28"),
            _make_resolved_horizon("q1", "acled", 0, "2024-08-20"),
        ]
        peer_pool = {
            "peer1": {"q1": 0.9},
            "peer2": {"q1": 0.3},
        }
        scoring_key_to_base = {_scoring_key(q): q.id for q in resolved}
        remapped_pool: dict[str, dict[str, float]] = {}
        for fid, fcast_map in peer_pool.items():
            remapped: dict[str, float] = {}
            for sk, base_id in scoring_key_to_base.items():
                if base_id in fcast_map:
                    remapped[sk] = fcast_map[base_id]
            remapped_pool[fid] = remapped

        result = adjust_for_difficulty(remapped_pool, resolved)
        effects = result.question_effects
        assert "q1_2024-07-28" in effects
        assert "q1_2024-08-20" in effects
        assert effects["q1_2024-07-28"] != effects["q1_2024-08-20"]

    def test_adjusted_scoring_with_multi_horizon(self) -> None:
        resolved = [
            _make_resolved_horizon("q1", "acled", 1, "2024-07-28"),
            _make_resolved_horizon("q1", "acled", 0, "2024-08-20"),
            _make_resolved_horizon("q2", "acled", 1, "2024-07-28"),
        ]
        peer_pool = {
            "peer1": {"q1": 0.8, "q2": 0.7},
            "peer2": {"q1": 0.4, "q2": 0.5},
        }
        forecasts = {"q1": 0.5, "q2": 0.5}
        result = score_forecasts(
            forecasts, resolved,
            difficulty_adjusted=True,
            all_forecasts=peer_pool,
        )
        assert result.difficulty_adjusted
        assert result.n_dataset == 3
        assert result.n_missing == 0
