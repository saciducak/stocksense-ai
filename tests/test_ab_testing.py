"""Tests for A/B testing framework."""

import numpy as np
import pytest

from src.mlops.ab_testing import ABTestFramework, TestDecision


class TestABTestFramework:
    """Test A/B testing statistical framework."""

    def test_insufficient_data(self) -> None:
        """Should return INSUFFICIENT_DATA when not enough samples."""
        ab = ABTestFramework(min_samples=100)
        ab.record_predictions(
            actual=np.array([1.0, 2.0]),
            champion_pred=np.array([1.1, 2.1]),
            challenger_pred=np.array([1.2, 2.2]),
        )
        result = ab.evaluate()
        assert result.decision == TestDecision.INSUFFICIENT_DATA

    def test_promote_decision(self) -> None:
        """Should PROMOTE when challenger is significantly better."""
        np.random.seed(42)
        ab = ABTestFramework(significance_level=0.05, min_samples=50)

        # Challenger is consistently better
        actual = np.random.randn(200)
        champion_pred = actual + np.random.randn(200) * 2  # High error
        challenger_pred = actual + np.random.randn(200) * 0.1  # Low error

        ab.record_predictions(actual, champion_pred, challenger_pred)
        result = ab.evaluate()
        assert result.decision == TestDecision.PROMOTE
        assert result.improvement_pct > 0

    def test_reject_decision(self) -> None:
        """Should REJECT when champion is better."""
        np.random.seed(42)
        ab = ABTestFramework(significance_level=0.05, min_samples=50)

        actual = np.random.randn(200)
        champion_pred = actual + np.random.randn(200) * 0.1  # Low error
        challenger_pred = actual + np.random.randn(200) * 2  # High error

        ab.record_predictions(actual, champion_pred, challenger_pred)
        result = ab.evaluate()
        assert result.decision == TestDecision.REJECT

    def test_traffic_routing(self) -> None:
        """Traffic routing should respect configured split."""
        ab = ABTestFramework(champion_traffic=0.9)

        # Run many iterations and check distribution
        np.random.seed(42)
        challenger_count = sum(
            ab.should_route_to_challenger() for _ in range(10000)
        )
        ratio = challenger_count / 10000
        assert 0.05 < ratio < 0.15  # ~10% ± 5%

    def test_reset(self) -> None:
        """Reset should clear all data."""
        ab = ABTestFramework(min_samples=10)
        ab.record_predictions(
            np.array([1.0]), np.array([1.1]), np.array([0.9]),
        )
        ab.reset()
        status = ab.get_status()
        assert status["samples_collected"] == 0

    def test_result_to_dict(self) -> None:
        """Result should be serializable."""
        ab = ABTestFramework(min_samples=5)
        ab.record_predictions(
            np.random.randn(10),
            np.random.randn(10),
            np.random.randn(10),
        )
        result = ab.evaluate()
        d = result.to_dict()
        assert "decision" in d
        assert "p_value" in d

    def test_get_status(self) -> None:
        """Status should report progress."""
        ab = ABTestFramework(min_samples=100)
        ab.record_predictions(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.1, 2.1, 3.1]),
            np.array([1.2, 2.2, 3.2]),
        )
        status = ab.get_status()
        assert status["samples_collected"] == 3
        assert status["progress_pct"] == 3.0
        assert status["ready_for_evaluation"] is False
