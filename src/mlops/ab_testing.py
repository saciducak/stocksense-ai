"""A/B testing framework for model comparison.

Implements statistical testing to determine whether a challenger
model significantly outperforms the current champion model.

This module addresses the job posting requirement:
    - "A/B test süreçleri"

Methodology:
    1. Split traffic between champion and challenger models
    2. Collect predictions and errors for both
    3. Use paired t-test to determine statistical significance
    4. Report whether to promote, reject, or continue testing
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TestDecision(str, Enum):
    """Possible outcomes of an A/B test."""

    PROMOTE = "promote"           # Challenger is significantly better
    REJECT = "reject"             # Champion is significantly better
    INCONCLUSIVE = "inconclusive" # Not enough evidence yet
    INSUFFICIENT_DATA = "insufficient_data"  # Need more samples


@dataclass
class ABTestResult:
    """Result of an A/B test comparison.

    Attributes:
        decision: Whether to promote, reject, or continue.
        p_value: Statistical significance (lower = more significant).
        champion_mean_error: Mean error of the champion model.
        challenger_mean_error: Mean error of the challenger model.
        improvement_pct: Percentage improvement of challenger over champion.
        num_samples: Number of test samples used.
        confidence_level: Required confidence level for the test.
    """

    decision: TestDecision
    p_value: float
    champion_mean_error: float
    challenger_mean_error: float
    improvement_pct: float
    num_samples: int
    confidence_level: float
    test_timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for API/logging."""
        return {
            "decision": self.decision.value,
            "p_value": round(self.p_value, 6),
            "champion_mean_error": round(self.champion_mean_error, 6),
            "challenger_mean_error": round(self.challenger_mean_error, 6),
            "improvement_pct": round(self.improvement_pct, 2),
            "num_samples": self.num_samples,
            "confidence_level": self.confidence_level,
            "timestamp": self.test_timestamp.isoformat(),
        }

    def __str__(self) -> str:
        """Human-readable test result."""
        return (
            f"A/B Test Result: {self.decision.value.upper()}\n"
            f"  Samples: {self.num_samples}\n"
            f"  Champion Error: {self.champion_mean_error:.6f}\n"
            f"  Challenger Error: {self.challenger_mean_error:.6f}\n"
            f"  Improvement: {self.improvement_pct:+.2f}%\n"
            f"  P-value: {self.p_value:.6f} "
            f"(threshold: {1 - self.confidence_level:.2f})"
        )


class ABTestFramework:
    """Statistical A/B testing for model comparison.

    Implements a rigorous statistical approach:
    1. Both models predict on the same test data
    2. Paired t-test compares their error distributions
    3. Decision based on p-value and significance level

    Why paired t-test?
        We're comparing the same models on the same data points,
        so errors are naturally paired. A paired test has more
        statistical power than an unpaired test.

    Traffic splitting strategy:
        - Start with 90/10 (champion/challenger) for safety
        - If challenger shows promise, move to 70/30
        - Only promote after statistical significance is achieved

    Example:
        >>> ab = ABTestFramework(significance_level=0.05, min_samples=100)
        >>> # Collect predictions from both models
        >>> ab.record_predictions(
        ...     actual=actual_prices,
        ...     champion_pred=champion_predictions,
        ...     challenger_pred=challenger_predictions,
        ... )
        >>> result = ab.evaluate()
        >>> print(result)
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        min_samples: int = 100,
        champion_traffic: float = 0.9,
    ) -> None:
        """Initialize A/B test framework.

        Args:
            significance_level: Required p-value threshold (default 0.05 = 95% confidence).
            min_samples: Minimum samples before making a decision.
            champion_traffic: Fraction of traffic for champion model.
        """
        self.significance_level = significance_level
        self.min_samples = min_samples
        self.champion_traffic = champion_traffic
        self.challenger_traffic = 1.0 - champion_traffic

        # Storage for test data
        self._champion_errors: list[float] = []
        self._challenger_errors: list[float] = []
        self._test_started: Optional[datetime] = None

        logger.info(
            f"ABTestFramework initialized: α={significance_level}, "
            f"min_samples={min_samples}, traffic={champion_traffic}/{self.challenger_traffic}"
        )

    def record_predictions(
        self,
        actual: NDArray,
        champion_pred: NDArray,
        challenger_pred: NDArray,
    ) -> None:
        """Record prediction results from both models.

        Computes squared errors for each model and stores them
        for later statistical analysis.

        Args:
            actual: Actual observed values.
            champion_pred: Champion model predictions.
            challenger_pred: Challenger model predictions.
        """
        if self._test_started is None:
            self._test_started = datetime.now()

        # Compute squared errors
        champion_errors = (actual - champion_pred) ** 2
        challenger_errors = (actual - challenger_pred) ** 2

        self._champion_errors.extend(champion_errors.flatten().tolist())
        self._challenger_errors.extend(challenger_errors.flatten().tolist())

        logger.info(
            f"  Recorded {len(actual)} samples "
            f"(total: {len(self._champion_errors)})"
        )

    def evaluate(self) -> ABTestResult:
        """Evaluate the A/B test and make a decision.

        Uses paired t-test to compare error distributions.
        The null hypothesis is that both models have equal performance.

        Returns:
            ABTestResult with decision, p-value, and detailed metrics.
        """
        num_samples = len(self._champion_errors)

        # Check minimum sample requirement
        if num_samples < self.min_samples:
            logger.warning(
                f"Insufficient data: {num_samples}/{self.min_samples} samples"
            )
            return ABTestResult(
                decision=TestDecision.INSUFFICIENT_DATA,
                p_value=1.0,
                champion_mean_error=np.mean(self._champion_errors) if self._champion_errors else 0,
                challenger_mean_error=np.mean(self._challenger_errors) if self._challenger_errors else 0,
                improvement_pct=0.0,
                num_samples=num_samples,
                confidence_level=1 - self.significance_level,
            )

        champion_arr = np.array(self._champion_errors)
        challenger_arr = np.array(self._challenger_errors)

        # Paired t-test (two-sided)
        t_statistic, p_value = stats.ttest_rel(champion_arr, challenger_arr)

        champion_mean = float(np.mean(champion_arr))
        challenger_mean = float(np.mean(challenger_arr))

        # Improvement percentage (negative means challenger is worse)
        if champion_mean > 0:
            improvement_pct = ((champion_mean - challenger_mean) / champion_mean) * 100
        else:
            improvement_pct = 0.0

        # Decision logic
        if p_value < self.significance_level:
            if challenger_mean < champion_mean:
                decision = TestDecision.PROMOTE
            else:
                decision = TestDecision.REJECT
        else:
            decision = TestDecision.INCONCLUSIVE

        result = ABTestResult(
            decision=decision,
            p_value=float(p_value),
            champion_mean_error=champion_mean,
            challenger_mean_error=challenger_mean,
            improvement_pct=improvement_pct,
            num_samples=num_samples,
            confidence_level=1 - self.significance_level,
        )

        logger.info(f"\n{result}")
        return result

    def should_route_to_challenger(self) -> bool:
        """Determine if the next request should go to the challenger.

        Uses the configured traffic split ratio.

        Returns:
            True if request should go to challenger model.
        """
        return np.random.random() < self.challenger_traffic

    def reset(self) -> None:
        """Reset the A/B test — clear all collected data."""
        self._champion_errors.clear()
        self._challenger_errors.clear()
        self._test_started = None
        logger.info("A/B test reset — all data cleared")

    def get_status(self) -> dict:
        """Get current A/B test status.

        Returns:
            Dictionary with test progress and preliminary results.
        """
        num_samples = len(self._champion_errors)
        return {
            "samples_collected": num_samples,
            "min_samples_required": self.min_samples,
            "progress_pct": min(100, (num_samples / self.min_samples) * 100),
            "champion_traffic": self.champion_traffic,
            "challenger_traffic": self.challenger_traffic,
            "test_started": self._test_started.isoformat() if self._test_started else None,
            "ready_for_evaluation": num_samples >= self.min_samples,
        }

    def get_config(self) -> dict:
        """Return framework configuration for logging."""
        return {
            "significance_level": self.significance_level,
            "min_samples": self.min_samples,
            "champion_traffic": self.champion_traffic,
            "challenger_traffic": self.challenger_traffic,
        }
