"""Tests for the data pipeline modules."""

import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineer import FeatureEngineer
from src.data.news_fetcher import NewsFetcher, NewsArticle, NewsBatch
from src.data.preprocessor import DataPreprocessor, ScalingMethod


# ─── Feature Engineer Tests ──────────────────────────────────────────────────

class TestFeatureEngineer:
    """Test technical indicator computations."""

    @pytest.fixture
    def sample_ohlcv(self) -> pd.DataFrame:
        """Create sample OHLCV data for testing."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)

        return pd.DataFrame({
            "Open": close + np.random.randn(n) * 0.3,
            "High": close + np.abs(np.random.randn(n)) * 0.5,
            "Low": close - np.abs(np.random.randn(n)) * 0.5,
            "Close": close,
            "Volume": np.random.randint(1_000_000, 10_000_000, n),
        }, index=dates)

    def test_compute_all_returns_dataframe(self, sample_ohlcv: pd.DataFrame) -> None:
        """Ensure compute_all returns a DataFrame with features."""
        engineer = FeatureEngineer()
        features = engineer.compute_all(sample_ohlcv)

        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_ohlcv)
        assert features.shape[1] > 10  # Should have many features

    def test_rsi_range(self, sample_ohlcv: pd.DataFrame) -> None:
        """RSI should be between 0 and 100."""
        engineer = FeatureEngineer()
        features = engineer.compute_all(sample_ohlcv)

        rsi = features["RSI"].dropna()
        assert rsi.min() >= 0
        assert rsi.max() <= 100

    def test_sma_periods(self, sample_ohlcv: pd.DataFrame) -> None:
        """SMA columns should be created for each period."""
        periods = [10, 20, 50]
        engineer = FeatureEngineer(sma_periods=periods)
        features = engineer.compute_all(sample_ohlcv)

        for period in periods:
            assert f"SMA_{period}" in features.columns

    def test_macd_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        """MACD should create three columns."""
        engineer = FeatureEngineer()
        features = engineer.compute_all(sample_ohlcv)

        assert "MACD" in features.columns
        assert "MACD_Signal" in features.columns
        assert "MACD_Hist" in features.columns

    def test_bollinger_bands(self, sample_ohlcv: pd.DataFrame) -> None:
        """Bollinger Bands should have upper > lower."""
        engineer = FeatureEngineer()
        features = engineer.compute_all(sample_ohlcv)

        valid_idx = features["BB_Upper"].dropna().index
        upper = features.loc[valid_idx, "BB_Upper"]
        lower = features.loc[valid_idx, "BB_Lower"]
        assert (upper >= lower).all()


# ─── Preprocessor Tests ──────────────────────────────────────────────────────

class TestDataPreprocessor:
    """Test data preprocessing pipeline."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample stock data."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        close = 100 + np.cumsum(np.random.randn(n) * 0.3)

        return pd.DataFrame({
            "Open": close + np.random.randn(n) * 0.2,
            "High": close + np.abs(np.random.randn(n)),
            "Low": close - np.abs(np.random.randn(n)),
            "Close": close,
            "Volume": np.random.randint(1_000_000, 10_000_000, n),
        }, index=dates)

    def test_fit_transform_returns_split(self, sample_data: pd.DataFrame) -> None:
        """fit_transform should return SplitDataset with train/val/test."""
        preprocessor = DataPreprocessor(sequence_length=30, forecast_horizon=5)
        split = preprocessor.fit_transform(sample_data)

        assert split.train.num_samples > 0
        assert split.val.num_samples > 0
        assert split.test.num_samples > 0

    def test_sequence_shape(self, sample_data: pd.DataFrame) -> None:
        """Sequences should have correct shapes."""
        seq_len = 30
        horizon = 5
        preprocessor = DataPreprocessor(sequence_length=seq_len, forecast_horizon=horizon)
        split = preprocessor.fit_transform(sample_data)

        assert split.train.X.shape[1] == seq_len
        assert split.train.y.shape[1] == horizon
        assert split.train.X.shape[2] == split.train.num_features

    def test_no_data_leakage(self, sample_data: pd.DataFrame) -> None:
        """Scaler should be fitted on training data only."""
        preprocessor = DataPreprocessor(sequence_length=30, forecast_horizon=5)
        split = preprocessor.fit_transform(sample_data)

        # Training data should be scaled to [0, 1] approximately
        assert split.train.X.min() >= -0.5
        assert split.train.X.max() <= 1.5

    def test_inverse_transform(self, sample_data: pd.DataFrame) -> None:
        """Inverse transform should recover original scale."""
        preprocessor = DataPreprocessor(sequence_length=30, forecast_horizon=5)
        split = preprocessor.fit_transform(sample_data)

        # Get some scaled values and invert
        scaled = split.train.y[0]
        original = preprocessor.inverse_transform_target(scaled)
        assert len(original) == len(scaled)

    def test_scaling_methods(self, sample_data: pd.DataFrame) -> None:
        """Both scaling methods should work."""
        for method in ScalingMethod:
            preprocessor = DataPreprocessor(
                sequence_length=30,
                forecast_horizon=5,
                scaling_method=method,
            )
            split = preprocessor.fit_transform(sample_data)
            assert split.train.num_samples > 0


# ─── News Fetcher Tests ──────────────────────────────────────────────────────

class TestNewsBatch:
    """Test NewsBatch functionality."""

    def test_filter_by_keyword(self) -> None:
        """Keyword filtering should work correctly."""
        articles = [
            NewsArticle(title="Apple stock rises", summary="Good news", source="test"),
            NewsArticle(title="Google announces AI", summary="Tech news", source="test"),
            NewsArticle(title="Apple revenue beats", summary="Strong quarter", source="test"),
        ]
        batch = NewsBatch(articles=articles)

        filtered = batch.filter_by_keyword("Apple")
        assert filtered.count == 2

    def test_empty_batch(self) -> None:
        """Empty batch should have count 0."""
        batch = NewsBatch()
        assert batch.count == 0
        assert batch.texts == []

    def test_texts_property(self) -> None:
        """Texts property should combine title and summary."""
        articles = [
            NewsArticle(title="Test", summary="Summary", source="test"),
        ]
        batch = NewsBatch(articles=articles)
        assert len(batch.texts) == 1
        assert "Test" in batch.texts[0]


class TestNewsFetcher:
    """Test NewsFetcher deduplication and cleaning."""

    def test_deduplication(self) -> None:
        """Duplicate titles should be removed."""
        fetcher = NewsFetcher()
        articles = [
            NewsArticle(title="Same Title", summary="v1", source="a"),
            NewsArticle(title="Same Title", summary="v2", source="b"),
            NewsArticle(title="Different Title", summary="v3", source="c"),
        ]
        unique = fetcher._deduplicate(articles)
        assert len(unique) == 2

    def test_clean_html(self) -> None:
        """HTML tags should be stripped."""
        cleaned = NewsFetcher._clean_html("<p>Hello <b>world</b></p>")
        assert cleaned == "Hello world"

    def test_clean_html_no_tags(self) -> None:
        """Text without tags should pass through."""
        text = "No HTML here"
        assert NewsFetcher._clean_html(text) == text
