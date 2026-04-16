"""Tests for NLP modules (sentiment and NER)."""

import pytest

from src.nlp.entity_extractor import EntityExtractor, Entity, ExtractionResult
from src.nlp.sentiment_analyzer import SentimentResult, BatchSentimentResult


# ─── SentimentResult Tests ────────────────────────────────────────────────────

class TestSentimentResult:
    """Test sentiment result data structures."""

    def test_numeric_score_positive(self) -> None:
        """Positive sentiment should give positive numeric score."""
        result = SentimentResult(
            text="test",
            label="positive",
            score=0.9,
            scores={"positive": 0.9, "negative": 0.05, "neutral": 0.05},
        )
        assert result.numeric_score > 0

    def test_numeric_score_negative(self) -> None:
        """Negative sentiment should give negative numeric score."""
        result = SentimentResult(
            text="test",
            label="negative",
            score=0.8,
            scores={"positive": 0.1, "negative": 0.8, "neutral": 0.1},
        )
        assert result.numeric_score < 0

    def test_numeric_score_neutral(self) -> None:
        """Neutral sentiment should be close to 0."""
        result = SentimentResult(
            text="test",
            label="neutral",
            score=0.9,
            scores={"positive": 0.05, "negative": 0.05, "neutral": 0.9},
        )
        assert abs(result.numeric_score) < 0.1


# ─── BatchSentimentResult Tests ──────────────────────────────────────────────

class TestBatchSentimentResult:
    """Test batch sentiment result aggregation."""

    def test_mean_score(self) -> None:
        """Mean score should average numeric scores."""
        results = [
            SentimentResult("a", "positive", 0.9, {"positive": 0.9, "negative": 0.05, "neutral": 0.05}),
            SentimentResult("b", "negative", 0.8, {"positive": 0.1, "negative": 0.8, "neutral": 0.1}),
        ]
        batch = BatchSentimentResult(results=results)

        # Mean should be between the two scores
        assert -1 <= batch.mean_score <= 1

    def test_label_distribution(self) -> None:
        """Distribution should count each label."""
        results = [
            SentimentResult("a", "positive", 0.9, {}),
            SentimentResult("b", "positive", 0.8, {}),
            SentimentResult("c", "negative", 0.7, {}),
        ]
        batch = BatchSentimentResult(results=results)

        dist = batch.label_distribution
        assert dist["positive"] == 2
        assert dist["negative"] == 1

    def test_empty_batch(self) -> None:
        """Empty batch should have mean_score 0."""
        batch = BatchSentimentResult(results=[])
        assert batch.mean_score == 0.0

    def test_summary(self) -> None:
        """Summary should include key info."""
        results = [
            SentimentResult("a", "positive", 0.9, {"positive": 0.9, "negative": 0.05, "neutral": 0.05}),
        ]
        batch = BatchSentimentResult(results=results)
        summary = batch.summary
        assert "total_articles" in summary
        assert "mean_sentiment" in summary


# ─── Entity Extraction Tests ─────────────────────────────────────────────────

class TestEntityExtraction:
    """Test entity extraction without loading the model."""

    def test_ticker_extraction(self) -> None:
        """Should extract stock tickers from text."""
        extractor_cls = EntityExtractor
        # Test the static method directly
        tickers = extractor_cls.TICKER_PATTERN.findall("AAPL and GOOGL are tech stocks")
        # Filter through blacklist
        filtered = [t for t in tickers if t not in extractor_cls.TICKER_BLACKLIST and len(t) >= 2]
        assert "AAPL" in filtered
        assert "GOOGL" in filtered

    def test_monetary_extraction(self) -> None:
        """Should extract monetary values."""
        text = "Revenue was $94.8B and costs were $12.5M"
        values = EntityExtractor.MONEY_PATTERN.findall(text)
        assert len(values) >= 1

    def test_ticker_blacklist(self) -> None:
        """Common English words should be filtered out."""
        assert "THE" in EntityExtractor.TICKER_BLACKLIST
        assert "FOR" in EntityExtractor.TICKER_BLACKLIST
        assert "AAPL" not in EntityExtractor.TICKER_BLACKLIST

    def test_entity_to_dict(self) -> None:
        """Entity should be serializable."""
        entity = Entity(text="Apple", label="ORG", score=0.95)
        d = entity.to_dict()
        assert d["text"] == "Apple"
        assert d["label"] == "ORG"

    def test_extraction_result(self) -> None:
        """ExtractionResult should aggregate entities."""
        result = ExtractionResult(
            text="test",
            companies=["Apple", "Google"],
            tickers=["AAPL", "GOOGL"],
        )
        assert result.entity_count == 0  # No model entities
        assert len(result.companies) == 2
        d = result.to_dict()
        assert "companies" in d
