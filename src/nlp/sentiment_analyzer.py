"""FinBERT-based financial sentiment analyzer.

Uses the ProsusAI/finbert model — a BERT variant fine-tuned on
50,000+ financial news articles — to classify text into
positive, negative, or neutral sentiment with confidence scores.

Why FinBERT over generic BERT?
    Generic BERT misclassifies financial jargon. For example:
    - "The stock fell 5%" → Generic BERT: neutral, FinBERT: negative
    - "Earnings beat expectations" → Generic BERT: neutral, FinBERT: positive
    Domain-specific pre-training captures financial context accurately.

This module addresses the job posting requirements:
    - "Metin sınıflandırma" (text classification)
    - "NLP/LLM bileşenlerini yazılım sistemlerine entegre etme"
"""

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SentimentResult:
    """Result of sentiment analysis for a single text.

    Attributes:
        text: Original input text.
        label: Predicted sentiment label (positive/negative/neutral).
        score: Confidence score for the predicted label (0-1).
        scores: Dictionary of all class probabilities.
    """

    text: str
    label: str
    score: float
    scores: dict[str, float]

    @property
    def numeric_score(self) -> float:
        """Convert sentiment to numeric value for model input.

        Returns:
            Score between -1 (very negative) and 1 (very positive).
        """
        return (
            self.scores.get("positive", 0.0)
            - self.scores.get("negative", 0.0)
        )


@dataclass
class BatchSentimentResult:
    """Container for batch sentiment analysis results.

    Attributes:
        results: List of individual sentiment results.
    """

    results: list[SentimentResult]

    @property
    def mean_score(self) -> float:
        """Average numeric sentiment across all texts."""
        if not self.results:
            return 0.0
        return sum(r.numeric_score for r in self.results) / len(self.results)

    @property
    def label_distribution(self) -> dict[str, int]:
        """Count of each sentiment label."""
        dist: dict[str, int] = {"positive": 0, "negative": 0, "neutral": 0}
        for r in self.results:
            dist[r.label] = dist.get(r.label, 0) + 1
        return dist

    @property
    def summary(self) -> dict:
        """Summary statistics for API/logging."""
        return {
            "total_articles": len(self.results),
            "mean_sentiment": round(self.mean_score, 4),
            "distribution": self.label_distribution,
        }


class SentimentAnalyzer:
    """Financial sentiment analyzer using FinBERT.

    Provides single-text and batch inference with GPU acceleration
    when available. Includes caching to avoid re-analyzing identical texts.

    Architecture:
        Text → FinBERT Tokenizer → FinBERT Model → Softmax → Label + Score

    Example:
        >>> analyzer = SentimentAnalyzer()
        >>> result = analyzer.analyze("Apple reports record Q4 earnings")
        >>> print(f"{result.label}: {result.score:.2f}")
        positive: 0.94

        >>> batch = analyzer.analyze_batch(["Stock crashed", "Revenue soared"])
        >>> print(batch.summary)
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        max_length: int = 512,
        batch_size: int = 16,
        device: Optional[str] = None,
    ) -> None:
        """Initialize FinBERT sentiment analyzer.

        Args:
            model_name: HuggingFace model identifier.
            max_length: Maximum token length for input texts.
            batch_size: Batch size for batch inference.
            device: Device to run on ('cuda', 'mps', 'cpu', or None for auto).
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self._cache: dict[str, SentimentResult] = {}

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Load model and tokenizer
        logger.info(f"Loading FinBERT model: {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Label mapping for FinBERT
        self.labels = ["positive", "negative", "neutral"]

        logger.info(f"✅ FinBERT loaded successfully on {self.device}")

    @torch.no_grad()
    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of a single text.

        Args:
            text: Financial text to analyze.

        Returns:
            SentimentResult with label, score, and all class probabilities.
        """
        # Check cache
        cache_key = text[:200]  # Use first 200 chars as cache key
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        ).to(self.device)

        # Inference
        outputs = self.model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

        # Extract results
        scores = {
            label: float(prob)
            for label, prob in zip(self.labels, probabilities)
        }
        predicted_idx = probabilities.argmax().item()
        predicted_label = self.labels[predicted_idx]
        predicted_score = float(probabilities[predicted_idx])

        result = SentimentResult(
            text=text[:100] + "..." if len(text) > 100 else text,
            label=predicted_label,
            score=predicted_score,
            scores=scores,
        )

        # Cache result
        self._cache[cache_key] = result
        return result

    @torch.no_grad()
    def analyze_batch(self, texts: list[str]) -> BatchSentimentResult:
        """Analyze sentiment of multiple texts efficiently.

        Uses batch processing for GPU efficiency. This is significantly
        faster than analyzing texts one by one.

        Args:
            texts: List of financial texts to analyze.

        Returns:
            BatchSentimentResult with all individual results and aggregates.
        """
        if not texts:
            return BatchSentimentResult(results=[])

        logger.info(f"Analyzing sentiment for {len(texts)} texts...")
        all_results: list[SentimentResult] = []

        # Process in batches for memory efficiency
        for batch_start in range(0, len(texts), self.batch_size):
            batch_texts = texts[batch_start:batch_start + self.batch_size]

            # Check cache for each text
            uncached_texts: list[str] = []
            uncached_indices: list[int] = []
            batch_results: list[Optional[SentimentResult]] = [None] * len(batch_texts)

            for i, text in enumerate(batch_texts):
                cache_key = text[:200]
                if cache_key in self._cache:
                    batch_results[i] = self._cache[cache_key]
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)

            # Process uncached texts
            if uncached_texts:
                inputs = self.tokenizer(
                    uncached_texts,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True,
                    padding=True,
                ).to(self.device)

                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)

                for j, (idx, text) in enumerate(zip(uncached_indices, uncached_texts)):
                    probs = probabilities[j]
                    scores = {
                        label: float(prob)
                        for label, prob in zip(self.labels, probs)
                    }
                    predicted_idx = probs.argmax().item()

                    result = SentimentResult(
                        text=text[:100] + "..." if len(text) > 100 else text,
                        label=self.labels[predicted_idx],
                        score=float(probs[predicted_idx]),
                        scores=scores,
                    )
                    batch_results[idx] = result
                    self._cache[text[:200]] = result

            all_results.extend(r for r in batch_results if r is not None)

        batch_result = BatchSentimentResult(results=all_results)
        logger.info(
            f"  ✅ Sentiment analysis complete: "
            f"mean={batch_result.mean_score:.3f}, "
            f"distribution={batch_result.label_distribution}"
        )
        return batch_result

    def clear_cache(self) -> None:
        """Clear the sentiment cache."""
        self._cache.clear()
        logger.info("Sentiment cache cleared")

    def get_config(self) -> dict:
        """Return analyzer configuration for logging."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "cache_size": len(self._cache),
        }
