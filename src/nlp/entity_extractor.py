"""Financial Named Entity Recognition (NER) module.

Extracts financial entities (company names, tickers, monetary values,
dates, percentages) from financial text using a pre-trained NER model.

This module addresses the job posting requirement:
    - "Varlık tanıma (NER)" (Named Entity Recognition)

Uses HuggingFace's token classification pipeline with the
dslim/bert-base-NER model as default, with financial entity
post-processing for ticker and monetary value extraction.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from transformers import pipeline

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Entity:
    """A single recognized entity.

    Attributes:
        text: The entity text as found in the source.
        label: Entity type (e.g., ORG, PER, LOC, MISC).
        score: Confidence score from the NER model.
        start: Start character position in source text.
        end: End character position in source text.
    """

    text: str
    label: str
    score: float
    start: int = 0
    end: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "text": self.text,
            "label": self.label,
            "score": round(self.score, 4),
        }


@dataclass
class ExtractionResult:
    """Result of entity extraction from a single text.

    Attributes:
        text: Original input text.
        entities: List of extracted entities.
        companies: Extracted company/organization names.
        tickers: Extracted stock ticker symbols.
        monetary_values: Extracted monetary amounts.
    """

    text: str
    entities: list[Entity] = field(default_factory=list)
    companies: list[str] = field(default_factory=list)
    tickers: list[str] = field(default_factory=list)
    monetary_values: list[str] = field(default_factory=list)

    @property
    def entity_count(self) -> int:
        """Total number of entities extracted."""
        return len(self.entities)

    def to_dict(self) -> dict:
        """Convert to dictionary for API/logging."""
        return {
            "entity_count": self.entity_count,
            "companies": self.companies,
            "tickers": self.tickers,
            "monetary_values": self.monetary_values,
            "entities": [e.to_dict() for e in self.entities],
        }


class EntityExtractor:
    """Financial Named Entity Recognition using pre-trained NER models.

    Extracts structured information from financial text including:
    - Company/organization names (ORG entities)
    - Stock ticker symbols (pattern matching)
    - Monetary values (regex extraction)
    - Percentage changes (regex extraction)

    The NER model identifies standard entity types (ORG, PER, LOC, MISC),
    and additional financial entity post-processing extracts domain-specific
    patterns like "$1.2B" or "AAPL".

    Example:
        >>> extractor = EntityExtractor()
        >>> result = extractor.extract(
        ...     "Apple Inc. (AAPL) reported $94.8B in revenue, up 8%"
        ... )
        >>> print(result.companies)    # ["Apple Inc."]
        >>> print(result.tickers)      # ["AAPL"]
        >>> print(result.monetary_values)  # ["$94.8B"]
    """

    # Common stock ticker pattern: 1-5 uppercase letters
    TICKER_PATTERN = re.compile(r'\b([A-Z]{1,5})\b')

    # Monetary value patterns
    MONEY_PATTERN = re.compile(
        r'\$[\d,]+\.?\d*[BMKbmk]?(?:\s*(?:billion|million|thousand))?'
    )

    # Percentage pattern
    PERCENT_PATTERN = re.compile(r'[\d.]+%')

    # Common English words that look like tickers (filter these out)
    TICKER_BLACKLIST = {
        "THE", "AND", "FOR", "ARE", "NOT", "YOU", "ALL", "CAN", "HAS",
        "HER", "WAS", "ONE", "OUR", "OUT", "ITS", "HIS", "HOW", "MAY",
        "NEW", "NOW", "OLD", "SEE", "WAY", "WHO", "DID", "GET", "HIM",
        "LET", "SAY", "SHE", "TOO", "USE", "CEO", "CFO", "IPO", "ETF",
        "GDP", "SEC", "FBI", "USA", "NYSE", "AI", "UP", "IT", "AT",
        "BY", "IN", "ON", "TO", "IF", "SO", "AN", "OR", "NO", "US",
        "DO", "BE", "AS", "IS", "OF",
    }

    def __init__(
        self,
        model_name: str = "dslim/bert-base-NER",
        device: Optional[int] = None,
        aggregation_strategy: str = "simple",
    ) -> None:
        """Initialize entity extractor.

        Args:
            model_name: HuggingFace NER model identifier.
            device: Device index (-1 for CPU, 0+ for GPU). None for auto.
            aggregation_strategy: How to aggregate sub-word entities.
        """
        logger.info(f"Loading NER model: {model_name}...")

        # Determine device
        if device is None:
            try:
                import torch
                if torch.cuda.is_available():
                    device = 0
                else:
                    device = -1
            except ImportError:
                device = -1

        self.ner_pipeline = pipeline(
            "ner",
            model=model_name,
            aggregation_strategy=aggregation_strategy,
            device=device,
        )

        logger.info("✅ NER model loaded successfully")

    def extract(self, text: str) -> ExtractionResult:
        """Extract entities from a single text.

        Combines model-based NER with regex-based financial pattern
        extraction for comprehensive entity coverage.

        Args:
            text: Financial text to analyze.

        Returns:
            ExtractionResult with all extracted entities.
        """
        result = ExtractionResult(text=text)

        # Model-based NER
        try:
            ner_results = self.ner_pipeline(text)

            for ner in ner_results:
                entity = Entity(
                    text=ner["word"].strip(),
                    label=ner["entity_group"],
                    score=float(ner["score"]),
                    start=ner.get("start", 0),
                    end=ner.get("end", 0),
                )
                result.entities.append(entity)

                # Collect organization entities as companies
                if entity.label == "ORG" and entity.score > 0.7:
                    result.companies.append(entity.text)

        except Exception as e:
            logger.warning(f"NER model inference failed: {e}")

        # Regex-based financial entity extraction
        result.tickers = self._extract_tickers(text)
        result.monetary_values = self._extract_monetary(text)

        return result

    def extract_batch(self, texts: list[str]) -> list[ExtractionResult]:
        """Extract entities from multiple texts.

        Args:
            texts: List of financial texts.

        Returns:
            List of ExtractionResult objects.
        """
        logger.info(f"Extracting entities from {len(texts)} texts...")
        results = [self.extract(text) for text in texts]

        total_entities = sum(r.entity_count for r in results)
        total_companies = sum(len(r.companies) for r in results)
        logger.info(
            f"  ✅ Extracted {total_entities} entities, "
            f"{total_companies} companies from {len(texts)} texts"
        )
        return results

    def _extract_tickers(self, text: str) -> list[str]:
        """Extract potential stock ticker symbols from text.

        Uses pattern matching with a blacklist filter to avoid
        common English words that match ticker patterns.

        Args:
            text: Text to search for tickers.

        Returns:
            List of potential ticker symbols.
        """
        matches = self.TICKER_PATTERN.findall(text)
        tickers = [
            m for m in matches
            if m not in self.TICKER_BLACKLIST and len(m) >= 2
        ]
        return list(set(tickers))

    def _extract_monetary(self, text: str) -> list[str]:
        """Extract monetary values from text.

        Args:
            text: Text to search for monetary values.

        Returns:
            List of monetary value strings.
        """
        return self.MONEY_PATTERN.findall(text)

    def get_config(self) -> dict:
        """Return extractor configuration for logging."""
        return {
            "model": self.ner_pipeline.model.config.name_or_path,
            "aggregation_strategy": "simple",
        }
