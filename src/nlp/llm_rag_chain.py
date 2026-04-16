"""Financial Language Model (RAG) System.

This module utilizes LangChain and Ollama (Local LLM) to generate
executive financial summaries based on quantitative price predictions
and qualitative news sentiments.

Features:
- Connects to localized `llama3` via Ollama for zero-cost secure inference.
- Constructs an expert persona prompt for accurate Wall-Street style analysis.
- Consolidates quantitative model results with context to perform RAG.
"""

from typing import List, Dict, Any, Optional

try:
    from langchain_community.chat_models import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FinancialRAGSystem:
    """Retrieval-Augmented Generation for Stock Analysis."""
    
    def __init__(self, model_name: str = "llama3", temperature: float = 0.2):
        """Initialize Local RAG system via Ollama.
        
        Args:
            model_name: The name of the local Ollama model.
            temperature: Creativity coefficient (low for finance).
        """
        self.model_name = model_name
        
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain is not installed. LLM RAG features disabled.")
            self.llm = None
            return
            
        try:
            self.llm = ChatOllama(model=model_name, temperature=temperature)
            logger.info(f"Initialized Local LLM via Ollama: {model_name}")
            self._build_chain()
        except Exception as e:
            logger.error(f"Failed to connect to local Ollama. Ensure `ollama serve` is running. Error: {e}")
            self.llm = None
            
    def _build_chain(self):
        """Build the LangChain LCEL pipeline."""
        system_prompt = (
            "You are a senior quantitative financial analyst at a top-tier hedge fund. "
            "Your job is to read raw numerical predictions from our deep learning ensemble "
            "(Transformer + LSTM) and the FinBERT sentiment analysis of recent news, "
            "and provide a concise, highly professional executive summary. "
            "Do NOT use generic financial disclaimers like 'I am an AI...'. "
            "Be direct, analytical, and structured."
        )
        
        human_prompt = (
            "Ticker: {ticker}\n"
            "Forecast Horizon: {days} days\n"
            "Predicted Price Trajectory: {predictions}\n"
            "Mean Sentiment Score (-1 to 1): {sentiment}\n\n"
            "Recent News Context:\n{news}\n\n"
            "Provide your Executive Investment Strategy based solely on this provided context:"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
        
        self.chain = prompt | self.llm | StrOutputParser()
        
    def generate_report(
        self,
        ticker: str,
        predictions: List[float],
        sentiment_score: Optional[float],
        news_texts: List[str],
        days: int = 5
    ) -> str:
        """Generate the RAG analysis report using the underlying LLM.
        
        Args:
            ticker: The stock ticker (e.g., AAPL).
            predictions: Predicted prices.
            sentiment_score: Aggregate sentiment from FinBERT.
            news_texts: List of top news articles currently relating to the ticker.
            days: Forecast duration.
            
        Returns:
            A tailored markdown financial report.
        """
        if not self.llm:
            return "Local LLM functionality is currently disabled or LangChain is missing."
            
        logger.info(f"Requesting RAG LLM summary for {ticker}...")
        
        try:
            formatted_news = "\n- ".join(news_texts) if news_texts else "No significant recent news available."
            prediction_str = ", ".join([f"${p:.2f}" for p in predictions])
            
            # Execute Chain
            report = self.chain.invoke({
                "ticker": ticker,
                "days": days,
                "predictions": prediction_str,
                "sentiment": f"{sentiment_score:.2f}" if sentiment_score else "N/A",
                "news": formatted_news
            })
            
            return report
        except Exception as e:
            logger.error(f"Error during RAG generation: {e}")
            return f"Failed to generate analysis. Ensure Ollama {self.model_name} is running natively."

