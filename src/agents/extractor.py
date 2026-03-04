import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from src.models import DocumentProfile, ExtractedDocument, ExtractionCost
from src.strategies.base import BaseExtractor
from src.strategies.fast_text import FastTextExtractor
from src.strategies.layout import LayoutExtractor
from src.strategies.vision import VisionExtractor

logger = logging.getLogger(__name__)

class ExtractionRouter:
    """
    Routes document extraction to the appropriate strategy based on the DocumentProfile.
    Handles escalation if the initial strategy fails or yields low confidence.
    Logs all attempts to a ledger.
    """

    def __init__(self, ledger_path: str = ".refinery/extraction_ledger.jsonl"):
        """
        Initialize the ExtractionRouter.
        
        Args:
            ledger_path (str): Path to the JSONL ledger file for logging attempts.
        """
        self.ledger_path = Path(ledger_path)
        
        # Ensure ledger directory exists
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize strategies
        # In a production system, these might be injected or lazily loaded
        self.fast_text = FastTextExtractor()
        self.layout = LayoutExtractor()
        # Vision usually needs API keys from env, passing simple config here if needed
        self.vision = VisionExtractor(config={"model": "gpt-4o-mini"}) 

        logger.info(f"ExtractionRouter initialized. Ledger: {self.ledger_path}")

    def extract(self, file_path: str, profile: DocumentProfile) -> ExtractedDocument:
        """
        Selects a strategy, extracts content, handles escalation, and logs the attempt.
        
        Args:
            file_path (str): Path to the document.
            profile (DocumentProfile): The profile from TriageAgent.
            
        Returns:
            ExtractedDocument: The final extracted content.
        """
        start_time = time.time()
        
        # 1. Select Initial Strategy
        strategy = self._select_strategy(profile)
        strategy_name = strategy.__class__.__name__
        
        logger.info(f"Selected initial strategy: {strategy_name} for {file_path}")
        
        try:
            # 2. Attempt Extraction
            result = strategy.extract(file_path)
            confidence = self._calculate_confidence(result)
            
            # 3. Escalation Guard
            # If FastText was used but confidence is low, escalate to Layout
            if isinstance(strategy, FastTextExtractor) and confidence < 0.8:
                logger.warning(f"FastText confidence low ({confidence:.2f}). Escalating to LayoutExtractor.")
                
                # Log the failed/insufficient attempt
                self._log_attempt(
                    file_path=file_path,
                    strategy_name=strategy_name,
                    confidence=confidence,
                    cost_estimate=0.0, # FastText assumed negligible
                    duration=time.time() - start_time,
                    status="escalated_low_confidence"
                )
                
                # Switch Strategy
                strategy = self.layout
                strategy_name = "LayoutExtractor (Escalated)"
                start_time = time.time() # Reset timer for new attempt
                
                # Retry
                result = strategy.extract(file_path)
                confidence = 1.0 # Layout usually assumed higher confidence for now
            
            # 4. Final Logging
            duration = time.time() - start_time
            cost_estimate = self._estimate_cost(strategy, result)
            
            self._log_attempt(
                file_path=file_path,
                strategy_name=strategy_name,
                confidence=confidence,
                cost_estimate=cost_estimate,
                duration=duration,
                status="success"
            )
            
            return result

        except Exception as e:
            logger.error(f"Extraction failed with strategy {strategy_name}: {e}")
            
            # Log failure
            self._log_attempt(
                file_path=file_path,
                strategy_name=strategy_name,
                confidence=0.0,
                cost_estimate=0.0,
                duration=time.time() - start_time,
                status=f"failed: {str(e)}"
            )
            raise

    def _select_strategy(self, profile: DocumentProfile) -> BaseExtractor:
        """Determines the best strategy based on cost/complexity profile."""
        if profile.estimated_extraction_cost == ExtractionCost.FAST_TEXT_SUFFICIENT:
            return self.fast_text
        elif profile.estimated_extraction_cost == ExtractionCost.NEEDS_LAYOUT_MODEL:
            return self.layout
        elif profile.estimated_extraction_cost == ExtractionCost.NEEDS_VISION_MODEL:
            return self.vision
        else:
            # Default fallback
            return self.layout

    def _calculate_confidence(self, doc: ExtractedDocument) -> float:
        """
        Aggregates confidence from the extracted document.
        """
        # If metadata has generic score
        if "confidence" in doc.metadata:
            return float(doc.metadata["confidence"])
            
        # Check specific metadata from FastText
        low_conf_pages = doc.metadata.get("low_confidence_pages", [])
        if low_conf_pages:
            # Heuristic: if any page is low confidence, overall result is shaky
            # Return average confidence of blocks if available, otherwise penalty
            if doc.text_blocks:
                return sum(b.confidence for b in doc.text_blocks) / len(doc.text_blocks)
            return 0.4
            
        # Default high confidence if no flags
        return 1.0

    def _estimate_cost(self, strategy: BaseExtractor, doc: ExtractedDocument) -> float:
        """
        Returns estimated USD cost for the valid strategy run.
        """
        if isinstance(strategy, FastTextExtractor):
            return 0.0001 # Micro-cost for CPU time abstraction
        if isinstance(strategy, LayoutExtractor):
            return 0.01 # Mock cost per page/doc for deep learning inference
        if isinstance(strategy, VisionExtractor):
            # Vision extractor might put cost in metadata
            return doc.metadata.get("cost_usd", 0.0)
        return 0.0

    def _log_attempt(self, file_path: str, strategy_name: str, confidence: float, 
                     cost_estimate: float, duration: float, status: str):
        """
        Appends a record to the JSONL ledger.
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "file_path": str(file_path),
            "strategy": strategy_name,
            "confidence": round(confidence, 4),
            "cost_estimate_usd": round(cost_estimate, 6),
            "duration_seconds": round(duration, 4),
            "status": status
        }
        
        try:
            with open(self.ledger_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to extraction ledger: {e}")

if __name__ == "__main__":
    # Test stub
    print("Router module loaded.")
