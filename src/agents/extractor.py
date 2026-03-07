"""
Stage 2: Extraction Router Agent.
Reads a DocumentProfile and routes the PDF to the appropriate extraction strategy.
Implements Escalation Guard logic.
""" 
import os
import time
import json
import logging
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

from src.models.schemas import DocumentProfile, ExtractedDocument
from src.strategies.extractors import FastTextExtractor, LayoutExtractor, VisionExtractor

# Configure logging (can be moved to main)
logger = logging.getLogger(__name__)

class ExtractionRouter:
    """
    Orchestrates the extraction process.
    - Decides initial strategy based on DocumentProfile.
    - Escalates if confidence drops below thresholds (from Rules YAML).
    - Logs all attempts to an audit ledger.
    """
    def __init__(self, rules_path: str = "rubric/extraction_rules.yaml"):
        self.rules = self._load_rules(rules_path)
        self.ledger_path = Path(".refinery/extraction_ledger.jsonl")
        
        # Initialize strategy instances
        self.fast_extractor = FastTextExtractor()
        self.layout_extractor = LayoutExtractor() # Docling Placeholder
        self.vision_extractor = VisionExtractor(token_limit=50000) # Vision Placeholder

    def _load_rules(self, path: str) -> Dict[str, Any]:
        """Loads escalation rules from YAML."""
        if not os.path.exists(path):
            logger.warning(f"Rules file {path} not found. Using defaults.")
            return {"strategies": {"fast_text": {"confidence_threshold": 0.8}, "layout": {"confidence_threshold": 0.9}}}
        with open(path, "r") as f:
            return yaml.safe_load(f)


    def extract(self, pdf_path: str, profile: Union[DocumentProfile, dict]) -> ExtractedDocument:
        """
        Main extraction flow.
        """
        doc_id = Path(pdf_path).stem
        
        # Determine strategy name
        cost_est = "needs_layout_model" # default
        if isinstance(profile, dict):
             cost_est = profile.get("estimated_extraction_cost", "needs_layout_model")
        elif profile:
             cost_est = profile.estimated_extraction_cost

        if cost_est == "fast_text_sufficient":
            initial_strategy_name = "fast_text"
        elif cost_est == "needs_vision_model":
            initial_strategy_name = "vision"
        else:
            initial_strategy_name = "layout"

        logger.info(f"Targeting strategy: {initial_strategy_name} for document {doc_id}")
        
        # 2. Executing Initial Strategy
        extracted_doc, confidence = self._execute_strategy(initial_strategy_name, pdf_path)
        
        # 3. Escalation Guard Logic
        # Check against rules if we started with a "cheap" strategy
        if initial_strategy_name == "fast_text":
            threshold = 0.8
            if self.rules and "strategies" in self.rules:
                 threshold = self.rules["strategies"].get("fast_text", {}).get("confidence_threshold", 0.8)
            
            if confidence < threshold:
                logger.warning(f"Strategy 'fast_text' failed confidence check ({confidence:.2f} < {threshold}). Escalating to 'layout'.")
                
                # Escalation: Retry with Layout Extractor
                extracted_doc, confidence = self._execute_strategy("layout", pdf_path)
                
        return extracted_doc

    def _execute_strategy(self, strategy_name: str, file_path: str) -> tuple[ExtractedDocument, float]:
        """
        Runs the chosen strategy.
        Returns result and confidence.
        """
        start_time = time.time()
        
        extractor = None
        if strategy_name == "fast_text":
            extractor = self.fast_extractor
        elif strategy_name == "layout":
            extractor = self.layout_extractor
        elif strategy_name == "vision":
            extractor = self.vision_extractor
            
        if not extractor:
            logger.error(f"Unknown strategy: {strategy_name}")
            return ExtractedDocument(doc_id="error"), 0.0

        try:
            result, confidence = extractor.extract(file_path)
            
            # Log this attempt immediately
            self._log_attempt(Path(file_path).stem, strategy_name, confidence, start_time)
            
            return result, confidence
        except Exception as e:
            logger.error(f"Strategy {strategy_name} execution failed: {e}")
            self._log_attempt(Path(file_path).stem, strategy_name, 0.0, start_time)
            return ExtractedDocument(doc_id="error"), 0.0

    def _log_attempt(self, doc_id: str, strategy: str, confidence: float, start_time: float):
        """
        Appends extraction attempt details to the ledger.
        """
        duration = time.time() - start_time
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "doc_id": doc_id,
            "strategy_used": strategy,
            "confidence_score": confidence,
            "processing_time_seconds": round(duration, 4),
            "status": "success" if confidence > 0 else "failed"
        }
        
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

# Simple main block for direct testing
if __name__ == "__main__":
    import sys
    # Mock profile for quick test without running Triage
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        dummy_profile = DocumentProfile(
            origin_type="native_digital",
            layout_complexity="single_column", # Should trigger fast_text
            language="en",
            domain_hint="general",
            estimated_extraction_cost="fast_text_sufficient"
        )
        
        router = ExtractionRouter()
        doc = router.extract(pdf_path, dummy_profile)
        print(f"Extraction result for {doc.doc_id}: {len(doc.text_blocks)} text blocks.")
