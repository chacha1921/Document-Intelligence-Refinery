import json
import logging
import time
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from src.models import DocumentProfile, ExtractedDocument, ExtractionCost
from src.strategies.base import BaseExtractor
from src.strategies.fast_text import FastTextExtractor
from src.strategies.layout import LayoutExtractor
from src.strategies.vision import VisionExtractor

logger = logging.getLogger(__name__)

class ExtractionRouter:
    """
    Dynamic strategy router for document extraction.
    Loads thresholds and escalation policies from `rubric/extraction_rules.yaml`.
    """

    def __init__(self, config_path: str = "rubric/extraction_rules.yaml", ledger_path: str = ".refinery/extraction_ledger.jsonl"):
        """
        Initialize the ExtractionRouter with external configuration.
        """
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load Configuration
        self.config = self._load_config(config_path)
        self.router_config = self.config.get("router", {})
        self.strategies_config = self.config.get("strategies", {})
        
        # Initialize strategies with their specific config
        # Map strategy names from config to instances
        self.strategy_map = {
            "FastTextExtractor": FastTextExtractor(self.config.get("strategies", {}).get("fast_text", {})),
            "LayoutExtractor": LayoutExtractor(self.config.get("strategies", {}).get("layout", {})),
            "VisionExtractor": VisionExtractor(self.config.get("strategies", {}).get("vision", {}))
        }
        
        # Define escalation order (list of strategy names)
        # Default chain if not in config
        self.escalation_chain = self.router_config.get("escalation_chain", [
            "FastTextExtractor", "LayoutExtractor", "VisionExtractor"
        ])
        
        self.human_review_threshold = self.router_config.get("human_review_threshold", 0.6)

        logger.info(f"ExtractionRouter initialized. Strategy Chain: {self.escalation_chain}")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Loads YAML configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    def extract(self, file_path: str, profile: DocumentProfile) -> ExtractedDocument:
        """
        Routes extraction based on profile and automatically escalates if confidence is low.
        Tracks the full history in metadata.
        """
        start_time = time.time()
        
        # 1. Determine Starting Strategy Index
        start_index = self._get_start_index(profile)
        
        current_doc = None
        strategy_history = []
        
        # Iterate through escalation chain starting from the determined point
        for i in range(start_index, len(self.escalation_chain)):
            strategy_name = self.escalation_chain[i]
            strategy = self.strategy_map.get(strategy_name)
            
            if not strategy:
                logger.warning(f"Strategy {strategy_name} not found in map. Skipping.")
                continue
                
            logger.info(f"Attempting extraction with {strategy_name} (Attempt {i - start_index + 1})")
            
            try:
                # Execute Strategy
                attempt_start = time.time()
                result = strategy.extract(file_path)
                duration = time.time() - attempt_start
                
                # Calculate Confidence
                confidence = self._calculate_confidence(result)
                cost = self._estimate_cost(strategy, result)
                
                # Check Strategy-Specific Threshold
                # (Strategies usually check internally but we double check against config if needed)
                # The strategy config was passed to the strategy, so let's rely on the result's metadata or router's view
                
                # Get the threshold for this strategy from config
                # Need mapping name -> config key
                # Simple heuristic: "FastTextExtractor" -> "fast_text"
                config_key = self._get_config_key(strategy_name)
                threshold = self.strategies_config.get(config_key, {}).get("confidence_threshold", 0.8)
                
                status = "success"
                should_escalate = False
                
                if confidence < threshold:
                    status = "low_confidence_escalating"
                    should_escalate = True
                    logger.warning(f"{strategy_name} confidence {confidence:.2f} < threshold {threshold}. Escalating.")
                
                # Log this attempt
                self._log_attempt(file_path, strategy_name, confidence, cost, duration, status)
                
                # Update current best result
                # Merge history
                if current_doc:
                    existing_history = current_doc.metadata.get("strategy_history", [])
                    result.metadata["strategy_history"] = existing_history + [strategy_name]
                else:
                     result.metadata["strategy_history"] = [strategy_name]

                current_doc = result
                strategy_history.append(strategy_name)
                
                if not should_escalate:
                    # Success!
                    return current_doc
            
            except Exception as e:
                logger.error(f"Strategy {strategy_name} failed: {e}")
                self._log_attempt(file_path, strategy_name, 0.0, 0.0, 0.0, f"failed: {e}")
                # Continue loop to escalate
        
        # If we exit loop, we either succeeded (returned early) or exhausted chain
        # If we are here, we exhausted the chain or had failures
        
        if current_doc:
            # Check final confidence against human review threshold
            final_conf = self._calculate_confidence(current_doc)
            if final_conf < self.human_review_threshold:
                logger.warning(f"Final confidence {final_conf:.2f} < {self.human_review_threshold}. Flagging for human review.")
                current_doc.metadata["needs_human_review"] = True
            
            return current_doc
        
        # If no document produced at all (all crashed)
        raise RuntimeError("All extraction strategies failed.")

    def _get_start_index(self, profile: DocumentProfile) -> int:
        """Maps profile cost to index in the escalation chain."""
        # Simple mapping assuming standard chain [FastText, Layout, Vision]
        # This implementation assumes the standard order in config.
        # Ideally, we'd search the list for the mapped strategy.
        
        target_strategy = "FastTextExtractor" # Default
        
        if profile.estimated_extraction_cost == ExtractionCost.NEEDS_LAYOUT_MODEL:
            target_strategy = "LayoutExtractor"
        elif profile.estimated_extraction_cost == ExtractionCost.NEEDS_VISION_MODEL:
            target_strategy = "VisionExtractor"
            
        try:
            return self.escalation_chain.index(target_strategy)
        except ValueError:
            logger.warning(f"Target strategy {target_strategy} not in escalation chain. Defaulting to start.")
            return 0

    def _get_config_key(self, strategy_name: str) -> str:
        """Helper to map class name to config key."""
        if "FastText" in strategy_name: return "fast_text"
        if "Layout" in strategy_name: return "layout"
        if "Vision" in strategy_name: return "vision"
        return "unknown"

    def _calculate_confidence(self, doc: ExtractedDocument) -> float:
        """
        Aggregates confidence from the extracted document metadata.
        """
        # Prefer specific metadata field "avg_confidence" if strategies set it
        if "avg_confidence" in doc.metadata:
            return float(doc.metadata["avg_confidence"])
            
        # Fallback to block avg
        if doc.text_blocks:
            return sum(b.confidence for b in doc.text_blocks) / len(doc.text_blocks)
            
        return 0.0

    def _estimate_cost(self, strategy: BaseExtractor, doc: ExtractedDocument) -> float:
        """
        Returns estimated USD cost for the strategy run.
        """
        # Check metadata first
        if "cost_usd" in doc.metadata:
            return doc.metadata["cost_usd"]
            
        # Fallback estimates
        if isinstance(strategy, FastTextExtractor):
            return 0.0001
        if isinstance(strategy, LayoutExtractor):
            return 0.01 
        return 0.0

    def _log_attempt(self, file_path: str, strategy_name: str, confidence: float, 
                     cost_estimate: float, duration: float, status: str):
        """Appends a record to the JSONL ledger."""
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
