import logging
import os
from typing import List, Dict, Any, Optional
from src.strategies.base import BaseExtractor
from src.models import ExtractedDocument, TextBlock, StructuredTable, Figure, BBox

logger = logging.getLogger(__name__)

class BudgetGuard:
    """
    Simple budget guard to track token usage and prevent overspending.
    """
    def __init__(self, max_tokens: int = 10000, max_cost_usd: float = 1.0, cost_input: float = 0.00015, cost_output: float = 0.0006):
        self.max_tokens = max_tokens
        self.max_cost_usd = max_cost_usd
        self.current_tokens = 0
        self.current_cost = 0.0
        self.cost_per_1k_input = cost_input
        self.cost_per_1k_output = cost_output

    def check_budget(self, estimated_tokens: int) -> bool:
        if self.current_tokens + estimated_tokens > self.max_tokens:
            logger.warning("Budget limit reached (tokens).")
            return False
        
        estimated_cost = (estimated_tokens / 1000) * self.cost_per_1k_input 
        if self.current_cost + estimated_cost > self.max_cost_usd:
             logger.warning("Budget limit reached (cost).")
             return False

        return True

    def update_usage(self, input_tokens: int, output_tokens: int):
        self.current_tokens += (input_tokens + output_tokens)
        cost = (input_tokens / 1000) * self.cost_per_1k_input + \
               (output_tokens / 1000) * self.cost_per_1k_output
        self.current_cost += cost
        logger.info(f"Updated usage: {self.current_tokens} tokens, ${self.current_cost:.4f}")


class VisionExtractor(BaseExtractor):
    """
    Strategy C: Vision-Language Model (VLM) Extraction.
    Uses GPT-4o-mini (via OpenRouter/OpenAI) to "see" the document.
    Includes budget protection and relies on config.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model_name = self.config.get("model", "gpt-4o-mini")
        self.budget_guard = BudgetGuard(
            max_tokens=self.config.get("max_tokens", 20000), 
            max_cost_usd=self.config.get("max_cost_usd", 2.0),
            cost_input=self.config.get("cost_per_1k_input", 0.00015),
            cost_output=self.config.get("cost_per_1k_output", 0.0006)
        )
        self.api_key = os.getenv("OPENAI_API_KEY") 

    def extract(self, file_path: str) -> ExtractedDocument:
        logger.info(f"VisionExtractor processing: {file_path}")
        
        # 1. Check Budget
        estimated_input_tokens = 1000 
        if not self.budget_guard.check_budget(estimated_input_tokens):
            raise RuntimeError("Vision extraction skipped due to budget constraints.")

        response_content = ""
        usage = {"prompt_tokens": 0, "completion_tokens": 0}

        try:
            if self.api_key:
                # Real call placeholder
                response_content = (
                    "Executive Summary\n"
                    "The document outlines Q3 financial performance.\n\n"
                    "| Metric | Value |\n|---|---|\n| Revenue | $5M |"
                )
                usage["prompt_tokens"] = 800
                usage["completion_tokens"] = 150
                logger.info("Vision API call simulated.")
            else:
                logger.warning("No OPENAI_API_KEY found. Returning mock VLM data.")
                response_content = "Mock Vision Content: Extracted text from image analysis."
                usage = {"prompt_tokens": 500, "completion_tokens": 50}

            # 3. Update Budget
            self.budget_guard.update_usage(usage["prompt_tokens"], usage["completion_tokens"])

            # 4. Parse Response (Demo logic)
            lines = response_content.split('\n')
            text_blocks = []
            tables = []
            
            # Use dummy BBox for VLM output as it doesn't always give coordinates
            dummy_bbox = BBox(x0=0.0, y0=0.0, x1=100.0, y1=100.0)
            
            for line in lines:
                if "|" in line: 
                    # Detect markdown table row
                    cells = [c.strip() for c in line.split('|') if c.strip()]
                    if cells:
                         if not tables:
                             tables.append(StructuredTable(
                                 data=[], bounding_box=dummy_bbox, page_number=1
                             ))
                         tables[-1].data.append(cells)
                else:
                    if line.strip():
                        text_blocks.append(TextBlock(
                            text=line.strip(),
                            bounding_box=dummy_bbox, 
                            page_number=1,
                            confidence=0.8
                        ))

            return ExtractedDocument(
                text_blocks=text_blocks,
                structured_tables=tables,
                figures=[], 
                metadata={
                    "extractor": "VisionExtractor",
                    "model": self.model_name,
                    "cost_usd": self.budget_guard.current_cost,
                    "strategy_history": ["VisionExtractor"],
                    "avg_confidence": 0.8 # Placeholder
                }
            )

        except Exception as e:
            logger.error(f"Vision extraction failed: {e}")
            raise
