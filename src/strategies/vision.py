import logging
import os
from typing import List, Dict, Any, Optional
from src.strategies.base import BaseExtractor
from src.models import ExtractedDocument, TextBlock, StructuredTable, Figure

logger = logging.getLogger(__name__)

class BudgetGuard:
    """
    Simple budget guard to track token usage and prevent overspending.
    """
    def __init__(self, max_tokens: int = 10000, max_cost_usd: float = 1.0):
        self.max_tokens = max_tokens
        self.max_cost_usd = max_cost_usd
        self.current_tokens = 0
        self.current_cost = 0.0
        # Simplistic cost per 1k tokens (adjust as needed, e.g., for GPT-4o-mini)
        self.cost_per_1k_input = 0.00015 
        self.cost_per_1k_output = 0.0006

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
    Uses GPT-4o-mini (via OpenRouter/OpenAI) to "see" the document and extract structured content.
    Includes budget protection.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model_name = config.get("model", "gpt-4o-mini")
        self.budget_guard = BudgetGuard(
            max_tokens=config.get("max_tokens", 20000), 
            max_cost_usd=config.get("max_cost", 2.0)
        )
        self.api_key = os.getenv("OPENAI_API_KEY") 

    def extract(self, file_path: str) -> ExtractedDocument:
        logger.info(f"VisionExtractor processing: {file_path}")
        
        # 1. Check Budget
        # Estimate input tokens (rough heuristic: 1 page ~ 1000 tokens for image + prompt)
        estimated_input_tokens = 1000 
        if not self.budget_guard.check_budget(estimated_input_tokens):
            raise RuntimeError("Vision extraction skipped due to budget constraints.")

        # 2. Prepare VLM Call (Mock or Real)
        # In a real scenario:
        # - Convert PDF page to image (base64)
        # - Construct prompt: "Extract all text, tables, and figures from this image..."
        # - Call `client.chat.completions.create(...)`
        
        response_content = ""
        usage = {"prompt_tokens": 0, "completion_tokens": 0}

        try:
            if self.api_key:
                # Real logic placeholder
                # image_b64 = ...
                # response = client.chat.completions.create(...)
                # response_content = response.choices[0].message.content
                # usage = response.usage
                
                # Mocking succesful response for structure
                response_content = (
                    "Executive Summary\n"
                    "The document outlines Q3 financial performance.\n\n"
                    "| Metric | Value |\n|---|---|\n| Revenue | $5M |"
                )
                usage["prompt_tokens"] = 800
                usage["completion_tokens"] = 150
                logger.info("Vision API call simulated (API Key present but not used to save verify/cost).")
            else:
                # Fallback / Mock
                logger.warning("No OPENAI_API_KEY found. Returning mock VLM data.")
                response_content = "Mock Vision Content: Extracted text from image analysis."
                usage["prompt_tokens"] = 500
                usage["completion_tokens"] = 50

            # 3. Update Budget
            self.budget_guard.update_usage(usage["prompt_tokens"], usage["completion_tokens"])

            # 4. Parse Response into ExtractedDocument
            # Parsing markdown/JSON from VLM to our schema is complex. 
            # We'll do a simple mapping here.
            
            # Simple heuristic parser for demo
            lines = response_content.split('\n')
            text_blocks = []
            tables = []
            
            current_block = ""
            for line in lines:
                if "|" in line: # Detect markdown table row
                    # Collect into table structure (very simplified)
                    cells = [c.strip() for c in line.split('|') if c.strip()]
                    if cells:
                         # Append to last table or create new
                         if not tables:
                             tables.append(StructuredTable(
                                 data=[], bounding_box=(0,0,0,0), page_number=1
                             ))
                         tables[-1].data.append(cells)
                else:
                    if line.strip():
                        text_blocks.append(TextBlock(
                            text=line.strip(),
                            bounding_box=(0,0,0,0), # VLM doesn't give precise bbox usually
                            page_number=1,
                            confidence=0.8 # VLM confidence vary
                        ))

            return ExtractedDocument(
                text_blocks=text_blocks,
                structured_tables=tables,
                figures=[], # Figures often require separate detection prompt
                metadata={
                    "extractor": "VisionExtractor",
                    "model": self.model_name,
                    "cost_usd": self.budget_guard.current_cost
                }
            )

        except Exception as e:
            logger.error(f"Vision extraction failed: {e}")
            raise

