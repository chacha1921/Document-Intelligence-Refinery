from typing import List, Any, Optional, Dict
import logging
from src.strategies.base import BaseExtractor
from src.models import ExtractedDocument, TextBlock, StructuredTable, Figure, BBox

logger = logging.getLogger(__name__)

class LayoutExtractor(BaseExtractor):
    """
    Strategy B: Layout-Analysis Extraction (Mocked Docling/MinerU).
    Uses a sophisticated layout model to handle multi-column text, tables, and figures correctly.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def extract(self, file_path: str) -> ExtractedDocument:
        logger.info(f"LayoutExtractor (Mock) processing: {file_path}")
        
        # 1. Simulate parsed document structure
        # In a real implementation:
        # - Call layout model (e.g. Docling)
        # - Convert its output format to our internal schema (TextBlock, StructuredTable, Figure)
        # - CRITICAL: Ensure correct reading order (top-left -> bottom-right for columns, etc.)
        
        raw_blocks = [
            # Block 1: Executive Summary (Full Width)
            TextBlock(
                text="Executive Summary: This document details the financial performance...",
                bounding_box=BBox(x0=50.0, y0=50.0, x1=550.0, y1=100.0), # Top of page
                page_number=1
            ),
            # Block 2: Col 1 Text (Left)
            TextBlock(
                text="Section 1: Revenue Analysis\nRevenue grew by 15% YoY mostly driven by...",
                bounding_box=BBox(x0=50.0, y0=110.0, x1=300.0, y1=400.0), 
                page_number=1
            ),
            # Block 3: Col 2 Figure (Right) - Should appear AFTER Block 2 in purely spatial sort if treating as columns
            # But strictly top-down, y0 might be same. Let's assume standard reading order: Col 1 then Col 2
        ]
        
        mock_tables = [
            StructuredTable(
                data=[
                    ["Quarter", "Revenue (M)", "Growth"],
                    ["Q1", "12.5", "+5%"],
                    ["Q2", "13.1", "+4.8%"]
                ],
                bounding_box=BBox(x0=50.0, y0=450.0, x1=550.0, y1=600.0), # Bottom of page
                page_number=1,
                caption="Financial Results 2024"
            )
        ]
        
        mock_figures = [
            Figure(
                alt_text="Bar chart showing revenue trends", # Semantic desc
                bounding_box=BBox(x0=350.0, y0=110.0, x1=550.0, y1=300.0), # Right column
                page_number=1,
                image_ref="fig_1.png"
            )
        ]

        # 2. Sort by Reading Order
        # Standard heuristic: Sort by Page -> Top (y0) -> Left (x0)
        # For multi-column, advanced logic might define columns first.
        # Here we use a stable sort:
        sorted_text_blocks = sorted(raw_blocks, key=lambda b: (b.page_number, b.bounding_box.y0, b.bounding_box.x0))

        logger.info(f"LayoutExtractor extracted {len(sorted_text_blocks)} text blocks, {len(mock_tables)} tables.")

        return ExtractedDocument(
            text_blocks=sorted_text_blocks,
            structured_tables=mock_tables,
            figures=mock_figures,
            metadata={
                "extractor": "LayoutExtractor (Mock)",
                "source_tool": "Docling/MinerU",
                "strategy_history": ["LayoutExtractor"],
                "avg_confidence": 0.94,
                "reading_order_preserved": True
            }
        )
