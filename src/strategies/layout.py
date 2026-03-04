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
        # Use BBox(x0, y0, x1, y1)
        mock_text_blocks = [
            TextBlock(
                text="Executive Summary: This document details the financial performance...",
                bounding_box=BBox(x0=50.0, y0=50.0, x1=550.0, y1=100.0),
                page_number=1,
                confidence=0.95
            ),
            TextBlock(
                text="Section 1: Revenue Analysis\nRevenue grew by 15% YoY mostly driven by...",
                bounding_box=BBox(x0=50.0, y0=110.0, x1=300.0, y1=400.0), # Left column example
                page_number=1,
                confidence=0.92
            ),
        ]
        
        mock_tables = [
            StructuredTable(
                data=[
                    ["Quarter", "Revenue (M)", "Growth"],
                    ["Q1", "12.5", "+5%"],
                    ["Q2", "13.1", "+4.8%"]
                ],
                bounding_box=BBox(x0=50.0, y0=450.0, x1=550.0, y1=600.0),
                page_number=1,
                caption="Financial Results 2024"
            )
        ]
        
        mock_figures = [
            Figure(
                alt_text="Bar chart showing revenue trends",
                bounding_box=BBox(x0=350.0, y0=110.0, x1=550.0, y1=300.0),
                page_number=1,
                image_ref="fig_1.png"
            )
        ]

        logger.info(f"LayoutExtractor extracted {len(mock_text_blocks)} text blocks, {len(mock_tables)} tables.")

        return ExtractedDocument(
            text_blocks=mock_text_blocks,
            structured_tables=mock_tables,
            figures=mock_figures,
            metadata={
                "extractor": "LayoutExtractor (Mock)",
                "source_tool": "Docling/MinerU",
                "strategy_history": ["LayoutExtractor"],
                "avg_confidence": 0.94
            }
        )
