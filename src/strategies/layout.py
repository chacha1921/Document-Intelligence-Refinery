from typing import List, Any
import logging
from src.strategies.base import BaseExtractor
from src.models import ExtractedDocument, TextBlock, StructuredTable, Figure

logger = logging.getLogger(__name__)

class LayoutExtractor(BaseExtractor):
    """
    Strategy B: Layout-Analysis Extraction (Mocked Docling/MinerU).
    Uses a sophisticated layout model to handle multi-column text, tables, and figures correctly.
    """

    def extract(self, file_path: str) -> ExtractedDocument:
        logger.info(f"LayoutExtractor (Mock) processing: {file_path}")
        
        # In a real implementation, this would call `docling.convert()` or similar.
        # For this exercise, we simulate the output structure.
        
        # 1. Simulate parsed document structure
        mock_text_blocks = [
            TextBlock(
                text="Executive Summary: This document details the financial performance...",
                bounding_box=(50.0, 50.0, 550.0, 100.0),
                page_number=1,
                confidence=0.95
            ),
            TextBlock(
                text="Section 1: Revenue Analysis\nRevenue grew by 15% YoY mostly driven by...",
                bounding_box=(50.0, 110.0, 300.0, 400.0), # Left column example
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
                bounding_box=(50.0, 450.0, 550.0, 600.0),
                page_number=1,
                caption="Financial Results 2024"
            )
        ]
        
        mock_figures = [
            Figure(
                alt_text="Bar chart showing revenue trends",
                bounding_box=(350.0, 110.0, 550.0, 300.0),
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
                "source_tool": "Docling/MinerU"
            }
        )
