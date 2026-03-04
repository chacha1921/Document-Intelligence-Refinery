import pdfplumber
import logging
from typing import List, Tuple
from src.strategies.base import BaseExtractor
from src.models import ExtractedDocument, TextBlock, Figure

logger = logging.getLogger(__name__)

class FastTextExtractor(BaseExtractor):
    """
    Strategy A: Fast, text-focused extraction using pdfplumber.
    Best for native digital PDFs with simple layouts.
    Includes confidence scoring based on text density vs image area.
    """

    def extract(self, file_path: str) -> ExtractedDocument:
        logger.info(f"FastTextExtractor processing: {file_path}")
        
        extracted_text_blocks = []
        extracted_figures = []
        low_confidence_pages = []

        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    
                    # 1. Text Extraction
                    text_content = page.extract_text() or ""
                    words = page.extract_words()
                    
                    # Estimate bounding box union of all words for the block (simplified)
                    # In a real scenario, we might cluster words into paragraphs. 
                    # Here we take the whole page text as one block or split by newline logic if desired.
                    # Let's treat the whole page text as one block for simplicity in this strategy.
                    if text_content:
                        # Find overall bbox for text
                        x0 = min((w['x0'] for w in words), default=0)
                        top = min((w['top'] for w in words), default=0) 
                        x1 = max((w['x1'] for w in words), default=page.width)
                        bottom = max((w['bottom'] for w in words), default=page.height)
                        
                        extracted_text_blocks.append(TextBlock(
                            text=text_content,
                            bounding_box=(float(x0), float(top), float(x1), float(bottom)),
                            page_number=page_num,
                            confidence=1.0 # Will adjust later based on page scoring
                        ))

                    # 2. Image Extraction (Metadata only for figures list)
                    # We don't extract binary data here, just references
                    for img in page.images:
                         x0, top, x1, bottom = img['x0'], img['top'], img['x1'], img['bottom']
                         extracted_figures.append(Figure(
                             bounding_box=(float(x0), float(top), float(x1), float(bottom)),
                             page_number=page_num,
                             # In a real app, we'd extract and save the image here
                             image_ref=f"page_{page_num}_img_{len(extracted_figures)}"
                         ))

                    # 3. Confidence Scoring
                    confidence = self._calculate_confidence(page, text_content)
                    
                    # Update confidence for blocks on this page
                    # In a real system, we'd want per-block confidence, but page-level is fine here.
                    for block in extracted_text_blocks:
                        if block.page_number == page_num:
                            block.confidence = confidence

                    if confidence < 0.5:
                        low_confidence_pages.append(page_num)
                        logger.warning(f"Page {page_num} has low confidence ({confidence:.2f}).")

            doc = ExtractedDocument(
                text_blocks=extracted_text_blocks,
                figures=extracted_figures,
                metadata={
                    "extractor": "FastTextExtractor",
                    "low_confidence_pages": low_confidence_pages
                }
            )
            return doc

        except Exception as e:
            logger.error(f"FastText extraction failed: {e}")
            raise

    def _calculate_confidence(self, page, text_content: str) -> float:
        """
        Calculates a confidence score (0.0 to 1.0) based on character density and image ratio.
        """
        # Metrics
        page_area = float(page.width * page.height)
        if page_area == 0: return 0.0

        char_count = len(text_content)
        
        # Calculate image area coverage
        image_area = 0.0
        for img in page.images:
            w = img['x1'] - img['x0']
            h = img['bottom'] - img['top']
            image_area += w * h
        
        image_ratio = image_area / page_area
        
        # Heuristics for Confidence
        # 1. High image ratio (> 50%) -> Likely scanned or slide -> Lower text confidence
        # 2. Very low char count per page (< 50 chars) -> Empty or image-only -> Low confidence
        
        score = 1.0
        
        if image_ratio > 0.5:
            score -= 0.4  # Penalty for high image content
        elif image_ratio > 0.3:
            score -= 0.2
            
        if char_count < 50:
             score -= 0.4 # Penalty for very sparse text
        elif char_count < 100:
             score -= 0.1
             
        # Clamp score 0 to 1
        return max(0.0, min(1.0, score))
