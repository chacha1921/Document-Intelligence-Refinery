import pdfplumber
import logging
from typing import Optional, Dict, Any, List, Tuple
from src.strategies.base import BaseExtractor
from src.models import ExtractedDocument, TextBlock, Figure, BBox

logger = logging.getLogger(__name__)

class FastTextExtractor(BaseExtractor):
    """
    Strategy A: Fast, text-focused extraction using pdfplumber.
    Best for native digital PDFs with simple layouts.
    Includes confidence scoring based on text density vs image area.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.conf_threshold = self.config.get("confidence_threshold", 0.85)

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
                    
                    if text_content:
                        # Find overall bbox for text
                        x0 = min((w['x0'] for w in words), default=0.0)
                        top = min((w['top'] for w in words), default=0.0)
                        x1 = max((w['x1'] for w in words), default=page.width) # Fallback to page width
                        bottom = max((w['bottom'] for w in words), default=page.height) # Fallback to page height
                        
                        # Validate bbox coordinates before creating BBox
                        if x0 > x1: x0, x1 = x1, x0
                        if top > bottom: top, bottom = bottom, top

                        extracted_text_blocks.append(TextBlock(
                            text=text_content,
                            bounding_box=BBox(x0=float(x0), y0=float(top), x1=float(x1), y1=float(bottom)),
                            page_number=page_num,
                            confidence=1.0 # Will adjust later based on page scoring
                        ))

                    # 2. Image Extraction
                    for img in page.images:
                         x0, top, x1, bottom = img['x0'], img['top'], img['x1'], img['bottom']
                         if x0 > x1: x0, x1 = x1, x0
                         if top > bottom: top, bottom = bottom, top
                         
                         extracted_figures.append(Figure(
                             bounding_box=BBox(x0=float(x0), y0=float(top), x1=float(x1), y1=float(bottom)),
                             page_number=page_num,
                             image_ref=f"page_{page_num}_img_{len(extracted_figures)}"
                         ))

                    # 3. Confidence Scoring
                    confidence = self._calculate_confidence(page, text_content)
                    
                    # Update confidence for blocks on this page
                    for block in extracted_text_blocks:
                        if block.page_number == page_num:
                            block.confidence = confidence

                    if confidence < self.conf_threshold:
                        low_confidence_pages.append(page_num)
                        logger.warning(f"Page {page_num} has low confidence ({confidence:.2f} < {self.conf_threshold}).")

            doc = ExtractedDocument(
                text_blocks=extracted_text_blocks,
                figures=extracted_figures,
                metadata={
                    "extractor": "FastTextExtractor",
                    "low_confidence_pages": low_confidence_pages,
                    "strategy_history": ["FastTextExtractor"],
                    "avg_confidence": sum(b.confidence for b in extracted_text_blocks) / len(extracted_text_blocks) if extracted_text_blocks else 0.0
                }
            )
            return doc

        except Exception as e:
            logger.error(f"FastText extraction failed: {e}")
            raise

    def _calculate_confidence(self, page, text_content: str) -> float:
        """
        Calculates a multi-signal confidence score (0.0 to 1.0).
        Signals:
        1. Character Density (vs page size)
        2. Image Area Ratio (high images = low text confidence)
        3. Alphanumeric Ratio (detect garbage/encoding issues)
        4. Tofu/Replacement Character Ratio (detect OCR failure)
        """
        page_area = float(page.width * page.height)
        if page_area == 0: return 0.0

        char_count = len(text_content)
        
        # Signal 1 & 2: Image Ratio & Density (Base Score)
        image_area = 0.0
        for img in page.images:
            w = img['x1'] - img['x0']
            h = img['bottom'] - img['top']
            if w > 0 and h > 0:
                image_area += w * h
        
        image_ratio = image_area / page_area
        score = 1.0
        
        min_chars = self.config.get("min_chars_per_page", 50)
        image_pen_thresh = self.config.get("image_penalty_threshold", 0.5)

        if image_ratio > image_pen_thresh:
            score -= 0.4 
        elif image_ratio > (image_pen_thresh * 0.6):
            score -= 0.2
            
        if char_count < min_chars:
             score -= 0.4
        elif char_count < (min_chars * 2):
             score -= 0.1

        # Signal 3: Alphanumeric Ratio (Garbage text check)
        if char_count > 0:
            alnum_count = sum(c.isalnum() for c in text_content)
            alnum_ratio = alnum_count / char_count
            if alnum_ratio < 0.5: # Mostly symbols/garbage
                score -= 0.3
                
        # Signal 4: Tofu Characters ( or similar)
        if char_count > 0:
            tofu_count = text_content.count('') + text_content.count('\ufffd')
            if tofu_count > 0:
                tofu_ratio = tofu_count / char_count
                if tofu_ratio > 0.1: # >10% of text is bad
                    score -= 0.5
                elif tofu_ratio > 0.01:
                    score -= 0.2
             
        # Clamp score 0 to 1
        return max(0.0, min(1.0, score))
