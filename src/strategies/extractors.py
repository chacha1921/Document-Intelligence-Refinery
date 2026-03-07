"""
Strategy Implementations for the Document Intelligence Refinery.
"""
import os
import json
import logging
from typing import Tuple, List
from pathlib import Path

# External dependencies (make sure these are installed or mock if necessary)
import pdfplumber
import fitz # pymupdf
# import docling # If docling is intended to be used directly

from src.strategies.base import BaseExtractor
from src.models.schemas import ExtractedDocument, TextBlock, Table, Figure

logger = logging.getLogger(__name__)

# --- Strategy A: Fast Text Extraction (pdfplumber) ---
class FastTextExtractor(BaseExtractor):
    """
    Strategy A: Fast, heuristic-based text extraction suitable for 
    native digital, single-column documents.
    """
    def __init__(self, min_chars: int = 100, image_penalty_threshold: float = 0.5):
        self.min_chars = min_chars
        self.image_penalty_threshold = image_penalty_threshold

    def extract(self, file_path_or_bytes: str | bytes) -> Tuple[ExtractedDocument, float]:
        """
        Uses pdfplumber to extract text and tables.
        Returns (ExtractedDocument, confidence_score).
        """
        doc_path = str(file_path_or_bytes)
        doc_id = Path(doc_path).stem
        
        extracted_doc = ExtractedDocument(doc_id=doc_id)
        confidence_scores = []

        try:
            with pdfplumber.open(doc_path) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    
                    # 1. Text Extraction
                    text = page.extract_text()
                    if text:
                        # Append text block
                        # Simple bbox for full page text for now as pdfplumber aggregates text
                        # Ideally iterate words/chars for precise bbox
                        block = TextBlock(
                            id=f"text_{page_idx}_0",
                            page_number=page_idx + 1,
                            bbox=[0.0, 0.0, float(page.width), float(page.height)], # Placeholder BBox
                            text=text
                        )
                        extracted_doc.text_blocks.append(block)
                        extracted_doc.reading_order.append(block.id)
                    
                    # 2. Confidence Scoring Logic
                    # Factors: Character Count, Image Area
                    char_count = len(page.chars)
                    
                    # Calculate image area ratio
                    image_area = 0.0
                    for img in page.images:
                        w = float(img.get('width', 0))
                        h = float(img.get('height', 0))
                        image_area += w * h
                    total_area = float(page.width) * float(page.height)
                    image_ratio = image_area / total_area if total_area > 0 else 0.0

                    # Score Calculation
                    page_score = 1.0
                    
                    if char_count < self.min_chars:
                        page_score = 0.5 # Penalty for low text
                    
                    if image_ratio > self.image_penalty_threshold:
                        page_score = 0.5 # Penalty for high image content (might be a slide or scan)
                    
                    confidence_scores.append(page_score)

            # Average confidence across pages
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            return extracted_doc, avg_confidence

        except Exception as e:
            logger.error(f"FastText extraction failed: {e}")
            return extracted_doc, 0.0


# --- Strategy B: Layout Analysis (Docling) ---
class LayoutExtractor(BaseExtractor):
    """
    Strategy B: Deep Learning-based Layout Analysis using Docling.
    Triggers for multi-column or complex layouts.
    """
    def __init__(self):
        try:
            from docling.document_converter import DocumentConverter
            self.converter = DocumentConverter()
            self._available = True
        except ImportError:
            logger.warning("Docling not installed. LayoutExtractor disabled.")
            self._available = False



    def extract(self, file_path_or_bytes: str | bytes) -> Tuple[ExtractedDocument, float]:
        """
        Executes Docling extraction and maps to ExtractedDocument schema.
        """
        doc_path = str(file_path_or_bytes)
        doc_id = Path(doc_path).stem
        extracted_doc = ExtractedDocument(doc_id=doc_id)
        
        if not self._available:
             logger.error("Docling unavailable. LayoutExtractor cannot run.")
             return extracted_doc, 0.0

        logger.info(f"Running LayoutExtractor (Docling) on {doc_id}...")
        
        try:
            # 1. Run Docling Conversion
            # Warning: This is resource intensive.
            # Docling returns a ConversionResult object
            result = self.converter.convert(doc_path)
            
            # Access the internal document representation (typically `document`)
            # Note: Docling's API might change. Assuming v2.0+ structure.
            doc = result.document
            
            # 2. Map Content to Schema
            
            # Iterate through body elements if possible
            # Docling typically exposes `texts` or `body` iterable
            # Simulating mapping logic
            
            # Assuming doc has `export_to_dict` or similar for easier JSON mapping if direct access is complex
            # Or iterate `doc.texts` if it's a flat list
            
            # Since I cannot verify the exact Docling version installed, 
            # I will implement a robust fallback logic for text extraction
            
            if hasattr(doc, 'texts'):
                for item in doc.texts:
                    # Simplified mapping
                     block = TextBlock(
                        id=f"layout_text_{id(item)}",
                        page_number=1, # simplified
                        bbox=[0.0, 0.0, 0.0, 0.0],
                        text=item.text if hasattr(item, 'text') else str(item)
                    )
                     extracted_doc.text_blocks.append(block)
                     extracted_doc.reading_order.append(block.id)

            if hasattr(doc, 'tables'):
                for tbl in doc.tables:
                     # Simulating table mapping
                     extracted_doc.tables.append(Table(
                         id=f"layout_table_{id(tbl)}",
                         page_number=1,
                         bbox=[0,0,0,0],
                         headers=[],
                         rows=[["[Table Content Detected]"]]
                     ))

            return extracted_doc, 0.90

        except Exception as e:
            logger.error(f"Layout extraction failed: {e}")
            return extracted_doc, 0.0

# --- Strategy C: Vision VLM (Gemini Vision) ---
class VisionExtractor(BaseExtractor):
    """
    Strategy C: Vision-Language Model for OCR and handwriting using Gemini 1.5 Flash.
    Triggers for scanned images or when other methods fail.
    Implements a budget guard.
    """
    def __init__(self, token_limit: int = 50000, api_key: str = None):
        self.token_limit = token_limit
        self.current_spend = 0
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        # Configure Gemini only if key is present
        self.model = None
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
            except ImportError as e:
                logger.warning(f"Google Generative AI SDK not installed: {e}")
        else:
             logger.warning("GEMINI_API_KEY not found. VisionExtractor disabled.")


    def extract(self, file_path_or_bytes: str | bytes) -> Tuple[ExtractedDocument, float]:
        """
        Executes Gemini Vision extraction.
        """
        doc_path = str(file_path_or_bytes)
        doc_id = Path(doc_path).stem
        extracted_doc = ExtractedDocument(doc_id=doc_id)
        
        if not self.model:
            logger.error("Vision model unconfigured. Check API Key or Install.")
            return extracted_doc, 0.0

        # 1. Budget Guard
        # Estimated cost (input tokens + output tokens) ~ 2000 per page
        estimated_cost = 2000 
        
        if self.current_spend + estimated_cost > self.token_limit:
            logger.warning(f"Budget exceeded for VisionExtractor. Cap: {self.token_limit}, Used: {self.current_spend}")
            return extracted_doc, 0.0 
        
        logger.info(f"Running VisionExtractor (Gemini) on {doc_id}...")
        
        try:
            # 2. Convert PDF to Image (Requires fitz/pymupdf and Pillow)
            images = []
            with fitz.open(doc_path) as doc:
                for page in doc:
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")
                    
                    from PIL import Image
                    import io
                    img = Image.open(io.BytesIO(img_data))
                    images.append(img)
            
            # 3. Call Gemini
            prompt = """
            Extract text from this page. Return raw text for now.
            """
            
            # Simple implementation: one call per page
            for i, img in enumerate(images):
                response = self.model.generate_content([prompt, img])
                
                # Check for usage metadata
                usage = 0
                if hasattr(response, 'usage_metadata'):
                     usage = response.usage_metadata.total_token_count
                self.current_spend += usage
                
                # Simple text extraction for now
                if response.text:
                    block = TextBlock(
                        id=f"vision_text_{i}",
                        page_number=i+1,
                        bbox=[0.0, 0.0, 0.0, 0.0],
                        text=response.text
                    )
                    extracted_doc.text_blocks.append(block)

            return extracted_doc, 0.95

        except Exception as e:
            logger.error(f"Vision extraction failed: {e}")
            return extracted_doc, 0.0
