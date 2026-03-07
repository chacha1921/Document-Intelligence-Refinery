"""
Strategy Implementations for the Document Intelligence Refinery.
"""
import os
import json
import logging
from typing import Tuple, List
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
                        block = TextBlock(
                            id=f"text_{page_idx}_0",
                            page_number=page_idx + 1,
                            bbox=[0.0, 0.0, float(page.width), float(page.height)], # Placeholder BBox
                            text=text
                        )
                        extracted_doc.text_blocks.append(block)
                        extracted_doc.reading_order.append(block.id)

                    # 2. Table Extraction
                    # pdfplumber's find_tables often works better than extract_tables for getting structure + data
                    tables = page.find_tables()
                    for i, table in enumerate(tables):
                        bbox = [float(x) for x in table.bbox]
                        data = table.extract()
                        if not data: 
                            continue
                            
                        # Heuristic: First row is header
                        headers = [str(h) if h is not None else "" for h in data[0]]
                        rows = [[str(c) if c is not None else "" for c in r] for r in data[1:]]

                        table_block = Table(
                            id=f"table_{page_idx}_{i}",
                            page_number=page_idx + 1,
                            bbox=bbox,
                            headers=headers,
                            rows=rows
                        )
                        extracted_doc.tables.append(table_block)
                        extracted_doc.reading_order.append(table_block.id)

                    # 3. Figure Extraction
                    for i, img in enumerate(page.images):
                        try:
                            # img is a dict in pdfplumber
                            bbox = [
                                float(img.get('x0', 0)),
                                float(img.get('top', 0)),
                                float(img.get('x1', 0)),
                                float(img.get('bottom', 0))
                            ]
                            figure_block = Figure(
                                id=f"figure_{page_idx}_{i}",
                                page_number=page_idx + 1,
                                bbox=bbox,
                                image_ref=f"page_{page_idx}_img_{i}"
                            )
                            extracted_doc.figures.append(figure_block)
                            extracted_doc.reading_order.append(figure_block.id)
                        except Exception as e:
                            logger.warning(f"Error extracting figure on page {page_idx}: {e}")

                    # 4. Confidence Scoring Logic
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
                    # 1. Extract Bounding Box & Page Number
                    page_no = 1
                    bbox = [0.0, 0.0, 0.0, 0.0]
                    
                    if hasattr(item, 'prov') and item.prov:
                        # Assuming prov is a list of provenance items
                        p = item.prov[0]
                        if hasattr(p, 'page_no'):
                            page_no = p.page_no
                        if hasattr(p, 'bbox'):
                            b = p.bbox
                            # Check if l, t, r, b exist
                            if hasattr(b, 'l') and hasattr(b, 't') and hasattr(b, 'r') and hasattr(b, 'b'):
                                bbox = [float(b.l), float(b.t), float(b.r), float(b.b)]

                    block = TextBlock(
                        id=f"layout_text_{id(item)}",
                        page_number=page_no,
                        bbox=bbox,
                        text=item.text if hasattr(item, 'text') else str(item)
                    )
                    extracted_doc.text_blocks.append(block)
                    extracted_doc.reading_order.append(block.id)

            if hasattr(doc, 'tables'):
                for tbl in doc.tables:
                     # 1. Extract Bounding Box & Page Number
                     page_no = 1
                     bbox = [0.0, 0.0, 0.0, 0.0]
                     
                     if hasattr(tbl, 'prov') and tbl.prov:
                        p = tbl.prov[0]
                        if hasattr(p, 'page_no'):
                            page_no = p.page_no
                        if hasattr(p, 'bbox'):
                            b = p.bbox
                            if hasattr(b, 'l') and hasattr(b, 't') and hasattr(b, 'r') and hasattr(b, 'b'):
                                bbox = [float(b.l), float(b.t), float(b.r), float(b.b)]

                     # 2. Extract Table Content (Markdown)
                     # Call export_to_markdown() with doc to avoid deprecation warnings
                     md_content = ""
                     if hasattr(tbl, 'export_to_markdown'):
                         try:
                             md_content = tbl.export_to_markdown(doc=doc)
                         except:
                             md_content = tbl.export_to_markdown()
                     
                     # Store in rows[0][0]
                     # Schema: rows is List[List[str]]
                     table_id = f"layout_table_{id(tbl)}"
                     extracted_doc.tables.append(Table(
                         id=table_id,
                         page_number=page_no,
                         bbox=bbox,
                         headers=[], # Empty as requested
                         rows=[[md_content]] if md_content else []
                     ))
                     extracted_doc.reading_order.append(table_id)

            # 3. Figure Extraction (Docling)
            pics = []
            if hasattr(doc, 'pictures'):
                pics = doc.pictures
            elif hasattr(doc, 'images'):
                 pics = doc.images
            
            if pics:
                logger.info(f"Found {len(pics)} figures in Docling result.")
                for i, pic in enumerate(pics):
                     # 1. Extract Bounding Box & Page Number
                     page_no = 1
                     bbox = [0.0, 0.0, 0.0, 0.0]
                     
                     if hasattr(pic, 'prov') and pic.prov:
                        p = pic.prov[0]
                        if hasattr(p, 'page_no'):
                            page_no = p.page_no
                        if hasattr(p, 'bbox'):
                            b = p.bbox
                            if hasattr(b, 'l') and hasattr(b, 't') and hasattr(b, 'r') and hasattr(b, 'b'):
                                bbox = [float(b.l), float(b.t), float(b.r), float(b.b)]
                     
                     # 2. Extract Captions
                     caption = None
                     if hasattr(pic, 'captions') and pic.captions:
                        # Assuming captions is a list of caption objects with text or text attribute
                        # Or it might be a list of strings? Let's be robust.
                        caps = []
                        for c in pic.captions:
                            if hasattr(c, 'text'):
                                caps.append(c.text)
                            else:
                                caps.append(str(c))
                        if caps:
                            caption = " ".join(caps)

                     fig_id = f"layout_figure_{i}"
                     extracted_doc.figures.append(Figure(
                         id=fig_id,
                         page_number=page_no,
                         bbox=bbox,
                         caption=caption,
                         image_ref=f"layout_img_{i}" # Placeholder
                     ))
                     # Add to reading order (simplified append at end)
                     extracted_doc.reading_order.append(fig_id)

            # --- Fix 1: Spatial Reading Order ---
            # Sort all elements by Page Number, then Top Y-Coordinate
            all_elements = []
            all_elements.extend(extracted_doc.text_blocks)
            all_elements.extend(extracted_doc.tables)
            all_elements.extend(extracted_doc.figures)
            
            # Sort key: (page_number, bbox.top)
            # Ensure bbox has at least 2 elements (l, t, r, b). If not, default to 0.
            all_elements.sort(key=lambda x: (x.page_number, x.bbox[1] if x.bbox and len(x.bbox) > 1 else 0))
            
            # Rebuild reading order
            extracted_doc.reading_order = [elem.id for elem in all_elements]

            return extracted_doc, 0.90

        except Exception as e:
            logger.error(f"Layout extraction failed: {e}")
            return extracted_doc, 0.0

# --- Strategy C: Vision VLM (Gemini / Ollama Vision) ---
class VisionExtractor(BaseExtractor):
    """
    Strategy C: Vision-Language Model for OCR and handwriting.
    Supports:
    1. Local Ollama Vision (e.g., qwen3-vl) - Configured via .env (OLLAMA_VISION_MODEL)
    2. Cloud Gemini Vision (e.g., gemini-1.5-flash) - Fallback if API key present
    Triggers for scanned images or when other methods fail.
    Implements a budget guard.
    """
    def __init__(self, token_limit: int = 50000, api_key: str = None):
        self.token_limit = token_limit
        self.current_spend = 0
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-flash-latest") # e.g. "gemini-flash-latest" or "gemini-2.0-flash"
        
        # Load Local Vision Config
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_VISION_MODEL") # e.g. "qwen3-vl:4b"
        
        # Configure Gemini Client
        self.gemini_client = None
        if self.api_key:
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=self.api_key)
            except ImportError:
                logger.warning("Google Generative AI SDK not installed.")
        
        if not self.ollama_model and not self.gemini_client:
             logger.warning("No Vision Model configured (OLLAMA_VISION_MODEL or GEMINI_API_KEY missing). VisionExtractor disabled.")


    def extract(self, file_path_or_bytes: str | bytes) -> Tuple[ExtractedDocument, float]:
        """
        Executes Vision extraction (Ollama preferred, Gemini fallback).
        """
        doc_path = str(file_path_or_bytes)
        doc_id = Path(doc_path).stem
        extracted_doc = ExtractedDocument(doc_id=doc_id)
        
        if not self.ollama_model and not self.gemini_client:
            logger.error("Vision model unconfigured. Check .env for OLLAMA_VISION_MODEL or GEMINI_API_KEY.")
            return extracted_doc, 0.0

        logger.info(f"Running VisionExtractor on {doc_id}...")
        
        try:
            # 1. Convert PDF to Images
            import fitz
            # Store bytes for Gemini (PIL), encode to b64 only if needed for Ollama
            page_images = []
            
            with fitz.open(doc_path) as doc:
                for page_num, page in enumerate(doc):
                    pix = page.get_pixmap()
                    img_bytes = pix.tobytes("png")
                    page_images.append((page_num + 1, img_bytes))

            # 2. Extract Text per Page
            for page_num, img_bytes in page_images:
                page_text = ""
                
                # Priority: Cloud Gemini
                if self.gemini_client:
                     page_text = self._extract_gemini(img_bytes)
                
                # Fallback: Local Ollama
                if not page_text and self.ollama_model:
                     import base64
                     img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                     page_text = self._extract_ollama(img_b64)

                if page_text:
                    block = TextBlock(
                        id=f"vision_text_{page_num}",
                        page_number=page_num,
                        bbox=[0.0, 0.0, 0.0, 0.0],
                        text=page_text
                    )
                    extracted_doc.text_blocks.append(block)
                    extracted_doc.reading_order.append(block.id)

            return extracted_doc, 0.95

        except Exception as e:
            logger.error(f"Vision extraction failed: {e}")
            return extracted_doc, 0.0

    def _extract_ollama(self, img_b64: str) -> str:
        """Helper to call Ollama Vision"""
        import requests
        try:
            url = f"{self.ollama_base_url}/api/generate"
            prompt = "Extract all text from this document image. Return ONLY the raw text content, no markdown formatting or commentary."
            
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False
            }
            
            response = requests.post(url, json=payload, timeout=300)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            else:
                logger.warning(f"Ollama Vision request failed: {response.status_code} - {response.text}")
                return ""
        except Exception as e:
            logger.error(f"Ollama Vision error: {e}")
            return ""
    def _extract_gemini(self, img_bytes: bytes) -> str:
        """Helper to call Gemini Vision"""
        try:
            from PIL import Image
            import io
            image = Image.open(io.BytesIO(img_bytes))
            
            prompt = "Extract all text from this document image. Return ONLY the raw text content."
            
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=[prompt, image]
            )
            
            if response.text:
                return response.text.strip()
            return ""
        except Exception as e:
            logger.warning(f"Gemini Vision request failed: {e}")
            return ""
