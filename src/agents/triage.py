"""
Stage 1: Triage Agent for the Document Intelligence Refinery.
Analyzes an incoming PDF and generates a DocumentProfile.
"""
import sys
import os
import json
import logging
import time
import requests
import base64
import yaml
from dotenv import load_dotenv
import fitz # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any, Counter
import pdfplumber
from src.models.schemas import DocumentProfile

class TriageAgent:
    """
    Analyzes a document to determine its origin, layout complexity, domain, and estimated extraction cost.
    """
    def __init__(self, output_dir: str = ".refinery/profiles"):
        self.logger = logging.getLogger(__name__)

        # Use absolute path relative to workspace root if simpler, or relative to cwd
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load environment variables
        load_dotenv()
        
        # Load extraction rules
        try:
            # Try from cwd first
            self.rules_path = Path("rubric/extraction_rules.yaml")
            if not self.rules_path.exists():
                # Try relative to this file's location
                self.rules_path = Path(__file__).resolve().parent.parent.parent / "rubric/extraction_rules.yaml"
                
            if self.rules_path.exists():
                with open(self.rules_path, "r") as f:
                    self.rules = yaml.safe_load(f) or {}
                # print(f"Loaded rules from {self.rules_path}")
            else:
                self.rules = {}
                print(f"Warning: Rules file not found at {self.rules_path}. Using internal defaults.")
        except Exception as e:
            print(f"Error loading rules: {e}")
            self.rules = {}

    def analyze(self, pdf_path: str) -> DocumentProfile:
        """
        Main entry point for analyzing a PDF.
        """
        if not os.path.exists(pdf_path):
             raise FileNotFoundError(f"File not found: {pdf_path}")

        doc_id = Path(pdf_path).stem
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # 1. Origin Type Detection
                origin_type = self._detect_origin_type(pdf)
                
                # 2. Layout Complexity Detection
                layout_complexity = self._detect_layout_complexity(pdf)
                
                # 3. Domain Hint Classifier
                # Extract text from the first few pages for domain classification
                full_text = ""
                # Limit to first 5 pages to save time/memory
                pages_to_scan = pdf.pages[:5] 
                
                for page in pages_to_scan:
                    text = page.extract_text()
                    if text:
                        full_text += text + " "
                

                domain_hint = self._classify_domain(full_text, pdf_path)
                
                # 4. Estimated Cost
                estimated_extraction_cost = self._estimate_cost(origin_type, layout_complexity)
                
                # Language detection
                if origin_type == "scanned_image" or not full_text.strip():
                    # Use Vision Model for language detection on scanned docs
                    language, language_confidence = self._detect_language_vision_ollama(pdf_path)
                else:
                    language, language_confidence = self._detect_language(full_text)

                profile = DocumentProfile(
                    origin_type=origin_type,
                    layout_complexity=layout_complexity,
                    language=language,
                    language_confidence=language_confidence,
                    domain_hint=domain_hint,
                    estimated_extraction_cost=estimated_extraction_cost
                )
                
                self._save_profile(doc_id, profile)
                return profile

        except Exception as e:
            print(f"Error analyzing document: {e}")
            # Return a fallback or re-raise. For this pipeline, re-raising is better.
            raise



    def _get_sample_pages(self, pdf, max_samples: int = 5) -> list:
        """
        Stratified sampling: Start, End, Middle.
        Safely returns a list of page objects.
        """
        all_pages = pdf.pages
        total_pages = len(all_pages)
        
        if total_pages <= max_samples:
            return all_pages
            
        # Strategy: Always take first and last
        # Then distribute remaining (max_samples - 2) across the middle
        indices = {0, total_pages - 1}
        
        if max_samples > 2:
            middle_count = max_samples - 2
            step = total_pages / (middle_count + 1)
            for i in range(1, middle_count + 1):
                indices.add(int(step * i))
                
        sorted_indices = sorted(list(indices))
        return [all_pages[i] for i in sorted_indices if i < total_pages]

    def _detect_origin_type(self, pdf) -> str:
        """
        Distinguishes between native_digital, scanned_image, mixed, form_fillable.
        Uses Stratified Sampling and "Weakest Link" rule.
        """
        # Thresholds from rules
        thresholds = self.rules.get("thresholds", {})
        min_chars = thresholds.get("character_density_min", 100) # Default 100 if missing
        max_image_ratio = thresholds.get("image_area_ratio_max", 0.5) # Default 0.5 if missing

        # Check for AcroForm presence (strong signal for form_fillable)
        if hasattr(pdf, 'doc') and pdf.doc.catalog and 'AcroForm' in pdf.doc.catalog:
            return "form_fillable" # Immediate return if form

        pages_to_check = self._get_sample_pages(pdf, max_samples=10)
        if not pages_to_check:
            return "mixed" # Empty PDF?
            
        scanned_page_count = 0
        native_page_count = 0
        
        for page in pages_to_check:
            total_chars = len(page.chars)
            total_page_area = float(page.width) * float(page.height)
            
            # Calculate image area
            total_image_area = 0.0
            for img in page.images:
                w = float(img.get('width', 0))
                h = float(img.get('height', 0))
                total_image_area += w * h
            
            image_ratio = total_image_area / total_page_area if total_page_area > 0 else 0
            
            # Heuristic per page
            # If ANY page looks scanned (low text, high image), flag it
            if total_chars < min_chars and image_ratio > max_image_ratio:
                scanned_page_count += 1
            elif total_chars > min_chars:
                native_page_count += 1
        
        # Weakest Link Rule: If we found scanned pages mixed with native, it's mixed or scanned
        if scanned_page_count > 0:
            if native_page_count > 0:
                return "mixed"
            return "scanned_image"
            
        return "native_digital"

    def _detect_layout_complexity(self, pdf) -> str:
        """
        Detects if single_column, multi_column, table_heavy, figure_heavy, or mixed.
        Uses Stratified Sampling and "Weakest Link" rule.
        """
        pages_to_check = self._get_sample_pages(pdf, max_samples=5)
        
        complexity_scores = Counter()
        
        for page in pages_to_check:
            # Check for tables
            tables = page.find_tables()
            if tables and len(tables) > 0:
                complexity_scores['table_heavy'] += 1

            # Check for figures (images)
            if len(page.images) > 2:
                complexity_scores['figure_heavy'] += 1
            
            # Check for columns
            if page.chars:
                width = float(page.width)
                mid_x = width / 2
                
                # Canvas the central 10% strip
                strip_width = width * 0.1
                left_bound = mid_x - (strip_width / 2)
                right_bound = mid_x + (strip_width / 2)
                
                # Count chars in left, center, right regions
                left_chars = 0
                right_chars = 0
                center_chars = 0
                
                for c in page.chars:
                    x0 = float(c.get('x0', 0))
                    if x0 < left_bound:
                        left_chars += 1
                    elif x0 > right_bound:
                        right_chars += 1
                    elif x0 >= left_bound and x0 <= right_bound: 
                        center_chars += 1
                
                # Heuristic: Significant text on both sides but empty center gutter
                # Adjust threshold based on char density
                if left_chars > 20 and right_chars > 20 and center_chars < 5:
                    complexity_scores['multi_column'] += 1
                else:
                    complexity_scores['single_column'] += 1

        # Decision Logic (Prioritize complexity - Weakest Link)
        # If any page is table heavy, the doc is table heavy (extraction becomes harder)
        if complexity_scores['table_heavy'] > 0:
             return "table_heavy"
        
        if complexity_scores['multi_column'] > 0:
             return "multi_column"
             
        if complexity_scores['figure_heavy'] > 0:
             return "figure_heavy"
            
        return "single_column"


    def _classify_domain(self, text: str, pdf_path: str = None) -> str:
        """
        Domain classification with robust fallbacks:
        1. Text-based Cloud (Gemini) - if text exists
        2. Text-based Local (Ollama Llama3.2) - if text exists & Gemini fails
        3. Vision-based Local (Ollama Qwen-VL) - if text is missing or scanned
        4. Keyword Heuristics - final resort
        """
        # 0. Immediate Vision Fallback if no text (Scanned Document)
        if not text.strip() and pdf_path:
            self.logger.info("No text extracted (scanned document). Using Vision-based domain classification...")
            return self._classify_domain_vision_ollama(pdf_path) or "general"

        # 1. Try Gemini (Text)
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                
                system_prompt = (
                    "Classify the following document text into exactly one of these categories: "
                    "financial, legal, technical, medical, general. "
                    "Return ONLY the single category word."
                )
                
                # Truncate text to avoid token limits if necessary (e.g., first 8000 chars)
                truncated_text = text[:8000] 
                
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = client.models.generate_content(
                            model=os.getenv("GEMINI_MODEL", "gemini-flash-latest"), 
                            contents=f"{system_prompt}\n\nDocument Text:\n{truncated_text}" 
                        )
                        
                        if response and response.text:
                            # print(f"Gemini response: {response.text.strip()}")
                            category = response.text.strip().lower()
                            allowed = ["financial", "legal", "technical", "medical", "general"]
                            if category in allowed:
                                return category
                            for a in allowed:
                                if a in category:
                                    return a
                        break
                    except Exception as e:
                        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                            wait = (attempt + 1) * 2
                            if attempt < max_retries - 1:
                                print(f"Gemini API rate limited. Retrying in {wait}s...")
                                time.sleep(wait)
                                continue
                        raise e
                            
            except Exception as e:
                # Log error but don't crash - proceed to fallback
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    self.logger.warning("Gemini domain classification skipped (Rate Limit Exceeded). Attempting Ollama fallback...")
                else:
                    self.logger.warning(f"Gemini domain classification failed: {e}. Attempting Ollama fallback...")
                
                # 2. Try Ollama Text (Llama 3.2) - Only if we have decent text
                if len(text.strip()) > 50: 
                    ollama_result = self._classify_domain_ollama(text)
                    self.logger.info(f"Ollama domain classification result: {ollama_result}")
                    if ollama_result:
                        return ollama_result
                
                # 3. Try Ollama Vision (Qwen-VL) - If Llama failed or text was garbage
                if pdf_path:
                    self.logger.info("Attempting Vision-based domain classification...")
                    vision_result = self._classify_domain_vision_ollama(pdf_path)
                    self.logger.info(f"Vision domain classification result: {vision_result}")
                    if vision_result:
                        return vision_result
                
                self.logger.warning("All local fallbacks failed. Falling back to keyword heuristics.")

        # Fallback: Keyword-based domain hinting
        text = text.lower()

        keywords = self.rules.get("domain_keywords")
        if not keywords:
             # Default fallback if rules missing
             keywords = {
                "financial": ["balance sheet", "income statement", "cash flow", "fiscal year", "audit", "revenue", "profit", "loss", "assets", "liabilities"],
                "legal": ["agreement", "contract", "parties", "witnesseth", "hereby", "indemnification", "jurisdiction", "plaintiff", "defendant", "pursuant"],
                "medical": ["patient", "diagnosis", "treatment", "symptoms", "clinical", "hospital", "prescription", "physician", "history"],
                "technical": ["algorithm", "system", "architecture", "api", "database", "server", "client", "interface", "parameter", "function", "method"],
            }
        
        scores = {k: 0 for k in keywords}
        
        for domain, keys in keywords.items():
            for key in keys:
                if key in text:
                    scores[domain] += 1
        
        # Return domain with max score if > 0
        best_domain = max(scores, key=scores.get)
        if scores[best_domain] > 0:
            return best_domain
            
        return "general"


    def _estimate_cost(self, origin: str, layout: str) -> str:
        """
        Rules for estimated extraction cost.
        """
        if origin == "scanned_image":
            return "needs_vision_model"
        
        if layout in ["multi_column", "table_heavy", "figure_heavy", "mixed"]:
            return "needs_layout_model"
            
        if origin == "native_digital" and layout == "single_column":
            return "fast_text_sufficient"
            
        if origin == "form_fillable":
             # Forms often need structure preservation, so layout model is safer than raw text dump
             return "needs_layout_model"

        return "needs_layout_model" # Fallback

    def _detect_language(self, text: str) -> tuple[str, float]:
        """
        Simple stop-word based language detection.
        Returns (language_code, confidence).
        """
        if not text:
            return "en", 0.0

        # Truncate text for performance
        text_sample = text[:5000].lower()
        words = set(text_sample.split())
        
        # Common stopwords for supported languages
        stopwords = {
            "en": {"the", "and", "is", "of", "to", "in", "that", "it"},
            "fr": {"le", "la", "et", "de", "un", "es", "est", "que"}, # 'es' is also 'plural' in FR sometimes or typo common
            "es": {"el", "la", "y", "de", "en", "un", "es", "que"},
            "de": {"der", "die", "und", "in", "den", "von", "zu", "das"}
        }
        
        scores = {}
        for lang, stops in stopwords.items():
            intersection = words.intersection(stops)
            scores[lang] = len(intersection)
            
        # If we found exceedingly few stopwords (e.g. < 5), it might be a partial match 
        # or a language not in our list (like Amharic with some English loanwords).
        # In this case, force Ollama check for better accuracy.
        if not scores or max(scores.values()) < 5:
             # Fallback to Ollama for language detection
             return self._detect_language_ollama(text)
            
        best_lang = max(scores, key=scores.get)
        best_score = scores[best_lang]
        
        # Confidence logic:
        # If we find 5+ stopwords, we are fairly confident (1.0).
        # Be careful not to divide by zero if best_score is low.
        confidence = min(1.0, best_score / 5.0) 
        
        return best_lang, confidence

    def _classify_domain_vision_ollama(self, pdf_path: str) -> str | None:
        """
        Fallback domain classification using local Vision-Language Model (qwen3-vl:4b) via Ollama.
        Analyses the first page image.
        """
        try:
            # 1. Render First Page to Image
            import base64
            doc = fitz.open(pdf_path)
            if doc.page_count < 1:
                return None
            
            page = doc.load_page(0)
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            
            # 2. Call Ollama Vision Model
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            url = f"{base_url}/api/generate"
            model = os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:4b") # Use env var
            
            prompt = (
                "Classify the document shown in this image into exactly one of: "
                "financial, legal, technical, medical, general. "
                "Return ONLY the single category word."
            )
            
            payload = {
                "model": model,
                "prompt": prompt,
                "images": [img_base64],
                "stream": False
            }
            
            # print(f"Sending vision request to {model}...")
            response = requests.post(url, json=payload, timeout=300)
            
            if response.status_code == 200:
                data = response.json()
                content = data.get("response", "").strip().lower()
                # print(f"Vision response raw: {content}")
                allowed = ["financial", "legal", "technical", "medical", "general"]
                
                if content in allowed:
                    return content
                # Fuzzy match
                for a in allowed:
                    if a in content:
                        return a
            return None
        except Exception as e:
            print(f"Vision domain classification failed: {e}")
            return None

    def _classify_domain_ollama(self, text: str) -> str | None:
        """
        Fallback domain classification using local Ollama instance.
        Returns category string or None if failed.
        """
        try:
            model = os.getenv("OLLAMA_TEXT_MODEL", "qwen3-vl:4b")
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            url = f"{base_url}/api/generate"
            
            system_prompt = (
                "Classify the document text into exactly one of: "
                "financial, legal, technical, medical, general. "
                "Return ONLY the single category word."
            )
            # print(f"{system_prompt}\n\nDocument Text:\n{text[:10000]}")
            payload = {
                "model": model,
                "prompt": f"{system_prompt}\n\nDocument Text:\n{text[:10000]}",
                "stream": False
            }
            
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                content = data.get("response", "").strip().lower()
                allowed = ["financial", "legal", "technical", "medical", "general"]
                
                if content in allowed:
                    return content
                for a in allowed:
                    if a in content:
                        return a
            return None
        except Exception as e:
            # print(f"Ollama error: {e}") # Optional logging
            return None

    def _detect_language_vision_ollama(self, pdf_path: str) -> tuple[str, float]:
        """
        Uses local Vision-Language Model (qwen3-vl:4b) via Ollama 
        to detect language from the first page image.
        """
        try:
            # 1. Render First Page to Image
            doc = fitz.open(pdf_path)
            if doc.page_count < 1:
                return "en", 0.0
            
            page = doc.load_page(0)
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            
            # 2. Call Ollama Vision Model
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            url = f"{base_url}/api/generate"
            model = os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:4b")
            
            prompt = (
                "Identify the primary language of the document shown in this image. "
                "Return ONLY the 2-letter ISO 639-1 language code (e.g. en, fr, de, am)."
            )
            
            payload = {
                "model": model,
                "prompt": prompt,
                "images": [img_base64],
                "stream": False
            }
            
            # print(f"Sending vision request to {model}...")
            response = requests.post(url, json=payload, timeout=300)
            
            if response.status_code == 200:
                data = response.json()
                lang_response = data.get("response", "").strip().lower()
                
                # Basic cleaning to extract code
                import re
                # Look for 2 lowercase letters standing alone
                match = re.search(r'\b[a-z]{2}\b', lang_response)
                
                if match:
                    return match.group(0), 0.9
                
                # Check for common names locally if LLM was verbose
                common_map = {"english": "en", "french": "fr", "spanish": "es", "german": "de", "amharic": "am"}
                for k, v in common_map.items():
                    if k in lang_response:
                        return v, 0.9
                        
            return "en", 0.0
            
        except Exception as e:
            # print(f"Vision language detection failed: {e}")
            return "en", 0.0

    def _detect_language_ollama(self, text: str) -> tuple[str, float]:
        """
        Detects language using local Ollama model.
        Returns (language_code, confidence).
        """
        self.logger.info("Detecting language with Ollama fallback...")
        try:
            model = os.getenv("OLLAMA_TEXT_MODEL", "qwen3-vl:4b")
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            url = f"{base_url}/api/generate"
            
            system_prompt = (
                "Identify the language of the provided text. "
                "Return ONLY the 2-letter ISO code (e.g., 'en', 'fr', 'es', 'de')."
            )
            print(f"Detecting language with Ollama fallback... {system_prompt}\n\nDocument Text:\n{text[:10000]}")
            payload = {
                "model": model,
                "prompt": f"{system_prompt}\n\nText Sample:\n{text[:1000]}",
                "stream": False
            }
            
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                lang_code = data.get("response", "").strip().lower()
                
                # Basic validation ISO 2 chars (roughly)
                if len(lang_code) == 2:
                    return lang_code, 0.8 # Moderate confidence for LLM
                
                # Check for common full names just in case
                common_map = {"english": "en", "french": "fr", "spanish": "es", "german": "de", "amharic": "am"}
                for k, v in common_map.items():
                    if k in lang_code:
                        return v, 0.8
                        
            return "en", 0.0
        except Exception as e:
            # print(f"Ollama lang detection failed: {e}")
            return "en", 0.0

    def _save_profile(self, doc_id: str, profile: DocumentProfile):
        output_path = self.output_dir / f"{doc_id}.json"
        with open(output_path, "w") as f:
            f.write(profile.model_dump_json(indent=2))
        if hasattr(self, 'logger'):
             self.logger.info(f"Profile saved to {output_path}")
        else:
             print(f"[TriageAgent] Profile saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        print(f"Analyzing {pdf_path}...")
        try:
            agent = TriageAgent()
            profile = agent.analyze(pdf_path)
            print("Analysis Complete:")
            print(profile.model_dump_json(indent=2))
        except Exception as e:
            print(f"Failed to analyze: {e}")
            sys.exit(1)
    else:
        print("Usage: python -m src.agents.triage <path_to_pdf>")
