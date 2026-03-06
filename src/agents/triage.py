"""
Stage 1: Triage Agent for the Document Intelligence Refinery.
Analyzes an incoming PDF and generates a DocumentProfile.
"""
import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Counter
import pdfplumber
from src.models.schemas import DocumentProfile

class TriageAgent:
    """
    Analyzes a document to determine its origin, layout complexity, domain, and estimated extraction cost.
    """
    def __init__(self, output_dir: str = ".refinery/profiles"):
        # Use absolute path relative to workspace root if simpler, or relative to cwd
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
                

                domain_hint = self._classify_domain(full_text)
                
                # 4. Estimated Cost
                estimated_extraction_cost = self._estimate_cost(origin_type, layout_complexity)
                
                # Language detection
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


    def _detect_origin_type(self, pdf) -> str:
        """
        Distinguishes between native_digital, scanned_image, mixed, form_fillable.
        """
        total_chars = 0
        total_image_area = 0
        total_page_area = 0
        has_acroform = False
        
        # Check for AcroForm presence (strong signal for form_fillable)
        if hasattr(pdf, 'doc') and pdf.doc.catalog and 'AcroForm' in pdf.doc.catalog:
            has_acroform = True

        # Check first 3 pages
        pages_to_check = pdf.pages[:3]
        if not pages_to_check:
            return "mixed" # Empty PDF?
            
        for page in pages_to_check:
            chars = page.chars
            total_chars += len(chars)
            
            # Calculate image area
            for img in page.images:
                w = float(img.get('width', 0))
                h = float(img.get('height', 0))
                total_image_area += w * h
                
            total_page_area += float(page.width) * float(page.height)

        # Heuristics
        # 1. Scanned: Very few characters, significant image area
        image_ratio = total_image_area / total_page_area if total_page_area > 0 else 0
        
        if total_chars < 500 and image_ratio > 0.4:
            return "scanned_image"

        # 2. Form Fillable: Explicit AcroForm or explicit structure
        if has_acroform:
            return "form_fillable"
            
        # 3. Native Digital
        if total_chars > 500:
            return "native_digital"
            
        return "mixed"

    def _detect_layout_complexity(self, pdf) -> str:
        """
        Detects if single_column, multi_column, table_heavy, figure_heavy, or mixed.
        """
        pages_to_check = pdf.pages[:3]
        total_tables = 0
        total_figures = 0
        is_multi_column = False
        
        for page in pages_to_check:
            # Check for tables
            tables = page.find_tables()
            if tables:
                total_tables += len(tables)

            # Check for figures (images)
            total_figures += len(page.images)
            
            # Check for columns
            if page.chars and not is_multi_column:
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
                    else:
                        center_chars += 1
                
                # If significant text on both sides but empty center
                if left_chars > 50 and right_chars > 50 and center_chars < 10:
                    is_multi_column = True

        # Decision Logic (Prioritize complexity)
        # If multiple complex features are present, return 'mixed'
        complexity_factors = 0
        if total_tables > 1: complexity_factors += 1
        if is_multi_column: complexity_factors += 1
        if total_figures > 2: complexity_factors += 1
        
        if complexity_factors > 1:
            return "mixed"

        if total_tables > 1:
             return "table_heavy"
        
        if is_multi_column:
             return "multi_column"
             
        if total_figures > 2:
             return "figure_heavy"
            
        return "single_column"


    def _classify_domain(self, text: str) -> str:
        """
        Keyword-based domain hinting.
        """
        text = text.lower()
        
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
            
        if not scores or max(scores.values()) == 0:
            return "en", 0.0 # Default fallback
            
        best_lang = max(scores, key=scores.get)
        best_score = scores[best_lang]
        
        # Confidence logic:
        # If we find 5+ stopwords, we are fairly confident (1.0).
        # Be careful not to divide by zero if best_score is low.
        confidence = min(1.0, best_score / 5.0) 
        
        return best_lang, confidence

    def _save_profile(self, doc_id: str, profile: DocumentProfile):
        output_path = self.output_dir / f"{doc_id}.json"
        with open(output_path, "w") as f:
            f.write(profile.model_dump_json(indent=2))
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
