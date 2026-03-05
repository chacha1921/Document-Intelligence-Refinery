import logging
import pdfplumber
import yaml
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Tuple
from collections import Counter

from src.models import (
    DocumentProfile, 
    OriginType, 
    LayoutComplexity, 
    ExtractionCost
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Pluggable Domain Strategy Interface ---

class DomainClassifierStrategy(ABC):
    """
    Abstract Base Class for domain classification.
    Allows swapping between simple keyword matching, VLM-based, or ML-based classifiers.
    """
    @abstractmethod
    def classify(self, text_sample: str) -> Tuple[Optional[str], float]:
        """
        Classifies the domain of the text.
        Returns: (domain_name, confidence_score)
        """
        pass

class KeywordDomainClassifier(DomainClassifierStrategy):
    """
    Classifies domain based on keyword frequency configuration.
    """
    def __init__(self, keywords_config: Dict[str, List[str]]):
        self.keywords_config = keywords_config

    def classify(self, text_sample: str) -> Tuple[Optional[str], float]:
        if not text_sample or not self.keywords_config:
            return None, 0.0

        text_lower = text_sample.lower()
        domain_scores = Counter()
        
        total_hits = 0
        for domain, keywords in self.keywords_config.items():
            for kw in keywords:
                if kw.lower() in text_lower:
                    domain_scores[domain] += 1
                    total_hits += 1
        
        if not domain_scores:
            return None, 0.0
            
        best_domain, count = domain_scores.most_common(1)[0]
        # Simple confidence: hits for this domain / total hits
        confidence = count / total_hits if total_hits > 0 else 0.0
        return best_domain, confidence

# --- Main Triage Agent ---

class TriageAgent:
    """
    Analyzes a PDF document to determine its properties and optimal extraction strategy.
    
    Attributes:
        config (Dict): Configuration dictionary loaded from extraction_rules.yaml.
        domain_classifier (DomainClassifierStrategy): Strategy for domain hints.
    """
    
    def __init__(self, config_path: str = "rubric/extraction_rules.yaml"):
        """
        Initialize the TriageAgent.
        """
        self.config = self._load_config(config_path)
        logger.info(f"TriageAgent initialized with config from {config_path}")
        
        # Initialize the configured domain strategy
        # In a more complex app, this could be dependency-injected
        self.domain_classifier = KeywordDomainClassifier(self.config.get("domain_keywords", {}))

    def _load_config(self, config_path: str) -> Dict:
        """Loads the YAML configuration file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            return {
                "thresholds": {"character_density_min": 50, "image_area_ratio_max": 0.4},
                "domain_keywords": {}
            }

    def analyze(self, file_path: str) -> DocumentProfile:
        """
        Analyzes the PDF document and returns a DocumentProfile.
        """
        logger.info(f"Starting analysis for document: {file_path}")
        
        try:
            with pdfplumber.open(file_path) as pdf:
                # 1. Determine Origin Type
                origin_type, pages_stats = self._determine_origin_type(pdf)
                logger.info(f"Determined Origin Type: {origin_type}")

                # 2. Determine Layout Complexity
                layout_complexity = self._determine_layout_complexity(pdf, pages_stats)
                logger.info(f"Determined Layout Complexity: {layout_complexity}")

                # 3. Determine Domain Hint (Using Pluggable Strategy)
                # usage: extract sample text first
                sample_text = self._extract_sample_text(pdf)
                domain_hint, domain_conf = self.domain_classifier.classify(sample_text)
                logger.info(f"Determined Domain Hint: {domain_hint} (Conf: {domain_conf:.2f})")
                
                # 4. Estimate Extraction Cost and Triage Confidence
                extraction_cost = self._estimate_extraction_cost(origin_type, layout_complexity)
                
                # Calculate overall classification confidence
                # Heuristic: Lower confidence if "Mixed" origin or weak domain signal
                triage_confidence = 1.0
                if origin_type == OriginType.MIXED:
                    triage_confidence *= 0.8
                if not domain_hint:
                    triage_confidence *= 0.9

                profile = DocumentProfile(
                    origin_type=origin_type,
                    layout_complexity=layout_complexity,
                    language="en", 
                    domain_hint=domain_hint,
                    estimated_extraction_cost=extraction_cost,
                    classification_confidence=triage_confidence
                )
                
                logger.info("Document analysis completed successfully.")
                return profile

        except Exception as e:
            logger.error(f"Error analyzing document {file_path}: {e}")
            raise

    def _extract_sample_text(self, pdf) -> str:
        """Extracts text from the first few pages for analysis."""
        sample_text = ""
        try:
            for page in pdf.pages[:3]:
                text = page.extract_text()
                if text:
                    sample_text += text + " "
        except Exception:
            pass # Be resilient to partial extraction failures
        return sample_text

    def _determine_origin_type(self, pdf) -> Tuple[OriginType, List[Dict]]:
        """
        Analyzes pages for character density to determine if it's digital, scanned, etc.
        """
        total_pages = len(pdf.pages)
        if total_pages == 0:
            # Handle empty PDF edge case
            return OriginType.MIXED, []

        native_text_pages = 0
        image_only_pages = 0
        form_elements_detected = False
        
        pages_stats = []
        
        char_density_min = self.config.get("thresholds", {}).get("character_density_min", 50)
        image_ratio_max = self.config.get("thresholds", {}).get("image_area_ratio_max", 0.4)
        
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            char_count = len(text)
            
            image_area = 0
            for img in page.images:
                w = img.get('x1', 0) - img.get('x0', 0)
                h = img.get('bottom', 0) - img.get('top', 0)
                if w > 0 and h > 0:
                   image_area += w * h
            
            page_area = max(1, page.width * page.height)
            image_ratio = image_area / page_area
            
            stats = {
                "page_num": i + 1,
                "char_count": char_count,
                "image_ratio": image_ratio,
                "has_tables": bool(page.find_tables())
            }
            pages_stats.append(stats)

            if char_count < char_density_min:
                # Very low text count -> Likely Scanned or Image-heavy
                image_only_pages += 1
            else:
                # Text is present
                if image_ratio > image_ratio_max:
                     # High text AND high image? Likely mixed layer or heavy graphics
                     # Could count as mixed, but often handled as digital with images
                     # For safety, if ratio is VERY high, assume scanned or complex
                     pass
                native_text_pages += 1

            if page.annots:
                for annot in page.annots:
                    if annot.get("subtype") == "Widget":
                        form_elements_detected = True

        # Decision Logic
        if form_elements_detected:
            return OriginType.FORM_FILLABLE, pages_stats
            
        if image_only_pages == total_pages:
            return OriginType.SCANNED_IMAGE, pages_stats
        
        # If we have significant native text pages (e.g. > 90%), treat as native
        if native_text_pages / total_pages > 0.9:
            return OriginType.NATIVE_DIGITAL, pages_stats
            
        return OriginType.MIXED, pages_stats

    def _determine_layout_complexity(self, pdf, pages_stats: List[Dict]) -> LayoutComplexity:
        """
        Determines layout complexity based on columns and tables.
        """
        if not pages_stats:
            return LayoutComplexity.SINGLE_COLUMN

        table_heavy_count = sum(1 for p in pages_stats if p["has_tables"])
        total_pages = len(pages_stats)
        
        if total_pages > 0 and (table_heavy_count / total_pages) > 0.3:
            return LayoutComplexity.TABLE_HEAVY

        multi_column_detected = False
        check_pages = pdf.pages[:min(total_pages, 3)] 
        
        for page in check_pages:
            words = page.extract_words()
            if not words:
                continue
                
            page_mid = page.width / 2
            left_side_words = [w for w in words if w['x1'] < page_mid]
            right_side_words = [w for w in words if w['x0'] > page_mid]
            
            if len(left_side_words) > 50 and len(right_side_words) > 50:
                 multi_column_detected = True
                 break
        
        if multi_column_detected:
            return LayoutComplexity.MULTI_COLUMN
            
        return LayoutComplexity.SINGLE_COLUMN

    def _estimate_extraction_cost(self, origin: OriginType, layout: LayoutComplexity) -> ExtractionCost:
        """
        Maps origin and layout to an extraction strategy/cost.
        """
        if origin == OriginType.SCANNED_IMAGE or origin == OriginType.MIXED:
            return ExtractionCost.NEEDS_VISION_MODEL
            
        if layout in [LayoutComplexity.TABLE_HEAVY, LayoutComplexity.MULTI_COLUMN, LayoutComplexity.FIGURE_HEAVY]:
            return ExtractionCost.NEEDS_LAYOUT_MODEL
            
        if origin == OriginType.FORM_FILLABLE:
            return ExtractionCost.NEEDS_LAYOUT_MODEL 
            
        return ExtractionCost.FAST_TEXT_SUFFICIENT

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        agent = TriageAgent()
        try:
            result = agent.analyze(sys.argv[1])
            print(result.model_dump_json(indent=2))
        except Exception as e:
            print(f"Analysis failed: {e}")
    else:
        print("Usage: python src/agents/triage.py <path_to_pdf>")
