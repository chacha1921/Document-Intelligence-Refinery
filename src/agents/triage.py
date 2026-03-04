import logging
import pdfplumber
import yaml
from pathlib import Path
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

class TriageAgent:
    """
    Analyzes a PDF document to determine its properties and optimal extraction strategy.
    
    Attributes:
        config (Dict): Configuration dictionary loaded from extraction_rules.yaml.
    """
    
    def __init__(self, config_path: str = "rubric/extraction_rules.yaml"):
        """
        Initialize the TriageAgent.
        
        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.config = self._load_config(config_path)
        logger.info(f"TriageAgent initialized with config from {config_path}")

    def _load_config(self, config_path: str) -> Dict:
        """Loads the YAML configuration file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            # Return a default fallback configuration
            return {
                "thresholds": {"character_density_min": 50, "image_area_ratio_max": 0.4},
                "domain_keywords": {}
            }

    def analyze(self, file_path: str) -> DocumentProfile:
        """
        Analyzes the PDF document and returns a DocumentProfile.
        
        Args:
            file_path (str): Path to the PDF file.
            
        Returns:
            DocumentProfile: The profiling result.
        """
        logger.info(f"Starting analysis for document: {file_path}")
        
        try:
            with pdfplumber.open(file_path) as pdf:
                # 1. Determine Origin Type & Basic Layout Stats
                origin_type, pages_stats = self._determine_origin_type(pdf)
                logger.info(f"Determined Origin Type: {origin_type}")

                # 2. Determine Layout Complexity
                layout_complexity = self._determine_layout_complexity(pdf, pages_stats)
                logger.info(f"Determined Layout Complexity: {layout_complexity}")

                # 3. Determine Domain Hint
                domain_hint = self._determine_domain_hint(pdf)
                logger.info(f"Determined Domain Hint: {domain_hint}")
                
                # 4. Determine Language (Simple Heuristic or placeholder)
                language = "en" # Placeholder for now, could use langdetect library if added to deps
                
                # 5. Estimate Extraction Cost
                extraction_cost = self._estimate_extraction_cost(origin_type, layout_complexity)
                logger.info(f"Estimated Extraction Cost: {extraction_cost}")

                profile = DocumentProfile(
                    origin_type=origin_type,
                    layout_complexity=layout_complexity,
                    language=language,
                    domain_hint=domain_hint,
                    estimated_extraction_cost=extraction_cost
                )
                
                logger.info("Document analysis completed successfully.")
                return profile

        except Exception as e:
            logger.error(f"Error analyzing document {file_path}: {e}")
            # Depending on requirements, might want to raise or return a default error profile
            # For now, raising to let caller handle
            raise

    def _determine_origin_type(self, pdf) -> Tuple[OriginType, List[Dict]]:
        """
        Analyzes pages for character density to determine if it's digital, scanned, etc.
        Returns the OriginType and a list of stats per page for further use.
        """
        total_pages = len(pdf.pages)
        native_text_pages = 0
        mixed_pages = 0
        image_only_pages = 0
        form_elements_detected = False
        
        pages_stats = []
        
        char_density_min = self.config.get("thresholds", {}).get("character_density_min", 50)
        
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            char_count = len(text)
            
            # Simple check for form fields (acroforms) - pdfplumber might not expose this easily on page level without 'annots'
            # But we can check if there are many rects and lines relative to text which might imply form
            # For robust form detection, we might need to inspect metadata or specific object types.
            # Here we check generic density.
            
            # Check for images
            # pdfplumber extracts images as well
            images = page.images
            image_area = 0
            # Calculate rough image area (simplified)
            for img in images:
                # bounding box is usually x0, top, x1, bottom
                w = img.get('x1', 0) - img.get('x0', 0)
                h = img.get('bottom', 0) - img.get('top', 0) # bottom - top
                if w > 0 and h > 0:
                   image_area += w * h
            
            page_area = page.width * page.height
            image_ratio = image_area / page_area if page_area > 0 else 0
            
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
                if image_ratio > self.config.get("thresholds", {}).get("image_area_ratio_max", 0.4):
                     mixed_pages += 1
                else:
                    native_text_pages += 1

            # Simple generic check for form elements in annots if available
            if page.annots:
                for annot in page.annots:
                    # 'Widget' type often implies form fields
                    if annot.get("subtype") == "Widget":
                        form_elements_detected = True

        # Decision Logic
        if form_elements_detected:
            return OriginType.FORM_FILLABLE, pages_stats
            
        if image_only_pages == total_pages:
            return OriginType.SCANNED_IMAGE, pages_stats
        
        if native_text_pages == total_pages:
            return OriginType.NATIVE_DIGITAL, pages_stats
            
        return OriginType.MIXED, pages_stats

    def _determine_layout_complexity(self, pdf, pages_stats: List[Dict]) -> LayoutComplexity:
        """
        Determines layout complexity based on columns and tables.
        """
        table_heavy_count = sum(1 for p in pages_stats if p["has_tables"])
        total_pages = len(pages_stats)
        
        # Heuristic: If > 30% of pages have tables -> Table Heavy
        if total_pages > 0 and (table_heavy_count / total_pages) > 0.3:
            return LayoutComplexity.TABLE_HEAVY

        # Check for multi-column layout on the first few pages
        multi_column_detected = False
        check_pages = pdf.pages[:min(total_pages, 3)] 
        
        for page in check_pages:
            # Simple heuristic: Check word x-positions
            # If we see a gap in the middle with text on both sides consistently...
            # pdfplumber approach: extract_words, look at distribution of x0
            words = page.extract_words()
            if not words:
                continue
                
            page_mid = page.width / 2
            left_side_words = [w for w in words if w['x1'] < page_mid]
            right_side_words = [w for w in words if w['x0'] > page_mid]
            
            # If significant text on both sides
            if len(left_side_words) > 50 and len(right_side_words) > 50:
                 multi_column_detected = True
                 break
        
        if multi_column_detected:
            return LayoutComplexity.MULTI_COLUMN
            
        return LayoutComplexity.SINGLE_COLUMN

    def _determine_domain_hint(self, pdf) -> Optional[str]:
        """
        Scans extraction text against domain keywords.
        """
        # Collect text from first few pages to save time
        sample_text = ""
        for page in pdf.pages[:3]:
            text = page.extract_text()
            if text:
                sample_text += text.lower() + " "
        
        if not sample_text:
            return None
            
        domain_scores = Counter()
        keywords_config = self.config.get("domain_keywords", {})
        
        for domain, keywords in keywords_config.items():
            for kw in keywords:
                if kw.lower() in sample_text:
                    domain_scores[domain] += 1
        
        if not domain_scores:
            return None
            
        # Return the domain with the highest score
        best_domain, count = domain_scores.most_common(1)[0]
        logger.info(f"Domain hints found: {dict(domain_scores)}")
        return best_domain

    def _estimate_extraction_cost(self, origin: OriginType, layout: LayoutComplexity) -> ExtractionCost:
        """
        Maps origin and layout to an extraction strategy/cost.
        """
        if origin == OriginType.SCANNED_IMAGE or origin == OriginType.MIXED:
            return ExtractionCost.NEEDS_VISION_MODEL
            
        if layout in [LayoutComplexity.TABLE_HEAVY, LayoutComplexity.MULTI_COLUMN, LayoutComplexity.FIGURE_HEAVY]:
            return ExtractionCost.NEEDS_LAYOUT_MODEL
            
        if origin == OriginType.FORM_FILLABLE:
            return ExtractionCost.NEEDS_LAYOUT_MODEL # Forms usually need structure understanding
            
        return ExtractionCost.FAST_TEXT_SUFFICIENT

if __name__ == "__main__":
    # Simple test execution if run directly
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
