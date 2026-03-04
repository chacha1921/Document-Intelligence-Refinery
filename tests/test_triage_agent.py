import pytest
from unittest.mock import MagicMock, patch
from src.agents.triage import TriageAgent, OriginType, LayoutComplexity, ExtractionCost

@pytest.fixture
def triage_agent():
    # Use config that's already in the repo or a mock one. 
    # For unit tests, we'll patch load_config to return a known config.
    with patch("src.agents.triage.TriageAgent._load_config") as mock_load:
        mock_load.return_value = {
            "thresholds": {"character_density_min": 50, "image_area_ratio_max": 0.4},
            "domain_keywords": {"financial": ["invoice", "total"]}
        }
        return TriageAgent()

def test_analyze_native_digital(triage_agent):
    """Test classification of a native digital document."""
    with patch("pdfplumber.open") as mock_open:
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        
        # Simulate high text content, no images
        mock_page.extract_text.return_value = "This is a native digital document. " * 10
        mock_page.images = []
        mock_page.width = 100
        mock_page.height = 100
        mock_page.find_tables.return_value = []
        mock_page.extract_words.return_value = [{'x0': 10, 'x1': 90}] # Simple layout
        mock_page.annots = []

        mock_pdf.pages = [mock_page]
        mock_open.return_value.__enter__.return_value = mock_pdf

        profile = triage_agent.analyze("dummy.pdf")

        assert profile.origin_type == OriginType.NATIVE_DIGITAL
        assert profile.layout_complexity == LayoutComplexity.SINGLE_COLUMN
        assert profile.estimated_extraction_cost == ExtractionCost.FAST_TEXT_SUFFICIENT

def test_analyze_scanned_image(triage_agent):
    """Test classification of a scanned document."""
    with patch("pdfplumber.open") as mock_open:
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        
        # Simulate essentially no text
        mock_page.extract_text.return_value = ""
        # Simulate full page image
        mock_page.images = [{'x0': 0, 'top': 0, 'x1': 100, 'bottom': 100}]
        mock_page.width = 100
        mock_page.height = 100
        mock_page.find_tables.return_value = []
        mock_page.extract_words.return_value = []
        mock_page.annots = []

        mock_pdf.pages = [mock_page]
        mock_open.return_value.__enter__.return_value = mock_pdf

        profile = triage_agent.analyze("dummy.pdf")

        assert profile.origin_type == OriginType.SCANNED_IMAGE
        assert profile.estimated_extraction_cost == ExtractionCost.NEEDS_VISION_MODEL

def test_analyze_table_heavy(triage_agent):
    """Test classification of a complex layout."""
    with patch("pdfplumber.open") as mock_open:
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        
        mock_page.extract_text.return_value = "Table data " * 20
        mock_page.images = []
        mock_page.width = 100
        mock_page.height = 100
        # Assume find_tables returns something truthy (list of tables)
        mock_page.find_tables.return_value = [1, 2, 3] 
        mock_page.extract_words.return_value = [{'x0': 10, 'x1': 90}]
        mock_page.annots = []

        mock_pdf.pages = [mock_page]
        mock_open.return_value.__enter__.return_value = mock_pdf

        profile = triage_agent.analyze("dummy.pdf")
        
        # Should detect table heavy
        assert profile.layout_complexity == LayoutComplexity.TABLE_HEAVY
        assert profile.estimated_extraction_cost == ExtractionCost.NEEDS_LAYOUT_MODEL

def test_domain_hint(triage_agent):
    """Test domain hint extraction."""
    with patch("pdfplumber.open") as mock_open:
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Please pay this Invoice Total: $500."
        mock_page.images = []
        mock_page.width = 100
        mock_page.height = 100
        mock_page.find_tables.return_value = []
        mock_page.extract_words.return_value = []
        mock_page.annots = []

        mock_pdf.pages = [mock_page]
        mock_open.return_value.__enter__.return_value = mock_pdf

        profile = triage_agent.analyze("dummy.pdf")
        
        assert profile.domain_hint == "financial"
