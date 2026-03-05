import pytest
from unittest.mock import MagicMock, patch
from src.agents.triage import TriageAgent, KeywordDomainClassifier
from src.models import OriginType, LayoutComplexity, ExtractionCost

# --- Mock Data ---

MOCK_CONFIG = {
    "domain_keywords": {
        "finance": ["invoice", "total", "tax"],
        "medical": ["patient", "diagnosis", "treatment"]
    },
    "thresholds": {
        "character_density_min": 50,
        "image_area_ratio_max": 0.4
    }
}

# --- Tests for Domain Classifier ---

def test_keyword_classifier_match():
    classifier = KeywordDomainClassifier(MOCK_CONFIG["domain_keywords"])
    text = "This invoice shows the total tax due."
    domain, conf = classifier.classify(text)
    assert domain == "finance"
    assert conf > 0.0

def test_keyword_classifier_no_match():
    classifier = KeywordDomainClassifier(MOCK_CONFIG["domain_keywords"])
    text = "The quick brown fox jumps over the lazy dog."
    domain, conf = classifier.classify(text)
    assert domain is None
    assert conf == 0.0

def test_keyword_classifier_empty_text():
    classifier = KeywordDomainClassifier(MOCK_CONFIG["domain_keywords"])
    domain, conf = classifier.classify("")
    assert domain is None
    assert conf == 0.0

# --- Tests for Triage Agent ---

@pytest.fixture
def mock_pdf():
    mock_pdf_obj = MagicMock()
    
    # Mock Page 1 (Text heavy)
    page1 = MagicMock()
    page1.extract_text.return_value = "This is a sample invoice text with total tax. " * 3 
    page1.width = 600
    page1.height = 800
    page1.images = []
    page1.find_tables.return_value = []
    # Mock extract_words to return valid bboxes
    page1.extract_words.return_value = [
        {'x0': 10, 'x1': 50, 'text': 'word1'}, 
        {'x0': 60, 'x1': 100, 'text': 'word2'}
    ] # Single column layout simulation
    page1.annots = []
    
    # It must be iterable if we iterate over pages
    mock_pdf_obj.pages = [page1]
    return mock_pdf_obj

@patch("src.agents.triage.pdfplumber.open")
@patch("src.agents.triage.TriageAgent._load_config")
def test_triage_agent_analyze_standard(mock_load_config, mock_pdf_open, mock_pdf):
    # Setup mocks
    mock_load_config.return_value = MOCK_CONFIG
    
    # We need to ensure pdfplumber.open returns a context manager that yields our mock_pdf
    mock_pdf_context = MagicMock()
    mock_pdf_context.__enter__.return_value = mock_pdf
    mock_pdf_open.return_value = mock_pdf_context
    
    # Run Agent
    agent = TriageAgent(config_path="dummy_path.yaml")
    profile = agent.analyze("dummy.pdf")
    
    # Assertions
    assert profile.origin_type == OriginType.NATIVE_DIGITAL
    assert profile.layout_complexity == LayoutComplexity.SINGLE_COLUMN
    assert profile.domain_hint == "finance"
    assert profile.estimated_extraction_cost == ExtractionCost.FAST_TEXT_SUFFICIENT
    assert profile.classification_confidence is not None

@patch("src.agents.triage.pdfplumber.open")
@patch("src.agents.triage.TriageAgent._load_config")
def test_triage_agent_scanned_image(mock_load_config, mock_pdf_open, mock_pdf):
    # Setup mocks
    mock_load_config.return_value = MOCK_CONFIG
    
    # Modify mock pdf to look like a scanned image (no text)
    mock_pdf.pages[0].extract_text.return_value = ""
    # Add a large image covering the page. Remember logic: w*h
    mock_pdf.pages[0].images = [{'x0': 0, 'top': 0, 'x1': 600, 'bottom': 800}] 
    
    mock_pdf_context = MagicMock()
    mock_pdf_context.__enter__.return_value = mock_pdf
    mock_pdf_open.return_value = mock_pdf_context
    
    # Run Agent
    agent = TriageAgent(config_path="dummy_path.yaml")
    profile = agent.analyze("dummy.pdf")
    
    # Assertions
    assert profile.origin_type == OriginType.SCANNED_IMAGE
    assert profile.estimated_extraction_cost == ExtractionCost.NEEDS_VISION_MODEL

@patch("src.agents.triage.pdfplumber.open")
@patch("src.agents.triage.TriageAgent._load_config")
def test_triage_agent_table_heavy(mock_load_config, mock_pdf_open, mock_pdf):
    # Setup mocks
    mock_load_config.return_value = MOCK_CONFIG
    
    # Modify to have tables
    # Triage logic: > 0.3 of pages have tables. 1/1 = 1.0 > 0.3
    mock_pdf.pages[0].find_tables.return_value = ["Table1", "Table2"]
    
    mock_pdf_context = MagicMock()
    mock_pdf_context.__enter__.return_value = mock_pdf
    mock_pdf_open.return_value = mock_pdf_context
    
    # Run Agent
    agent = TriageAgent(config_path="dummy_path.yaml")
    profile = agent.analyze("dummy.pdf")
    
    # Assertions
    # Note: Logic says > 0.3 ratio of pages with tables
    # However, look at logic: 
    # if total_pages > 0 and (table_heavy_count / total_pages) > 0.3:
    #    return LayoutComplexity.TABLE_HEAVY
    
    # Wait, my logic order in TriageAgent might return SCANNED_IMAGE first if text is low.
    # So I need to ensure text is high enough in this mock.
    # Text is "This is a sample invoice text with total tax." (length ~45 chars)
    # Default density threshold is 50 chars. So this might fail native text check in TriageAgent if not careful.
    # "This is a sample invoice text with total tax." is 45 chars.
    # Let's bump up the text length in the mock for safety.
    mock_pdf.pages[0].extract_text.return_value = "This is a sample invoice text with total tax. " * 3
    
    # Check assertions
    assert profile.layout_complexity == LayoutComplexity.TABLE_HEAVY
    # Needs layout model is correct for table heav
    assert profile.estimated_extraction_cost == ExtractionCost.NEEDS_LAYOUT_MODEL
