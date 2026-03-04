import pytest
from unittest.mock import MagicMock, patch
from src.agents.extractor import ExtractionRouter
from src.strategies.fast_text import FastTextExtractor
from src.strategies.layout import LayoutExtractor
from src.models import DocumentProfile, ExtractedDocument, ExtractionCost
from unittest.mock import Mock

@pytest.fixture
def router():
    return ExtractionRouter(ledger_path=".refinery/test_ledger.jsonl")

def test_extract_native_high_confidence(router):
    """Test successful high-confidence extraction (FastText)."""
    with patch.object(router, 'fast_text', autospec=True) as mock_ft:
        profile = DocumentProfile(
            origin_type="native_digital",
            layout_complexity="single_column",
            estimated_extraction_cost="fast_text_sufficient"
        )
        
        # Simulate successful FT Extraction
        mock_doc = ExtractedDocument(text_blocks=[MagicMock(confidence=1.0)], metadata={"confidence": 0.95})
        mock_ft.extract.return_value = mock_doc
        
        result = router.extract("dummy.pdf", profile)
        
        assert result.metadata["confidence"] == 0.95
        assert mock_ft.extract.called

def test_extract_escalation(router):
    """Test escalation when FastText fails/low confidence."""
    with patch.object(router, 'fast_text', autospec=True) as mock_ft, \
         patch.object(router, 'layout', autospec=True) as mock_layout:
         
        profile = DocumentProfile(
            origin_type="native_digital",
            layout_complexity="single_column",
            estimated_extraction_cost="fast_text_sufficient"
        )
        
        # 1. FastText returns low confidence
        mock_ft.extract.return_value = ExtractedDocument(
            text_blocks=[], # Maybe empty text
            metadata={"low_confidence_pages": [1]} # Flagged page
        )
        
        # 2. Layout Extractor succeeds
        mock_layout.extract.return_value = ExtractedDocument(
            text_blocks=[MagicMock(confidence=1.0)],
            metadata={"confidence": 0.99, "extractor": "LayoutExtractor"}
        )
        
        result = router.extract("dummy.pdf", profile)
        
        # Verify both called
        assert mock_ft.extract.called
        assert mock_layout.extract.called
        
        # Result should be from Layout
        assert result.metadata["extractor"] == "LayoutExtractor"

def test_extract_direct_layout(router):
    """Test direct routing to LayoutExtractor based on complexity."""
    with patch.object(router, 'layout', autospec=True) as mock_layout:
        profile = DocumentProfile(
            origin_type="native_digital",
            layout_complexity="table_heavy",
            estimated_extraction_cost="needs_layout_model"
        )
        
        mock_layout.extract.return_value = ExtractedDocument(
            text_blocks=[], 
            metadata={"confidence": 1.0}
        )
        
        router.extract("dummy.pdf", profile)
        
        assert mock_layout.extract.called

