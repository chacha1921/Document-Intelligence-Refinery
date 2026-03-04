import pytest
from unittest.mock import MagicMock, patch
from src.strategies.fast_text import FastTextExtractor
from src.models import ExtractedDocument

@pytest.fixture
def extractor():
    return FastTextExtractor()

def test_extract_high_confidence(extractor):
    """Test text extraction with high confidence."""
    with patch("pdfplumber.open") as mock_open:
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        
        # Simulate high text content, no images
        mock_page.extract_text.return_value = "This is a native digital document. " * 10
        mock_page.images = []
        mock_page.width = 100
        mock_page.height = 100
        # Assume extract_words returns list of dicts with coordinate keys
        mock_page.extract_words.return_value = [
            {'x0': 10, 'top': 10, 'x1': 90, 'bottom': 20},
            # ... more words
        ]

        mock_pdf.pages = [mock_page]
        mock_open.return_value.__enter__.return_value = mock_pdf

        doc = extractor.extract("dummy.pdf")
        
        confidence = doc.metadata.get("low_confidence_pages", [])
        
        # Should be empty or none, since confidence is high
        assert len(confidence) == 0
        assert doc.text_blocks[0].confidence > 0.8
        
def test_extract_low_confidence_scanner(extractor):
    """Test text extraction flagging low confidence on image-heavy pages."""
    with patch("pdfplumber.open") as mock_open:
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        
        # Simulate image-heavy page
        mock_page.extract_text.return_value = "Scanned text..." # Low text
        mock_page.images = [{'x0': 0, 'top': 0, 'x1': 100, 'bottom': 100}]
        mock_page.width = 100
        mock_page.height = 100
        mock_page.extract_words.return_value = []

        mock_pdf.pages = [mock_page]
        mock_open.return_value.__enter__.return_value = mock_pdf

        doc = extractor.extract("dummy.pdf")
        
        # Should flag page 1 as low confidence
        low_confidence_pages = doc.metadata.get("low_confidence_pages", [])
        assert 1 in low_confidence_pages
        
        # Check text block confidence
        if doc.text_blocks:
            assert doc.text_blocks[0].confidence < 0.5
