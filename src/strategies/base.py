from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import time
from src.models import ExtractedDocument

class BaseExtractor(ABC):
    """
    Abstract base class for all document extraction strategies.
    Each strategy must implement the `extract` method tailored to its specific approach.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the extractor with optional configuration.
        """
        self.config = config or {}

    @abstractmethod
    def extract(self, file_path: str) -> ExtractedDocument:
        """
        Extracts content from the document located at `file_path`.
        
        Args:
            file_path (str): Path to the document file.
            
        Returns:
            ExtractedDocument: The structured extracted content.
        """
        pass
