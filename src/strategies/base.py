"""
Strategy interface for the multi-strategy extraction engine.
"""
from abc import ABC, abstractmethod
from typing import Tuple
from src.models.schemas import ExtractedDocument

class BaseExtractor(ABC):
    """
    Abstract base class for all extraction strategies.
    Ensures a consistent interface for the extraction router.
    """
    
    @abstractmethod
    def extract(self, file_path_or_bytes: str | bytes) -> Tuple[ExtractedDocument, float]:
        """
        Extracts content from a document (page or file) and returns the 
        normalized ExtractedDocument object along with a confidence score.

        Args:
            file_path_or_bytes (str | bytes): The input document artifact.

        Returns:
            Tuple[ExtractedDocument, float]: A tuple containing the extraction result
                                             and a confidence score (0.0 to 1.0).
        """
        pass
