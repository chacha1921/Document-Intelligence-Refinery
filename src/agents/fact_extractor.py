from typing import List
from src.models import ExtractedDocument, FactRecord

class FactTableExtractor:
    """
    Simulated implementation of the FactTable extractor. 
    In a real scenario, this would use LLMs to extract KV pairs from LDUs.
    """
    
    def extract_facts(self, doc: ExtractedDocument) -> List[FactRecord]:
        """
        Scan document for structured facts.
        """
        # Placeholder logic
        return []
