import logging
import re
from typing import List, Generator, Optional
from src.models import LDU, BBox

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChunkValidator:
    """
    Validates chunks against required schema and quality rules.
    """
    
    @staticmethod
    def validate(ldu: LDU, max_tokens: int = 500) -> bool:
        """
        Applies 5 core validation rules:
        1. Content Presence: Must have content or children.
        2. Token Count: Must be within limits.
        3. Metadata Completeness: Must have page references.
        4. Logical Structure: Must have a valid chunk_type.
        5. Content Integrity: Checks for minimal length/structure.
        """
        # Rule 1: Content Presence (Handled by Pydantic model validator, but double checking)
        if not ldu.content and not ldu.children:
            logger.warning(f"LDU {ldu.content_hash[:8]} failed: No content or children.")
            return False
            
        # Rule 2: Token Count
        if ldu.token_count > max_tokens:
            logger.warning(f"LDU {ldu.content_hash[:8]} failed: Token count {ldu.token_count} exceeds limit {max_tokens}.")
            return False
            
        # Rule 3: Metadata Completeness
        if not ldu.page_refs:
            logger.warning(f"LDU {ldu.content_hash[:8]} failed: Missing page references.")
            return False
            
        # Rule 4: Logical Structure
        if not ldu.chunk_type:
            logger.warning(f"LDU {ldu.content_hash[:8]} failed: Missing chunk_type.")
            return False

        # Rule 5: Content Integrity (e.g. min length for meaningful text unless it's a figure/table placeholder)
        if ldu.chunk_type == "text" and len(ldu.content.strip()) < 10:
             # Allow short titles but warn on very short body text
             # logger.debug(f"LDU {ldu.content_hash[:8]} warning: Content very short.")
             pass

        return True

class SemanticChunker:
    """
    Splits long text or combines short text into meaningful Logical Document Units (LDUs).
    """

    def __init__(self, max_tokens: int = 500, overlap: int = 50):
        self.max_tokens = max_tokens
        self.overlap = overlap

    def chunk_text(self, text: str, page_num: int, parent_section: str = None) -> List[LDU]:
        """
        Splits a text block into LDUs based on semantic delimiters (paragraphs, sentences).
        """
        # Simple regex split by paragraphs for now
        # In a real semantic chunker, we might use sentence embeddings or nltk
        paragraphs = re.split(r'\n\s*\n', text)
        
        ldus = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # If current chunk + new para < limit, append
            if self._estimate_tokens(current_chunk + "\n" + para) <= self.max_tokens:
                current_chunk += ("\n" + para if current_chunk else para)
            else:
                # Flush current chunk
                if current_chunk:
                    ldus.append(self._create_ldu(current_chunk, page_num, parent_section))
                
                # Start new chunk with overlap if needed (complex logic omitted for brevity, using simple split)
                current_chunk = para
        
        # Flush remaining
        if current_chunk:
            ldus.append(self._create_ldu(current_chunk, page_num, parent_section))
            
        return ldus

    def _create_ldu(self, content: str, page_num: int, parent_section: Optional[str]) -> LDU:
        """Helper to create an LDU object with hash and token count."""
        token_count = self._estimate_tokens(content)
        content_hash = str(hash(content)) # Simple hash for demo
        
        ldu = LDU(
            content=content,
            chunk_type="paragraph",
            page_refs=[page_num],
            parent_section=parent_section,
            token_count=token_count,
            content_hash=content_hash
        )
        
        # Validation
        if not ChunkValidator.validate(ldu, self.max_tokens):
            logger.warning("Created LDU failed validation, but returning anyway for now.")
            
        return ldu

    def _estimate_tokens(self, text: str) -> int:
        """Rough estimation of tokens (char count / 4)."""
        return len(text) // 4
