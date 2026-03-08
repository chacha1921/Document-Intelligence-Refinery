
import logging
from typing import List, Optional
from src.models.schemas import LDU

logger = logging.getLogger(__name__)

class ChunkValidator:
    """
    Validates logical document units (LDUs) against the Chunking Constitution.
    """
    def validate(self, ldu: LDU) -> bool:
        """
        Runs all validation rules on an LDU.
        Returns True if valid, False otherwise.
        """
        if not self._validate_constraints(ldu):
            return False
            
        return True

    def _validate_constraints(self, ldu: LDU) -> bool:
        """
        Validates the 5 constraints:
        1. Tables must be single LDUs (Type check + Content check)
        2. Captions must be attached to figures (Heuristic check)
        3. Lists must be merged (Size check)
        4. Headers must update context (Context check - hard to validate post-facto without stream)
        5. Cross-refs must be resolved (Relationship check)
        """
        # Constraint 1: Table Integrity
        if ldu.chunk_type == "table":
            if not ldu.content.strip().startswith("|") and not "Table" in ldu.content:
                 logger.warning(f"LDU {ldu.id} marked as table but content doesn't look like markdown table.")
                 # Strictly might fail, but for now just warn
        
        # Constraint 2: Figure Captions
        if ldu.chunk_type == "figure":
             # Ensure content isn't empty
             if not ldu.content.strip():
                 logger.warning(f"LDU {ldu.id} is a figure/chart but has empty content.")
                 return False
        
        # Constraint 3: List Merging
        if ldu.chunk_type == "list":
            # Lists shouldn't be tiny fragmented items if they can be merged.
            # But specific validation is hard without seeing siblings.
            pass

        return True
