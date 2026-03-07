"""
Stage 3: The Semantic Chunking Engine.
Converts ExtractedDocument into Logical Document Units (LDUs) based on semantic rules.
"""
import re
import hashlib
import logging
from typing import List, Optional, Any, Dict
from src.models.schemas import ExtractedDocument, LDU, TextBlock, Table, Figure

logger = logging.getLogger(__name__)

class ChunkingEngine:
    """
    Groups extraction elements into semantic chunks (LDUs).
    Implements the Chunking Constitution rules.
    """
    
    def __init__(self):
        self.current_section = "Document Root"
        self.ldus: List[LDU] = []

    def chunk(self, doc: ExtractedDocument) -> List[LDU]:
        """
        Main entry point for chunking an ExtractedDocument.
        """
        self.ldus = []
        self.current_section = "Document Root"
        
        # Helper to map IDs to actual objects for fast lookup
        element_map: Dict[str, Any] = {}
        for block in doc.text_blocks:
            element_map[block.id] = block
        for table in doc.tables:
            element_map[table.id] = table
        for fig in doc.figures:
            element_map[fig.id] = fig
            
        for element_id in doc.reading_order:
            element = element_map.get(element_id)
            if not element:
                logger.warning(f"Element {element_id} in reading_order not found in element lists.")
                continue

            # Route based on type
            if isinstance(element, Table):
                self._process_table(element)
            elif isinstance(element, Figure):
                self._process_figure(element)
            elif isinstance(element, TextBlock):
                self._process_text(element)
            else:
                logger.warning(f"Unknown element type for {element_id}")

        return self.ldus

    def _process_table(self, table: Table):
        """
        Rule 1: Tables are single LDUs. Content is markdown.
        """
        # Serialize to Markdown if not already (assuming rows[0][0] holds MD from extractor refactor, 
        # or constructing it if strictly following schema)
        
        # Check if we have the markdown string stored in the first cell (as per previous refactor)
        # or construct it from headers/rows if it's a structural table
        content = ""
        if table.rows and len(table.rows) == 1 and len(table.rows[0]) == 1 and table.rows[0][0].strip().startswith("|"):
             # Likely pre-computed markdown
             content = table.rows[0][0]
        else:
            # Construct markdown
            headers_line = "| " + " | ".join(table.headers) + " |"
            separator_line = "| " + " | ".join(["---"] * len(table.headers)) + " |"
            rows_lines = []
            for row in table.rows:
                rows_lines.append("| " + " | ".join(row) + " |")
            content = f"{headers_line}\n{separator_line}\n" + "\n".join(rows_lines)

        self._create_ldu(
            content=content,
            chunk_type="table",
            page_refs=[table.page_number],
            bbox=table.bbox
        )

    def _process_figure(self, figure: Figure):
        """
        Rule 2: Figures.
        """
        content = f"[Figure ID: {figure.id}]"
        if figure.image_ref:
            content += f"(Ref: {figure.image_ref})"
        
        self._create_ldu(
            content=content,
            chunk_type="figure",
            page_refs=[figure.page_number],
            bbox=figure.bbox
        )

    def _process_text(self, block: TextBlock):
        text = block.text.strip()
        if not text:
            return

        # Rule 2 Extension: Caption Detection
        # If text starts with "Figure", current LDU is a figure, append it?
        # "If you detect a text block starting with "Figure", attach it to the preceding figure LDU"
        if text.lower().startswith("figure") and self.ldus and self.ldus[-1].chunk_type == "figure":
            last_ldu = self.ldus[-1]
            last_ldu.content += f"\nCaption: {text}"
            # Recalculate hash/bbox if strictly needed, but hash usually immutable ID. 
            # We'll just update content and maybe hash.
            last_ldu.content_hash = self._generate_hash(last_ldu.content, last_ldu.bounding_box, last_ldu.page_refs)
            return

        # Rule 4: Section Headers
        # Heuristic: Short length (< 100 chars), Title Case (mostly), No ending period
        is_header = False
        if len(text) < 100 and not text.endswith("."):
            # Simple title case check: majority of words start with uppercase?
            # Or just assume short line without period in reading order is likely a header.
            # Let's check for title case-ish (allowing for some stopwords)
            if text[0].isupper():
                 is_header = True
        
        if is_header:
            self.current_section = text
            self._create_ldu(
                content=text,
                chunk_type="header",
                page_refs=[block.page_number],
                bbox=block.bbox
            )
            return

        # Rule 3: Lists
        # Regex for numbered list: ^\d+\.\s
        is_list_item = re.match(r"^\d+\.\s", text)
        if is_list_item:
            # Check if previous LDU is a list
            if self.ldus and self.ldus[-1].chunk_type == "list":
                last_ldu = self.ldus[-1]
                if last_ldu.token_count < 500:
                    # Merge
                    last_ldu.content += f"\n{text}"
                    # Update metadata
                    last_ldu.bounding_box = self._merge_bboxes(last_ldu.bounding_box, block.bbox)
                    if block.page_number not in last_ldu.page_refs:
                        last_ldu.page_refs.append(block.page_number)
                    last_ldu.token_count = len(last_ldu.content.split())
                    last_ldu.content_hash = self._generate_hash(last_ldu.content, last_ldu.bounding_box, last_ldu.page_refs)
                    # Resolve CROSS-REFS on new content
                    self._resolve_relationships(last_ldu)
                    return
        
        # Default Text Chunk logic
        chunk_type = "list" if is_list_item else "text"

        # --- Fix 2: Merge Sequential Text Blocks ---
        if chunk_type == "text" and self.ldus:
            last_ldu = self.ldus[-1]
            # Check conditions: Last was text, Same Page (or close enough flow), Budget available
            if last_ldu.chunk_type == "text" and block.page_number in last_ldu.page_refs:
                new_token_count = last_ldu.token_count + len(text.split())
                if new_token_count < 500:
                    # Merge Logic
                    last_ldu.content += f"\n\n{text}"
                    last_ldu.bounding_box = self._merge_bboxes(last_ldu.bounding_box, block.bbox)
                    last_ldu.token_count = new_token_count
                    
                    # Update Hash (Content changed)
                    last_ldu.content_hash = self._generate_hash(last_ldu.content, last_ldu.bounding_box, last_ldu.page_refs)
                    
                    # Resolve any new relationships in the added text
                    self._resolve_relationships(last_ldu)
                    return

        # If merge not possible, create new
        self._create_ldu(
            content=text,
            chunk_type=chunk_type,
            page_refs=[block.page_number],
            bbox=block.bbox
        )

    def _create_ldu(self, content: str, chunk_type: str, page_refs: List[int], bbox: List[float]):
        """Creates and appends an LDU."""
        token_count = len(content.split()) # Simple whitespace tokenizer
        content_hash = self._generate_hash(content, bbox, page_refs)
        
        ldu = LDU(
            id=f"ldu_{content_hash[:8]}", # Deterministic ID or sequential? Hash is safer for dedup
            content=content,
            chunk_type=chunk_type,
            page_refs=page_refs,
            bounding_box=bbox,
            parent_section=self.current_section,
            token_count=token_count,
            content_hash=content_hash,
            relationships=[]
        )
        
        # Rule 5: Cross-references
        self._resolve_relationships(ldu)
        
        self.ldus.append(ldu)

    def _resolve_relationships(self, ldu: LDU):
        """
        Rule 5: Cross-references.
        Regex: (?:see\\s+)?(Table|Figure|Section)\\s+(\\d+)
        """
        matches = re.findall(r"(?:see\s+)?(Table|Figure|Section)\s+(\d+)", ldu.content, re.IGNORECASE)
        for ref_type, ref_id in matches:
            rel = f"{ref_type}_{ref_id}"
            if rel not in ldu.relationships:
                ldu.relationships.append(rel)

    def _generate_hash(self, content: str, bbox: List[float], page_refs: List[int]) -> str:
        """
        Generates a deterministic SHA-256 hash.
        """
        payload = f"{content}|{bbox}|{page_refs}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _merge_bboxes(self, bbox1: List[float], bbox2: List[float]) -> List[float]:
        """Union of two bounding boxes [x0, y0, x1, y1]"""
        # Assumes valid layout
        if not bbox1 or len(bbox1) != 4: return bbox2
        if not bbox2 or len(bbox2) != 4: return bbox1
        
        return [
            min(bbox1[0], bbox2[0]),
            min(bbox1[1], bbox2[1]),
            max(bbox1[2], bbox2[2]),
            max(bbox1[3], bbox2[3])
        ]
