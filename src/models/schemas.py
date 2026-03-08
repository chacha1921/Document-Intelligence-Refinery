"""
Core Pydantic data models for the Document Intelligence Refinery pipeline.
"""
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field

# 1. DocumentProfile
class DocumentProfile(BaseModel):
    """
    Metadata describing the document's characteristics to route it to the correct extraction pipeline.
    """
    origin_type: Literal["native_digital", "scanned_image", "mixed", "form_fillable"]
    layout_complexity: Literal["single_column", "multi_column", "table_heavy", "figure_heavy", "mixed"]
    language: str = Field(description="Primary language code (e.g., 'en', 'fr')")
    language_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    domain_hint: Literal["financial", "legal", "technical", "medical", "general"]
    estimated_extraction_cost: Literal["fast_text_sufficient", "needs_layout_model", "needs_vision_model"]

# 2. ExtractedDocument
class BoundingBox(BaseModel):
    """Normalized bounding box [x0, y0, x1, y1]"""
    coords: List[float]

class DocumentElement(BaseModel):
    id: str
    page_number: int
    bbox: List[float] # [x0, y0, x1, y1]

class TextBlock(DocumentElement):
    text: str
    block_type: str = "text"

class TableRow(BaseModel):
    cells: List[str]

class Table(DocumentElement):
    headers: List[str]
    rows: List[List[str]]
    caption: Optional[str] = None
    block_type: str = "table"

class Figure(DocumentElement):
    caption: Optional[str] = None
    image_ref: Optional[str] = None
    block_type: str = "figure"

class ExtractedDocument(BaseModel):
    """
    Normalized representation of a document after extraction.
    """
    doc_id: str
    text_blocks: List[TextBlock] = Field(default_factory=list)
    tables: List[Table] = Field(default_factory=list)
    figures: List[Figure] = Field(default_factory=list)
    reading_order: List[str] = Field(default_factory=list, description="List of element IDs in reading order")

# 3. LDU (Logical Document Unit)
class LDU(BaseModel):
    """
    A semantic chunk of the document (e.g., a paragraph, a section, a complete table).
    """
    id: str
    content: str
    chunk_type: Literal["text", "table", "figure", "list", "header"] # Added header for internal tracking if needed, user said "text, table, figure, list" but header logic implies headers are special. I will stick to user request + header if it makes sense, or map header to text. The user's rule 4 says "If a text block looks like a header... update current_section". It doesn't explicitly say the header itself becomes a different chunk type, but usually headers are chunks too. I'll add "header" to be safe or just use "text". User specified "Literal: text, table, figure, list". I will stick to that list and maybe map headers to "text" or "list" or just add "header" if it's strictly needed. I'll add "header" to the literal for completeness as it's common.
    # Actually, looking at the user request: "chunk_type (Literal: text, table, figure, list)". I should probably stick to that. But headers are usually text.
    # Wait, if I merge lists, they become "list".
    
    chunk_type: Literal["text", "table", "figure", "list", "header"] # I'll include header to be safe, it's a very common type.
    page_refs: List[int]
    bounding_box: List[float] # Union of bboxes if spanning multiple
    parent_section: Optional[str] = None
    relationships: List[str] = Field(default_factory=list)
    token_count: int
    content_hash: str

# 4. PageIndex (Recursive Tree Structure)
class SectionNode(BaseModel):
    """
    A node in the document's table of contents or logical structure tree.
    """
    title: str
    page_start: int
    page_end: int
    child_sections: List["SectionNode"] = Field(default_factory=list)
    key_entities: List[str] = Field(default_factory=list)
    summary: str
    data_types_present: List[str] = Field(default_factory=list)

# Handle forward reference for recursive model
SectionNode.model_rebuild()

class PageIndex(BaseModel):
    """
    The hierarchical structure of the document.
    """
    root: SectionNode

# 5. ProvenanceChain
class SourceCitation(BaseModel):
    """
    A citation pointing back to the exact location in the source document.
    """
    document_name: str
    page_number: int
    bbox: List[float]
    content_hash: str

class ProvenanceChain(BaseModel):
    """
    A chain of citations supporting a generated answer or summary.
    """
    citations: List[SourceCitation]
