from typing import List, Optional, Union, Dict, Any, Tuple
from enum import Enum
from pydantic import BaseModel, Field

# --- Enums for Document Analysis ---

class OriginType(str, Enum):
    NATIVE_DIGITAL = "native_digital"
    SCANNED_IMAGE = "scanned_image"
    MIXED = "mixed"
    FORM_FILLABLE = "form_fillable"


class LayoutComplexity(str, Enum):
    SINGLE_COLUMN = "single_column"
    MULTI_COLUMN = "multi_column"
    TABLE_HEAVY = "table_heavy"
    FIGURE_HEAVY = "figure_heavy"
    MIXED = "mixed"

class ExtractionCost(str, Enum):
    FAST_TEXT_SUFFICIENT = "fast_text_sufficient"
    NEEDS_LAYOUT_MODEL = "needs_layout_model"
    NEEDS_VISION_MODEL = "needs_vision_model"

# --- Document Profile Model ---

class DocumentProfile(BaseModel):
    """
    Classifies the document to determine the optimal extraction strategy.
    """
    origin_type: OriginType = Field(
        ..., 
        description="Source format of the document (e.g., scanned vs native digital)."
    )
    layout_complexity: LayoutComplexity = Field(
        ..., 
        description="Estimate of structural complexity."
    )
    language: str = Field(
        "en", 
        description="Primary language code (ISO 639-1)."
    )
    domain_hint: Optional[str] = Field(
        None, 
        description="Optional hint about the document domain (e.g., 'financial', 'medical')."
    )
    estimated_extraction_cost: ExtractionCost = Field(
        ..., 
        description="Recommended extraction strategy based on complexity."
    )


# --- Helper Models for Extracted Elements ---

# Defining BoundingBox as a tuple of 4 floats: (x0, y0, x1, y1)
BoundingBox = Tuple[float, float, float, float]

class TextBlock(BaseModel):
    """Represents a block of text with its coordinates."""
    text: str
    bounding_box: BoundingBox
    page_number: int
    confidence: float = 1.0


class StructuredTable(BaseModel):
    """Represents a structured table extracted from the document."""
    data: List[List[Union[str, float, None]]]  # Simple row-major representation
    bounding_box: BoundingBox
    page_number: int
    caption: Optional[str] = None


class Figure(BaseModel):
    """Represents a figure/image extracted from the document."""
    alt_text: Optional[str] = None
    bounding_box: BoundingBox
    page_number: int
    image_ref: Optional[str] = None # Reference to stored image asset


# --- Extracted Document Model ---

class ExtractedDocument(BaseModel):
    """
    Container for all raw extracted elements from a document.
    """
    text_blocks: List[TextBlock] = Field(default_factory=list)
    structured_tables: List[StructuredTable] = Field(default_factory=list)
    figures: List[Figure] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# --- Logical Document Unit (LDU) ---

class LDU(BaseModel):
    """
    A semantic chunk of the document ready for downstream processing.
    """
    content: str = Field(..., description="The textual content of the unit.")
    chunk_type: str = Field(..., description="Type of content (e.g., 'paragraph', 'table', 'figure').")
    page_refs: List[int] = Field(..., description="List of page numbers where this content appears.")
    bounding_box: Optional[BoundingBox] = Field(None, description="Bounding box on the primary page.")
    parent_section: Optional[str] = Field(None, description="Title of the section this unit belongs to.")
    token_count: int = Field(..., description="Number of tokens in the content.")
    content_hash: str = Field(..., description="Hash of the content for deduplication.")


# --- Page Index (Tree Structure) ---

class PageIndex(BaseModel):
    """
    Recursive tree structure representing the document's table of contents or logical sections.
    """
    title: str
    start_page: int
    end_page: Optional[int] = None
    children: List["PageIndex"] = Field(default_factory=list)

# Resolve forward reference for recursive model
PageIndex.model_rebuild()


# --- Provenance Chain ---

class ProvenanceChain(BaseModel):
    """
    Tracks the lineage of extracted information back to the source document.
    """
    source_document_id: str
    source_ldu_id: Optional[str] = None
    page_number: int
    bounding_box: Optional[BoundingBox] = None
    confidence_score: float = 1.0
    extraction_method: str = Field(..., description="Method used for extraction (e.g., 'ocr', 'layout_analysis').")
