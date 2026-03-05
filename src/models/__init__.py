from typing import List, Optional, Union, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, model_validator

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
    classification_confidence: float = Field(
        1.0, 
        ge=0.0, 
        le=1.0, 
        description="Confidence score of the classification logic."
    )


# --- Helper Models for Extracted Elements ---

class BBox(BaseModel):
    """
    Represents a specialized bounding box with validation logic.
    """
    x0: float = Field(..., description="Left coordinate")
    y0: float = Field(..., description="Top coordinate")
    x1: float = Field(..., description="Right coordinate")
    y1: float = Field(..., description="Bottom coordinate")

    @model_validator(mode='after')
    def check_coordinates(self) -> 'BBox':
        if self.x0 > self.x1:
            raise ValueError(f"x0 ({self.x0}) cannot be greater than x1 ({self.x1})")
        if self.y0 > self.y1:
            raise ValueError(f"y0 ({self.y0}) cannot be greater than y1 ({self.y1})")
        return self

class TextBlock(BaseModel):
    """Represents a block of text with its coordinates."""
    text: str = Field(..., min_length=1, description="Content cannot be empty")
    bounding_box: BBox
    page_number: int = Field(..., gt=0, description="Page number (1-based)")
    confidence: float = Field(1.0, ge=0.0, le=1.0)


class StructuredTable(BaseModel):
    """Represents a structured table extracted from the document."""
    data: List[List[Union[str, float, None]]]  # Simple row-major representation
    bounding_box: BBox
    page_number: int = Field(..., gt=0)
    caption: Optional[str] = None


class Figure(BaseModel):
    """Represents a figure/image extracted from the document."""
    alt_text: Optional[str] = None
    bounding_box: BBox
    page_number: int = Field(..., gt=0)
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
    
    @property
    def strategy_history(self) -> List[str]:
        return self.metadata.get("strategy_history", [])
        
    @property
    def needs_human_review(self) -> bool:
        return self.metadata.get("needs_human_review", False)


# --- Logical Document Unit (LDU) ---

class LDU(BaseModel):
    """
    A semantic chunk of the document ready for downstream processing.
    """
    content: str = Field(..., description="The textual content of the unit.")
    chunk_type: str = Field(..., description="Type of content (e.g., 'paragraph', 'table', 'figure').")
    page_refs: List[int] = Field(..., description="List of page numbers where this content appears.")
    bounding_box: Optional[BBox] = Field(None, description="Bounding box on the primary page.")
    parent_section: Optional[str] = Field(None, description="Title of the section this unit belongs to.")
    token_count: int = Field(..., description="Number of tokens in the content.")
    content_hash: str = Field(..., description="Hash of the content for deduplication.")
    children: List["LDU"] = Field(default_factory=list, description="Child LDUs for hierarchical structure") 

    @model_validator(mode='after')
    def validate_content_presence(self) -> 'LDU':
        if not self.content and not self.children:
             raise ValueError("LDU must have either content or children.")
        return self

# Resolve forward reference for recursive model
LDU.model_rebuild()


# --- Page Index (Tree Structure) ---

class PageIndex(BaseModel):
    """
    Recursive tree structure representing the document's table of contents or logical sections.
    """
    title: str
    start_page: int = Field(..., gt=0)
    end_page: Optional[int] = Field(None, gt=0)
    children: List["PageIndex"] = Field(default_factory=list)
    ldu_refs: List[str] = Field(default_factory=list, description="IDs of LDUs in this section")

    @model_validator(mode='after')
    def check_page_order(self) -> 'PageIndex':
        if self.end_page is not None and self.start_page > self.end_page:
            raise ValueError(f"start_page ({self.start_page}) cannot be greater than end_page ({self.end_page})")
        return self

# Resolve forward reference for recursive model
PageIndex.model_rebuild()


# --- Provenance Chain ---

class ProvenanceChain(BaseModel):
    """
    Tracks the lineage of extracted information back to the source document.
    """
    source_document_id: str
    source_ldu_id: Optional[str] = None
    content_hash: Optional[str] = Field(None, description="Hash of the source content for integrity control")
    page_number: int = Field(..., gt=0)
    bounding_box: Optional[BBox] = None
    confidence_score: float = Field(1.0, ge=0.0, le=1.0)
    extraction_method: str = Field(..., description="Method used for extraction (e.g., 'ocr', 'layout_analysis').")
    timestamp: str = Field(..., description="ISO 8601 timestamp of extraction")

    @model_validator(mode='after')
    def validate_source_tracing(self) -> 'ProvenanceChain':
        if not self.source_ldu_id and not self.bounding_box:
             # Ideally one should exist to trace back, though not strictly mandatory in all loose coupled systems
             pass 
        return self

class FactRecord(BaseModel):
    """
    Represents a discrete fact extracted from the document for the data layer.
    """
    entity: str
    attribute: str
    value: Union[str, float, int]
    unit: Optional[str] = None
    provenance: ProvenanceChain
