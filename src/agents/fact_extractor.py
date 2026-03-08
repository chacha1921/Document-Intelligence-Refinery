import json
import logging
import os
import sqlite3
from typing import List, Dict, Any
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from src.models.schemas import LDU
from src.storage.db import insert_fact, init_db, DB_PATH

logger = logging.getLogger(__name__)

# --- Pydantic Output Parser ---
class Fact(BaseModel):
    entity_name: str = Field(description="The primary subject of the fact (e.g., 'GDP', 'Tax Revenue', 'Inflation')")
    attribute: str = Field(description="The property being measured (e.g., 'total', 'growth rate', 'expenditure')")
    value: float = Field(description="The numerical value extracted. If strictly text, leave 0.0 or null.")
    unit: str = Field(description="The unit of measurement (e.g., 'USD', '%', 'billion ETB')")
    year: int = Field(description="The year this fact applies to, if available. 0 if not specified.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    source_text: str = Field(description="The exact snippet of text justifying this fact")

class FactExtraction(BaseModel):
    facts: List[Fact]

# --- Fact Extractor ---
class FactExtractor:
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not set. Fact extraction will be skipped.")
            self.llm = None
        else:
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0,
                google_api_key=api_key
            ).with_structured_output(FactExtraction)
            
            # Ensure DB is ready
            # init_db() # Call explicitly if needed, but safe to call multiple times? Currently init_db creates table IF NOT EXISTS.
            # But let's assume run_pipeline calls it once.
            
    def process_ldus(self, ldus: List[LDU], doc_id: str):
        """
        Iterates through LDUs, extracts facts, and saves to SQLite.
        """
        if not self.llm:
            return

        conn = sqlite3.connect(DB_PATH)
        success_count = 0
        
        # Filter relevant LDUs (tables and text with numbers)
        relevant_ldus = [
            ldu for ldu in ldus 
            if ldu.chunk_type == "table" or (ldu.chunk_type == "text" and any(c.isdigit() for c in ldu.content))
        ]
        
        logger.info(f"Extracting facts from {len(relevant_ldus)} relevant chunks...")
        
        for ldu in relevant_ldus:
            try:
                extraction = self._extract_from_chunk(ldu.content)
                if extraction and extraction.facts:
                    for fact in extraction.facts:
                        # Convert Pydantic to Dict and enrich with LDU metadata
                        fact_dict = fact.dict()
                        fact_dict['doc_id'] = doc_id
                        fact_dict['page_number'] = ldu.page_refs[0] if ldu.page_refs else 0
                        fact_dict['bbox'] = str(ldu.bounding_box)
                        fact_dict['content_hash'] = ldu.content_hash
                        
                        insert_fact(conn, fact_dict)
                        success_count += 1
            except Exception as e:
                logger.error(f"Error extracting facts from LDU {ldu.id}: {e}")
                
        conn.commit()
        conn.close()
        logger.info(f"Successfully extracted and stored {success_count} facts into {DB_PATH}")

    def _extract_from_chunk(self, content: str) -> FactExtraction:
        """
        Uses LLM to extract structured facts.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a precise data extraction agent. Extract numerical facts, financial figures, and key metrics from the provided text or table. Ignore general descriptions. Focus on Entity-Attribute-Value triples."),
            ("user", "Text/Table Content:\n{content}")
        ])
        
        chain = prompt | self.llm
        return chain.invoke({"content": content})

if __name__ == "__main__":
    # Test run
    extractor = FactExtractor()
    # Mock LDU
    mock_ldu = LDU(
        id="test_1",
        content="In 2021, the GDP growth rate was 5.6%.",
        chunk_type="text",
        page_refs=[1],
        bounding_box=[0,0,100,100],
        token_count=10,
        content_hash="abc"
    )
    extractor.process_ldus([mock_ldu], "test_doc")
