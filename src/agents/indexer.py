import os
import json
import logging
from typing import List, Dict, Any, Optional, Literal
from itertools import groupby
from pathlib import Path

from google import genai
from google.genai import types
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

from src.models.schemas import LDU, SectionNode

load_dotenv()
logger = logging.getLogger(__name__)

class PageIndexBuilder:
    """
    Builds a hierarchical page index from a list of LDUs (Logical Document Units).
    Supports both Google Gemini (Cloud) and Ollama (Local) backends.
    """
    def __init__(
        self, 
        provider: Literal["gemini", "ollama"] = "ollama",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.provider = provider
        self.gemini_client = None
        self.ollama_client = None
        
        if self.provider == "gemini":
            if not self.api_key:
                logger.warning("GEMINI_API_KEY not found. LLM enrichment will be skipped.")
            else:
                self.gemini_client = genai.Client(api_key=self.api_key)
            self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-1.5-flash-001")
            
        elif self.provider == "ollama":
            # Connect to local Ollama instance via OpenAI-compatible API
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            ollama_key = "ollama" # Required but unused by Ollama
            try:
                self.ollama_client = OpenAI(base_url=base_url, api_key=ollama_key)
                logger.info(f"Connected to Local LLM at {base_url}")
            except Exception as e:
                logger.error(f"Failed to connect to Ollama: {e}")
            
            # Default to a fast local model if not specified
            self.model_name = model_name or "llama3.2:latest" 

    def build(self, ldus: List[LDU], doc_id: str) -> List[SectionNode]:
        """
        Groups LDUs by parent_section, enriches them with LLM summaries, 
        and returns a list of SectionNodes.
        """
        # 1. Sort LDUs by parent_section (required for groupby)
        # We handle None parent_section by converting to empty string for sorting
        sorted_ldus = sorted(ldus, key=lambda x: x.parent_section if x.parent_section else "")
        
        sections: List[SectionNode] = []
        
        # 2. Group by parent_section
        for section_title, group in groupby(sorted_ldus, key=lambda x: x.parent_section):
            section_ldus = list(group)
            if not section_title:
                section_title = "Uncategorized"
                
            # Metadata Extraction
            page_numbers = []
            for ldu in section_ldus:
                page_numbers.extend(ldu.page_refs)
            
            if not page_numbers:
                page_start = 0
                page_end = 0
            else:
                page_start = min(page_numbers)
                page_end = max(page_numbers)
                
            data_types = sorted(list(set(ldu.chunk_type for ldu in section_ldus)))
            
            # 3. LLM Enrichment
            summary = "Summary unavailable"
            key_entities = []
            
            try:
                # Check for active client
                has_client = (self.provider == "gemini" and self.gemini_client) or \
                             (self.provider == "ollama" and self.ollama_client)
                             
                if has_client:
                    # Aggregate text
                    combined_text = "\n".join([ldu.content for ldu in section_ldus])
                    # Truncate to save token window
                    truncated_text = combined_text[:12000] 
                    
                    # Guardrail: Check word count before sending to LLM to prevent hallucinations on tiny chunks
                    word_count = len(truncated_text.split())
                    if word_count < 30:
                        summary = truncated_text
                        key_entities = []
                    else:
                        summary, key_entities = self._enrich_section(section_title, truncated_text)
            except Exception as e:
                logger.error(f"Error enriching section '{section_title}': {e}")
                
            # Create SectionNode
            node = SectionNode(
                title=section_title,
                page_start=page_start,
                page_end=page_end,
                child_sections=[], # Flat structure for now as per grouping logic
                key_entities=key_entities,
                summary=summary,
                data_types_present=data_types
            )
            sections.append(node)
            
        # 4. Sort by page_start
        sections.sort(key=lambda x: x.page_start)
            
        # 5. Output: Save to JSON
        output_dir = Path(".refinery/pageindex")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{doc_id}_index.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json.dumps([s.model_dump() for s in sections], indent=2))
        
        logger.info(f"Page index saved to {output_file}")
        
        return sections

    def _enrich_section(self, title: str, text: str) -> tuple[str, List[str]]:
        """
        Uses configured LLM provider to generate summary and key entities.
        """
        # Improved Prompt with precision instruction
        prompt = f"""
        You are a precise data extractor. Summarize the text. Do not invent external facts, names, or organizations.
        
        You are summarizing a section of a document titled "{title}".
        Content:
        {text}
        
        Please provide:
        1. A 2-3 sentence concise summary of this section.
        2. A list of 3-5 critical named entities (specific taxes, metrics, organizations, people) found in the text.
        
        Return valid JSON with keys 'summary' and 'key_entities'.
        """
        
        try:
            data = {}
            if self.provider == "gemini" and self.gemini_client:
                response = self.gemini_client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                data = json.loads(response.text)
            
            elif self.provider == "ollama" and self.ollama_client:
                response = self.ollama_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that extracts information in JSON format."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content
                data = json.loads(content)
            
            summary = data.get("summary", "Summary unavailable")
            raw_entities = data.get("key_entities", [])
            
            # Normalize entities to strings
            clean_entities = []
            if isinstance(raw_entities, list):
                for item in raw_entities:
                    if isinstance(item, str):
                        clean_entities.append(item)
                    elif isinstance(item, dict):
                        # Extract first string value or stringify
                        val = next((v for v in item.values() if isinstance(v, str)), str(item))
                        clean_entities.append(val)
                    else:
                        clean_entities.append(str(item))
                        
            return summary, clean_entities

        except Exception as e:
            logger.warning(f"LLM generation failed for section '{title}' with {self.provider}: {e}")
            return "Summary unavailable", []
