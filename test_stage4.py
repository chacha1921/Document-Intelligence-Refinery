import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.getcwd())

from src.models.schemas import LDU
from src.agents.indexer import PageIndexBuilder

# Load env vars
load_dotenv()

def test_stage_4():
    print("Testing Stage 4: Page Index Builder...")
    
    # 1. Load LDUs from previous stage
    # Adjust filename if needed based on previous run
    input_file = Path(".refinery/chunks/tax_expenditure_ethiopia_2021_22_ldus.json")
    
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found. Run Stage 3 first.")
        return
        
    print(f"Loading LDUs from {input_file}...")
    with open(input_file, "r") as f:
        data = json.load(f)
        ldus = [LDU(**item) for item in data]
        
    print(f"Loaded {len(ldus)} chunks.")
    
    # 2. Initialize Builder (Local LLM Mode)
    # Using 'ollama' provider with 'llama3.2:latest' model as requested
    print("Initializing PageIndexBuilder with local LLM (Ollama)...")
    indexer = PageIndexBuilder(provider="ollama", model_name="llama3.2:latest")
    
    if indexer.provider == "gemini" and not indexer.api_key:
        print("Warning: GEMINI_API_KEY not found. Summaries will be mocked.")
    elif indexer.provider == "ollama" and not indexer.ollama_client:
        print("Warning: Could not connect to Ollama. Ensure 'ollama serve' is running.")
        
    # 3. Build Index
    doc_id = "tax_expenditure_ethiopia_2021_22"
    print(f"Building index for {doc_id}...")
    sections = indexer.build(ldus, doc_id)
    
    # 4. Verify Output
    output_file = Path(f".refinery/pageindex/{doc_id}_index.json")
    
    if output_file.exists():
        print(f"Success! Index saved to {output_file}")
        print(f"Total Sections: {len(sections)}")
        
        # Print first few sections
        for i, section in enumerate(sections[:3]):
            print(f"\nSection {i+1}: {section.title}")
            print(f"  Pages: {section.page_start}-{section.page_end}")
            print(f"  Summary: {section.summary[:100]}...")
            print(f"  Entities: {section.key_entities}")
    else:
        print("Error: Output file not created.")

if __name__ == "__main__":
    test_stage_4()