import json
from pathlib import Path
from src.models.schemas import ExtractedDocument
from src.agents.chunker import ChunkingEngine
# from src.agents.indexer import PageIndexBuilder # Commented out as Stage 4 is pending

def test_stage3(doc_id: str):
    print(f"--- Starting Stage 3 Test for {doc_id} ---")
    
    # 1. Load the Extracted JSON from Stage 2
    input_path = Path(f".refinery/extracted/{doc_id}.json")
    if not input_path.exists():
        print(f"Error: {input_path} not found. Run Stage 2 first.")
        return

    with open(input_path, "r") as f:
        data = json.load(f)
        extracted_doc = ExtractedDocument(**data)

    # 2. Run the Chunking Engine
    print("Running Semantic Chunking Engine...")
    chunker = ChunkingEngine()
    ldus = chunker.chunk(extracted_doc)
    
    # Save the LDUs to disk so you can inspect them
    chunks_dir = Path(".refinery/chunks")
    chunks_dir.mkdir(parents=True, exist_ok=True)
    output_path = chunks_dir / f"{doc_id}_ldus.json"
    
    # Convert LDUs to dicts for JSON serialization
    serialized_ldus = [ldu.model_dump() for ldu in ldus]
    
    with open(output_path, "w") as f:
        json.dump(serialized_ldus, f, indent=2)
        
    print(f"Success! Generated {len(ldus)} chunks.")
    print(f"Saved to {output_path}")
    
    # Inspect first few chunks
    print("\n--- Preview of First 3 Chunks ---")
    for i, ldu in enumerate(ldus[:3]):
        print(f"Chunk {i+1} [{ldu.chunk_type}]: {ldu.content[:100]}...")

    # 3. Run the PageIndex Builder (Placeholder until Stage 4)
    # print("Running PageIndex Builder (Calling Gemini for Summaries)...")
    # indexer = PageIndexBuilder()
    # index_nodes = indexer.build(ldus)
    
    # # Save the PageIndex to disk
    # index_dir = Path(".refinery/pageindex")
    # index_dir.mkdir(parents=True, exist_ok=True)
    # with open(index_dir / f"{doc_id}_index.json", "w") as f:
    #     json.dump(index_nodes, f, indent=2)
    # print(f"Success! Built {len(index_nodes)} section nodes. Saved to {index_dir}/{doc_id}_index.json")

if __name__ == "__main__":
    # Test it on the document we just processed
    test_stage3("tax_expenditure_ethiopia_2021_22")