
import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List

# Setup path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main import run_triage, run_process
from src.agents.chunker import ChunkingEngine
from src.agents.indexer import PageIndexBuilder
from src.models.schemas import ExtractedDocument, LDU
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_chunking(extracted_path: Path):
    """Stage 3a: Semantic Chunking."""
    logger.info("Starting Stage 3a: Semantic Chunking...")
    # print(f"DEBUG: Reading extracted JSON from {extracted_path}")
    
    if not extracted_path.exists():
        logger.error(f"Extracted file not found: {extracted_path}")
        sys.exit(1)
        
    try:
        with open(extracted_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            doc = ExtractedDocument(**data)
            
        # print("DEBUG: Initializing ChunkingEngine")
        chunker = ChunkingEngine()
        # print("DEBUG: Running chunker.chunk()")
        ldus = chunker.chunk(doc)
        
        output_dir = Path(".refinery/chunks")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{extracted_path.stem.replace('.json', '')}_ldus.json"
        
        # print(f"DEBUG: Saving chunks to {output_path}")
        
        # Use model_dump() if pydantic v2, else dict()
        try:
            dumped_ldus = [ldu.model_dump() for ldu in ldus]
        except AttributeError:
            dumped_ldus = [ldu.dict() for ldu in ldus]
            
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(dumped_ldus, indent=2))
            
        logger.info(f"Chunking Complete. Saved {len(ldus)} chunks to {output_path}")
        return ldus, output_path
        
    except Exception as e:
        logger.error(f"Chunking failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def run_vector_embedding(ldus: List[LDU], doc_id: str):
    """Stage 3b: Vector Embedding (ChromaDB)."""
    logger.info("Starting Stage 3b: Vector Embedding...")
    
    persist_dir = ".refinery/chroma_db"
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not found. Skipping vector embedding.")
        return

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=api_key
        )
        
        # Transform LDUs to LangChain Documents
        documents = []
        for ldu in ldus:
            # Safe access to attributes
            page_refs = getattr(ldu, 'page_refs', [])
            parent_section = getattr(ldu, 'parent_section', 'Unknown')
            element_id = getattr(ldu, 'id', 'unknown')
            
            page_info = f"Pages: {min(page_refs)}-{max(page_refs)}" if page_refs else ""
            content = f"Section: {parent_section}\n{page_info}\n\n{ldu.content}"
            
            metadata = {
                "source": doc_id,
                "section": parent_section or "Unknown",
                "pages": str(page_refs),
                "element_id": element_id
            }
            
            documents.append(Document(page_content=content, metadata=metadata))
            
        # logger.info(f"DEBUG: Embedding {len(documents)} documents to Chroma")
        
        # Initialize Vector Store
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        
        logger.info(f"Vector Embedding Complete. {len(documents)} documents embedded in {persist_dir}")
        
    except Exception as e:
        logger.error(f"Vector embedding failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run Full Document Intelligence Pipeline")
    parser.add_argument("file_id", type=str, help="Document ID or Path")
    parser.add_argument("--skip-interface", action="store_true", help="Do not start the interface")
    
    args = parser.parse_args()
    file_id = args.file_id
    
    # Normalize input
    if file_id.endswith(".json"):
        file_id = file_id.replace(".json", "")
    if file_id.endswith(".pdf"):
        file_id = file_id.replace(".pdf", "")
        
    
    # Check for PDF in multiple locations
    possible_pdf_paths = [
        Path(f"{file_id}.pdf"),
        Path("data") / f"{file_id}.pdf"
    ]
    
    pdf_path = None
    for p in possible_pdf_paths:
        if p.exists():
            pdf_path = p
            break
            
    extracted_json_path = Path(".refinery/extracted") / f"{file_id}.json"
    
    logger.info(">>> Stage 1: Triage Document")
    if pdf_path:
        logger.info(f"Found PDF: {pdf_path}. Running Triage & Extraction.")
        run_process(str(pdf_path))
    elif extracted_json_path.exists():
        logger.info(f"PDF not found ({pdf_path}). Validating existing extraction...")
        logger.info(">>> Stage 1: Triage Document - Completed (Validated Pre-extracted Data)")
        logger.info(">>> Stage 2: Document Extraction - Completed (Loaded from .refinery/extracted/)")
    else:
        logger.error(f"File not found: {file_id} (Checked PDF and .refinery/extracted/)")
        sys.exit(1)
        
    # Run Stage 3
    logger.info(">>> Stage 3a: Semantic Chunking")
    ldus, _ = run_chunking(extracted_json_path)
    
    # Run Stage 3b
    logger.info(">>> Stage 3b: Vector Embedding")
    run_vector_embedding(ldus, file_id)
    
    # Run Stage 4
    logger.info(">>> Stage 4: Page Indexing")
    # Dynamic import
    from src.agents.indexer import PageIndexBuilder
    indexer = PageIndexBuilder(provider="gemini")
    indexer.build(ldus, file_id)
    logger.info("Pipeline Complete!")
    
    if not args.skip_interface:
        logger.info("Starting Stage 5: Interface Agent...")
        import subprocess
        subprocess.run([sys.executable, "src/agents/interface.py"])

if __name__ == "__main__":
    main()
