"""
Main entry point for the Document Intelligence Refinery pipeline.
"""
import argparse
import sys
import logging
from pathlib import Path
from src.agents.triage import TriageAgent
from src.agents.extractor import ExtractionRouter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def run_triage(file_path: str):
    """
    Executes the Stage 1 Triage process.
    """
    input_path = Path(file_path)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    logger.info(f"Starting Triage for: {input_path.name}")
    
    try:
        agent = TriageAgent()
        profile = agent.analyze(str(input_path))
        
        logger.info("Triage Complete.")
        print("\n--- Document Profile ---")
        print(profile.model_dump_json(indent=2))
        print("------------------------\n")
        return profile
        
    except Exception as e:
        logger.error(f"Triage failed: {e}")
        sys.exit(1)

def run_process(file_path: str):
    """
    Executes the full Pipeline: Stage 1 (Triage) -> Stage 2 (Extraction)
    """
    input_path = Path(file_path)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    logger.info(f"Starting End-to-End Process for: {input_path.name}")
    
    try:
        # Step 1: Triage
        triage_agent = TriageAgent()
        profile = triage_agent.analyze(str(input_path))
        logger.info(f"Triage Complete. Strategy: {profile.estimated_extraction_cost}")
        
        # Step 2: Extraction
        router = ExtractionRouter()
        extracted_doc = router.extract(str(input_path), profile=profile)
        
        logger.info("Extraction Complete.")
        print("\n--- Extracted Document Summary ---")
        print(f"Doc ID: {extracted_doc.doc_id}")
        print(f"Reading Order Items: {len(extracted_doc.reading_order)}")
        print(f"Text Blocks: {len(extracted_doc.text_blocks)}")
        print(f"Tables: {len(extracted_doc.tables)}")
        print("----------------------------------\n")
        
        # Determine output file
        output_path = Path(".refinery/extracted") / f"{input_path.stem}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(extracted_doc.model_dump_json(indent=2))
        logger.info(f"Extracted data saved to: {output_path}")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Document Intelligence Refinery: Enterprise Document Processing Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline stage to run")

    # Stage 1: Triage
    triage_parser = subparsers.add_parser("triage", help="Run Stage 1: Triage Signal Analysis")
    triage_parser.add_argument("file", type=str, help="Path to the input document (PDF)")

    # Full Pipeline: Process
    process_parser = subparsers.add_parser("process", help="Run Full Pipeline (Triage -> Extraction)")
    process_parser.add_argument("file", type=str, help="Path to the input document (PDF)")
    
    args = parser.parse_args()

    if args.command == "triage":
        run_triage(args.file)
    elif args.command == "process":
        run_process(args.file)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
