"""
Main entry point for the Document Intelligence Refinery pipeline.
"""
import argparse
import sys
import logging
from pathlib import Path
from src.agents.triage import TriageAgent

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
        
    except Exception as e:
        logger.error(f"Triage failed: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Document Intelligence Refinery: Enterprise Document Processing Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline stage to run")

    # Stage 1: Triage
    triage_parser = subparsers.add_parser("triage", help="Run Stage 1: Triage Signal Analysis")
    triage_parser.add_argument("file", type=str, help="Path to the input document (PDF)")

    # Future stages can be added here
    # extract_parser = subparsers.add_parser("extract", ...)
    
    args = parser.parse_args()

    if args.command == "triage":
        run_triage(args.file)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
