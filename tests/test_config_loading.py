import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.triage import TriageAgent

def test_config_loading():
    print("Testing config loading...")
    agent = TriageAgent()
    
    # Check if rules are loaded
    if not agent.rules:
        print("FAIL: Rules not loaded.")
        return
        
    print(f"Rules loaded: {agent.rules.keys()}")
    
    # Check thresholds
    thresholds = agent.rules.get("thresholds", {})
    print(f"Thresholds: {thresholds}")
    if "character_density_min" not in thresholds or thresholds["character_density_min"] != 50.0:
        print("FAIL: Incorrect thresholds.")
        return
    
    # Check domain keywords
    keywords = agent.rules.get("domain_keywords", {})
    print(f"Keywords keys: {keywords.keys()}")
    if "financial" not in keywords:
        print("FAIL: Missing domain keywords.")
        return
    
    # Check env var
    load_dotenv()
    print(f"OLLAMA_BASE_URL: {os.getenv('OLLAMA_BASE_URL')}")
    
    print("PASS: Config loading test passed.")

if __name__ == "__main__":
    test_config_loading()
