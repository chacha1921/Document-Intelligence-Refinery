
import os
from dotenv import load_dotenv, find_dotenv
from google import genai
from google.genai import types

# Force finding .env
path = find_dotenv(usecwd=True)
if path:
    print(f"Found .env at {path}")
    load_dotenv(path)
else:
    # Try finding it in Week3 root or similar
    # Assuming standard structure
    potential_paths = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(os.getcwd()), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), ".env")
    ]
    loaded = False
    for p in potential_paths:
        if os.path.exists(p):
            print(f"Found .env at {p}")
            load_dotenv(p)
            loaded = True
            break
    if not loaded:
        print("No .env found in standard locations")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    # Try to extract it from src/agents/indexer.py (hacky but effective if hidden)
    pass 

if not api_key:
    print("No GEMINI_API_KEY found after loading dotenv")
    exit(1)

client = genai.Client(api_key=api_key)

print("Listing supported embedding models...")
try:
    for m in client.models.list():
        print(f"Model: {m.name}")
        # print(dir(m)) # Debugging
        # The new SDK might not have supported_generation_methods or it's named differently
        # Just print all models for now to see what's available
except Exception as e:
    print(f"Error listing models: {e}")
