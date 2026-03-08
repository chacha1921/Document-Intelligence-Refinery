
import os
from google import genai

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("No GEMINI_API_KEY found")
    exit(1)

client = genai.Client(api_key=api_key)

print("Listing models...")
for m in client.models.list():
     print(f"Name: {m.name}")
