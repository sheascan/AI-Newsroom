import os
from google import genai

# Just strictly listing models
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

print("🔍 Asking Google for available models...")
try:
    # List models and filter for 'generateContent' support
    models = client.models.list()
    found = False
    for m in models:
        # Check if it supports content generation (chat)
        if "generateContent" in m.supported_generation_methods:
            name = m.name.split("/")[-1] # Clean up 'models/gemini-...'
            print(f"   ✅ FOUND: {name}")
            if "flash" in name:
                print(f"      (This is a good candidate for speed)")
            found = True
            
    if not found:
        print("   ❌ No content generation models found. Check API Key permissions?")
        
except Exception as e:
    print(f"   ❌ Error: {e}")