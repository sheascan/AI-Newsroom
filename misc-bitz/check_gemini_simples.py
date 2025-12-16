import os
from google import genai

# CHECK API KEY
if not os.environ.get("GOOGLE_API_KEY"):
    print("❌ Error: GOOGLE_API_KEY not found.")
    exit(1)

client = genai.Client()

print("🔍 Scanning for models...")

try:
    # Just grab everything
    pager = client.models.list()
    
    found_any = False
    for model in pager:
        # We just print the 'name' attribute which usually looks like "models/gemini-1.5-flash"
        if hasattr(model, 'name'):
            print(f"   👉 {model.name}")
            found_any = True
        
    if not found_any:
        print("   ⚠️ No models returned. Your API key might be valid but have no scope?")

except Exception as e:
    print(f"   ❌ Error: {e}")