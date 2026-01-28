import os
import sys
from google import genai
from google.genai import types

# 1. Setup
print("🧠 DIAGNOSTIC: Google GenAI Embedding Test")
api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    print("   ❌ Error: GOOGLE_API_KEY environment variable not found.")
    sys.exit(1)
else:
    print(f"   🔑 API Key detected: {api_key[:5]}...{api_key[-4:]}")

try:
    client = genai.Client(api_key=api_key)
except Exception as e:
    print(f"   ❌ Client Init Failed: {e}")
    sys.exit(1)

# 2. Discovery Phase
print("\n🔎 Step 1: Listing ALL available models for your key...")
available_embedders = []

try:
    # We iterate and print everything to see what Google is giving you
    for m in client.models.list():
        # Clean up the model name (remove 'models/' prefix for display)
        name = m.name
        print(f"   - Found: {name}")
        
        # Check if it looks like an embedding model
        if "embed" in name.lower():
            available_embedders.append(name)

except Exception as e:
    print(f"   ⚠️ Model listing failed: {e}")
    # If listing fails, we manually try the usual suspects
    available_embedders = ["models/text-embedding-004", "models/embedding-001"]

print(f"\n📋 Candidates detected: {available_embedders}")

# 3. Testing Phase
print("\n🧪 Step 2: Attempting to generate embeddings...")

valid_model = None

for model_name in available_embedders:
    print(f"   👉 Testing: {model_name} ...", end=" ", flush=True)
    try:
        result = client.models.embed_content(
            model=model_name,
            contents="Hello world, this is a test.",
            config=types.EmbedContentConfig(task_type="CLUSTERING")
        )
        # If we get here, it worked
        dims = len(result.embeddings[0].values)
        print(f"✅ SUCCESS! (Vector Dimensions: {dims})")
        valid_model = model_name
        break # We found one that works!
    except Exception as e:
        print(f"❌ FAIL.")
        print(f"      Reason: {e}")

print("\n---------------------------------------------------")
if valid_model:
    print(f"🎉 CONCLUSION: Please update your script to use: '{valid_model}'")
else:
    print("💀 CONCLUSION: No embedding models are accessible. You MUST use Gen 210 (Offline Mode).")