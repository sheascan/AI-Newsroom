import google.generativeai as genai
import os

# Make sure your API key is actually loaded
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ Error: GOOGLE_API_KEY is not set in your environment.")
else:
    genai.configure(api_key=api_key)
    print("✅ Key found. Listing available Gemini models...")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f" - {m.name}")