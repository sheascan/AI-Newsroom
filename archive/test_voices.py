import asyncio
import edge_tts
from pydub import AudioSegment
import os

# --- TEST CONFIGURATION ---
OUTPUT_FILE = "voice_test_output.mp3"

# We manually define the dialogue to ensure the input is perfect
SCRIPT = [
    ("Alice", "Hello Bob, can you hear me clearly?", "en-US-AriaNeural"),
    ("Bob",   "Yes Alice, I can hear you. My voice should sound deeper and male.", "en-US-GuyNeural"),
    ("Alice", "That is great news. The system is working.", "en-US-AriaNeural"),
    ("Bob",   "Confirmed. Test complete.", "en-US-GuyNeural")
]

async def generate_test_audio():
    print(f"🎤 Starting Voice Test...")
    combined = AudioSegment.empty()

    for speaker, text, voice in SCRIPT:
        print(f"   🔊 Generating {speaker} ({voice})...")
        
        # 1. Generate individual clip
        communicate = edge_tts.Communicate(text, voice)
        temp_file = f"temp_{speaker}.mp3"
        await communicate.save(temp_file)
        
        # 2. Load and append
        segment = AudioSegment.from_mp3(temp_file)
        combined += segment + AudioSegment.silent(duration=500) # Add 0.5s pause
        
        # 3. Cleanup
        os.remove(temp_file)

    # 4. Save final file
    combined.export(OUTPUT_FILE, format="mp3")
    print(f"\n✅ Test Complete! Saved to: {OUTPUT_FILE}")
    print(f"   👉 Run this to listen:  aplay {OUTPUT_FILE}  (or download it)")

if __name__ == "__main__":
    asyncio.run(generate_test_audio())