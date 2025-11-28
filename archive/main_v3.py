import os
import sys
import json
import asyncio
import edge_tts
import time
import random
import re
import email
import datetime
import requests
from urllib.parse import urlparse, urlunparse
from email import policy
from pydub import AudioSegment
from newspaper import Article
from google import genai
from google.genai import types

# --- CONFIGURATION & PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_DIR = os.path.join(DATA_DIR, "inputs")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
MUSIC_DIR = os.path.join(BASE_DIR, "music")

TODAY_STR = datetime.datetime.now().strftime("%d%b")

# NOISE FILTER
IGNORE_TERMS = [
    "unsubscribe", "manage your emails", "view in browser", "privacy policy",
    "facebook.com", "twitter.com", "instagram.com", "linkedin.com", 
    "google.com", "youtube.com", "help center", "contact us", "login", 
    "signup", "preferences", "advertisement"
]

# CHECK API KEY
if not os.environ.get("GOOGLE_API_KEY"):
    print("❌ Error: GOOGLE_API_KEY not found.")
    sys.exit(1)

client = genai.Client()

# VOICE CAST
VOICE_CAST = {
    "Politics": {"Sarah": "en-GB-SoniaNeural", "Mike": "en-US-GuyNeural"},
    "Technology": {"Tasha": "en-AU-NatashaNeural", "Chris": "en-US-ChristopherNeural"},
    "Sport": {"Connor": "en-IE-ConnorNeural", "Ryan": "en-GB-RyanNeural"},
    "Default": {"Alice": "en-GB-RyanNeural", "Bob": "en-US-AriaNeural"}
}

# --- HELPER: AUDIO NORMALIZATION ---
def normalize_volume(sound, target_dBFS=-35.0):
    """Adjusts volume to a specific target level (e.g., -35dB for background)."""
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

# --- HELPER: SAVE/LOAD STATE ---
def save_state(data):
    filepath = os.path.join(CACHE_DIR, "latest_run.json")
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    print(f"💾 State saved to cache.")

def load_state():
    filepath = os.path.join(CACHE_DIR, "latest_run.json")
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return None

# --- STAGE 1: SMART FILE LOADER (With Redirect Resolution) ---
def load_urls(filename):
    filepath = os.path.join(INPUT_DIR, filename)
    if not os.path.exists(filepath):
        if os.path.exists(filename): filepath = filename
        else:
            print(f"❌ Error: File '{filename}' not found.")
            sys.exit(1)

    print(f"📂 Processing: {filepath}")
    raw_urls = []
    
    try:
        if filename.lower().endswith(".eml"):
            print("   -> Detected Email format.")
            with open(filepath, "rb") as f:
                msg = email.message_from_binary_file(f, policy=policy.default)
            body = msg.get_body(preferencelist=('html')).get_content() if msg.get_body(preferencelist=('html')) else msg.get_body(preferencelist=('plain')).get_content()
            raw_urls = re.findall(r'href=["\'](https?://[^"\']+)["\']', body, re.IGNORECASE)

        elif filename.lower().endswith(".html") or "<DT><A HREF=" in open(filepath, "r", errors="ignore").read(1024):
            print("   -> Detected HTML Bookmark format.")
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            raw_urls = re.findall(r'href="(https?://[^"]+)"', content, re.IGNORECASE)

        else:
            print("   -> Detected Text list.")
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                raw_urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    except Exception as e:
        print(f"❌ Error reading file: {e}")
        sys.exit(1)

    # --- SMART DE-DUPLICATION ---
    print(f"   -> Found {len(raw_urls)} raw links. Cleaning & resolving...")
    clean_urls = set()
    final_urls = []
    
    filtered_raw = [u for u in raw_urls if not any(t in u.lower() for t in IGNORE_TERMS)]
    
    for url in filtered_raw:
        try:
            # Resolve Email Tracking Links
            if "link.news" in url or "click/" in url or "bit.ly" in url:
                try:
                    response = requests.head(url, allow_redirects=True, timeout=3)
                    real_url = response.url
                except:
                    real_url = url
            else:
                real_url = url

            # Strip query params to find true duplicates
            parsed = urlparse(real_url)
            clean_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))
            
            if clean_url.strip("/").endswith(".co.uk") or clean_url.strip("/").endswith(".com"):
                continue # Skip homepages

            if clean_url not in clean_urls:
                clean_urls.add(clean_url)
                final_urls.append(real_url)
                # print(f"      + Added: {clean_url[:50]}...")

        except Exception: continue

    print(f"   -> Reduced to {len(final_urls)} unique articles.")
    return final_urls

# --- STAGE 2: THE SCRAPER ---
def fetch_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return f"URL: {url}\nHEADLINE: {article.title}\nTEXT: {article.text[:2500]}..."
    except: return ""

# --- STAGE 3: THE EDITOR (Aggressive Clustering) ---
def ai_cluster_and_summarize(raw_texts):
    print(f"🧠 Analyzing {len(raw_texts)} articles...")
    
    prompt = """
    You are a Senior News Editor. 
    Your goal is to create exactly 3 or 4 'Audio News Briefings' from this list of articles.
    
    CRITICAL RULES:
    1. DO NOT create a separate cluster for every single article. Combine them!
    2. Each cluster MUST contain at least 2-3 articles (unless it's a major breaking headline).
    3. Group loosely related items under broad themes (e.g. 'Domestic News', 'International Events', 'Culture & Lifestyle').
    
    INSTRUCTIONS:
    1. THEMES: Politics, World News, Crime/Courts, Culture/Lifestyle.
    2. EXTRACT: 
       - 'topic': The broad theme name.
       - 'file_slug': A 3-4 word slug for the filename.
       - 'dossier': A combined summary script base for ALL articles in this group.
       - 'source_urls': The list of URLs used.
    
    OUTPUT JSON:
    {
        "clusters": [
            { "topic": "Politics", "file_slug": "Election_Update", "dossier": "...", "source_urls": [...] }
        ]
    }
    
    ARTICLES:
    """ + "\n\n".join(raw_texts)

    for _ in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            return json.loads(response.text)
        except: time.sleep(5)
    return {}

# --- STAGE 4: THE PRODUCER ---
def generate_script(topic, dossier, cast_names):
    host1, host2 = cast_names
    prompt = f"""
    Write a 3-minute podcast script.
    TOPIC: {topic}
    CHARACTERS: {host1} & {host2}.
    OUTPUT JSON: [ {{"speaker": "{host1}", "text": "..."}}, {{"speaker": "{host2}", "text": "..."}} ]
    DOSSIER: {dossier}
    """
    for _ in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            return json.loads(response.text)
        except: time.sleep(5)
    return []

# --- STAGE 5: THE SOUND ENGINEER (With Auto-Leveling) ---
async def produce_audio(topic, script, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    print(f"   🎙️ Recording: {filename}")
    
    cast_dict = VOICE_CAST["Default"]
    for key in VOICE_CAST:
        if key.lower() in topic.lower():
            cast_dict = VOICE_CAST[key]
            break
            
    combined_vocals = AudioSegment.empty()
    for i, line in enumerate(script):
        speaker = line["speaker"]
        voice = cast_dict.get(speaker, list(cast_dict.values())[0])

        communicate = edge_tts.Communicate(line["text"], voice)
        temp_file = os.path.join(CACHE_DIR, f"temp_{i}.mp3")
        await communicate.save(temp_file)
        
        segment = AudioSegment.from_mp3(temp_file)
        combined_vocals += segment + AudioSegment.silent(duration=300)
        os.remove(temp_file)

    selected_track = None
    if os.path.exists(MUSIC_DIR):
        mp3s = [f for f in os.listdir(MUSIC_DIR) if f.endswith(".mp3")]
        if mp3s:
            selected_track = os.path.join(MUSIC_DIR, random.choice(mp3s))

    try:
        if selected_track:
            music = AudioSegment.from_mp3(selected_track)
            
            # SMART FIX: Normalize background music to -35dB
            music = normalize_volume(music, target_dBFS=-35.0)
            
            while len(music) < len(combined_vocals): music += music
            final_mix = music[:len(combined_vocals) + 5000].fade_out(3000).overlay(combined_vocals)
        else:
            final_mix = combined_vocals
        final_mix.export(filepath, format="mp3")
        print(f"   ✅ Saved to outputs/")
    except Exception as e: print(f"   ❌ Mixing failed: {e}")

# --- STAGE 6: INDEX MAP ---
def generate_index_html(clusters_metadata):
    html_filename = os.path.join(OUTPUT_DIR, f"Podcast_Index_{TODAY_STR}.html")
    html_content = f"<html><head><title>{TODAY_STR} Briefing</title></head><body><h1>Daily Briefing {TODAY_STR}</h1>"
    
    for item in clusters_metadata:
        html_content += f"<h2>{item['topic']}</h2><p>🎧 <a href='{item['filename']}'>{item['filename']}</a></p><ul>"
        for url in item['source_urls']: html_content += f"<li><a href='{url}'>{url}</a></li>"
        html_content += "</ul>"
    
    with open(html_filename, "w") as f: f.write(html_content + "</body></html>")
    print(f"📚 Index Map saved to outputs/")

# --- MAIN ---
async def main():
    if len(sys.argv) < 2:
        print("❌ Usage: python3 main_v3.py <filename> OR python3 main_v3.py RESUME")
        return

    mode = sys.argv[1]
    
    if mode == "RESUME":
        print("🔄 Resuming from cache...")
        data = load_state()
        if not data:
            print("❌ No cache found. Run with a file first.")
            return
    else:
        urls = load_urls(mode)
        if not urls: return
        print("🔍 Scraping...")
        texts = [fetch_article_text(u) for u in urls if fetch_article_text(u)]
        data = ai_cluster_and_summarize(texts)
        if data: save_state(data)

    index_metadata = []
    for cluster in data.get("clusters", []):
        topic = cluster["topic"]
        slug = re.sub(r'[^a-zA-Z0-9_]', '', cluster.get("file_slug", "Update"))
        filename = f"{TODAY_STR}_{topic}_{slug}.mp3"
        
        print(f"\n📺 Producing: {topic}")
        
        cast_dict = VOICE_CAST["Default"]
        for key in VOICE_CAST:
            if key.lower() in topic.lower(): cast_dict = VOICE_CAST[key]
        cast_names = list(cast_dict.keys())

        script = generate_script(topic, cluster["dossier"], cast_names)
        if script:
            await produce_audio(topic, script, filename)
            index_metadata.append({"topic": topic, "filename": filename, "source_urls": cluster.get("source_urls", [])})

    if index_metadata: generate_index_html(index_metadata)

if __name__ == "__main__":
    asyncio.run(main())