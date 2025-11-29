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
import shutil
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

# Date Strings
TODAY_STR = datetime.datetime.now().strftime("%d%b")
TODAY_OUTPUT_DIR = os.path.join(OUTPUT_DIR, TODAY_STR)
TODAY_INPUT_ARCHIVE = os.path.join(INPUT_DIR, "archive", TODAY_STR)

# Storage Limit (MB)
STORAGE_LIMIT_MB = 200

# NOISE FILTER
IGNORE_TERMS = [
    "unsubscribe", "manage your emails", "view in browser", "privacy policy",
    "facebook.com", "twitter.com", "instagram.com", "linkedin.com", 
    "google.com", "youtube.com", "help center", "contact us", "login", 
    "signup", "preferences", "advertisement", "spotify.com", "apple.com"
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
    "Default": {"Alice": "en-US-AriaNeural", "Bob": "en-GB-RyanNeural"} 
}

# --- HELPER: AUDIO NORMALIZATION ---
def normalize_volume(sound, target_dBFS=-35.0):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

# --- HELPER: STORAGE MANAGER ---
def check_and_clean_storage():
    """Checks total size of DATA_DIR. If > 200MB, prompts to delete oldest daily folders."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(DATA_DIR):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    
    total_mb = total_size / (1024 * 1024)
    print(f"\n💾 Storage Status: {total_mb:.2f} MB / {STORAGE_LIMIT_MB} MB")
    
    if total_mb > STORAGE_LIMIT_MB:
        print("⚠️  Limit Exceeded! Looking for old files...")
        day_folders = []
        if os.path.exists(OUTPUT_DIR):
            for d in os.listdir(OUTPUT_DIR):
                path = os.path.join(OUTPUT_DIR, d)
                if os.path.isdir(path): day_folders.append(path)
        
        archive_root = os.path.join(INPUT_DIR, "archive")
        if os.path.exists(archive_root):
            for d in os.listdir(archive_root):
                path = os.path.join(archive_root, d)
                if os.path.isdir(path): day_folders.append(path)

        day_folders.sort(key=os.path.getmtime)
        
        if day_folders:
            oldest = day_folders[0]
            print(f"🗑️  Oldest folder found: {oldest}")
            # Auto-delete check
            if len(sys.argv) > 1 and sys.argv[1] == "AUTO":
                 try: shutil.rmtree(oldest)
                 except: pass
                 print("   ✅ Deleted (Auto-Clean).")
            else:
                choice = input("   >> Delete this folder to free space? [y/N]: ")
                if choice.lower() == 'y':
                    try: shutil.rmtree(oldest)
                    except: pass
                    print("   ✅ Deleted.")

# --- HELPER: SAVE/LOAD STATE ---
def save_state(data):
    if not os.path.exists(CACHE_DIR): os.makedirs(CACHE_DIR)
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

# --- STAGE 1: THE AGGREGATOR ---
def load_all_inputs(target_path):
    files_to_process = []
    if os.path.isdir(target_path):
        print(f"📂 Scanning Directory: {target_path}")
        for root, dirs, files in os.walk(target_path):
            if "archive" in root: continue 
            for file in files:
                if not file.startswith(".") and not file.endswith(".ini"):
                    files_to_process.append(os.path.join(root, file))
    elif os.path.exists(target_path):
        files_to_process.append(target_path)

    master_raw_urls = []
    processed_files = [] 

    print(f"   -> Processing {len(files_to_process)} input files...")

    for filepath in files_to_process:
        try:
            filename = os.path.basename(filepath)
            content_urls = []
            
            if filename.lower().endswith(".eml"):
                with open(filepath, "rb") as f:
                    msg = email.message_from_binary_file(f, policy=policy.default)
                body = None
                if msg.get_body(preferencelist=('html')):
                    body = msg.get_body(preferencelist=('html')).get_content()
                elif msg.get_body(preferencelist=('plain')):
                    body = msg.get_body(preferencelist=('plain')).get_content()
                if body:
                    content_urls = re.findall(r'href=["\'](https?://[^"\']+)["\']', body, re.IGNORECASE)

            elif filename.lower().endswith((".html", ".mhtml")):
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                content_urls = re.findall(r'href="(https?://[^"]+)"', content, re.IGNORECASE)

            else:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content_urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            
            master_raw_urls.extend(content_urls)
            processed_files.append(filepath)

        except Exception as e:
            print(f"   ⚠️ Could not read {filename}: {e}")

    # Deduplication
    print(f"   -> Found {len(master_raw_urls)} total links. Cleaning...")
    clean_urls_set = set()
    final_urls = []
    
    filtered_raw = [u for u in master_raw_urls if not any(t in u.lower() for t in IGNORE_TERMS)]
    
    for url in filtered_raw:
        try:
            parsed = urlparse(url)
            clean_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))
            if clean_url.strip("/").count("/") < 3: continue
            if clean_url not in clean_urls_set:
                clean_urls_set.add(clean_url)
                final_urls.append(url)
        except: continue

    print(f"   -> Consolidated into {len(final_urls)} unique articles.")
    return final_urls, processed_files

# --- STAGE 2: THE SCRAPER (LIGHTWEIGHT METADATA) ---
def fetch_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        headline = article.title
        # snippet increased to 500 chars as requested for Stage 2
        snippet = f"{headline}. {article.text[:500]}..."
        full_text = article.text[:2500].replace("\n", " ")
        
        return {
            "url": url,
            "headline": headline,
            "snippet": snippet,
            "full_text": full_text
        }
    except: return None

# --- STAGE 3: THE EDITOR (GEN 7: MAP-REDUCE) ---
def ai_cluster_and_summarize(article_objects):
    print(f"🧠 Analyzing {len(article_objects)} articles...")
    
    # Map ID -> Article
    indexed_articles = {i: obj for i, obj in enumerate(article_objects)}
    
    # Safety: Unlock Blockers
    safety_settings = [
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    ]

    # ----------------------------------------
    # PHASE A: THE SORTER (Titles -> Broad Themes)
    # ----------------------------------------
    print("   -> Phase A: Sorting into broad themes (Titles only)...")
    
    # Only send ID and Headline to keep it tiny
    sort_input = ""
    for i, obj in indexed_articles.items():
        sort_input += f"ID [{i}]: {obj['headline']}\n"
        
    sort_prompt = """
    You are a News Sorter. Assign each Article ID to ONE of these broad buckets:
    [World, Politics, Tech, Sport, Culture, Economy, Science].
    
    OUTPUT JSON FORMAT:
    {
        "World": [1, 5, 20],
        "Politics": [2, 8],
        "Tech": [3]
    }
    
    HEADLINES:
    """ + sort_input

    sorted_buckets = {}
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=sort_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                safety_settings=safety_settings
            )
        )
        if response.text:
             clean_text = response.text.replace("```json", "").replace("```", "").strip()
             sorted_buckets = json.loads(clean_text)
    except Exception as e:
        print(f"   ⚠️ Phase A Failed: {e}. Falling back to chunked processing.")
        # Fallback: Just process in chunks of 50
        sorted_buckets = {}
        ids = list(indexed_articles.keys())
        for i in range(0, len(ids), 50):
            sorted_buckets[f"Batch_{i}"] = ids[i:i+50]

    # ----------------------------------------
    # PHASE B: THE PLANNER (Snippets -> Clusters)
    # ----------------------------------------
    final_clusters = {"clusters": []}
    
    for theme, ids in sorted_buckets.items():
        # Clean IDs
        valid_ids = []
        if isinstance(ids, list):
            for x in ids:
                try: valid_ids.append(int(x))
                except: pass
        
        if not valid_ids: continue
        
        print(f"   -> Phase B: Clustering '{theme}' ({len(valid_ids)} articles)...")
        
        # Prepare Snippets for this specific bucket
        bucket_input = ""
        for i in valid_ids:
            if i in indexed_articles:
                bucket_input += f"ID [{i}]: {indexed_articles[i]['snippet']}\n\n"
        
        bucket_prompt = f"""
        You are the Editor for the '{theme}' desk.
        Group these {len(valid_ids)} snippets into 1-3 tight Podcast Segments (Episodes).
        
        RULES:
        1. Group by ID.
        2. MERGE duplicates/related stories.
        3. IGNORE irrelevant/tiny stories.
        
        OUTPUT JSON:
        {{
            "clusters": [
                {{ "topic": "{theme}: Subtopic", "file_slug": "{theme}_Slug", "article_ids": [1, 2] }}
            ]
        }}
        
        SNIPPETS:
        {bucket_input}
        """
        
        try:
            # We allow higher tokens here because we are processing a subset
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=bucket_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    safety_settings=safety_settings
                )
            )
            
            if response.text:
                clean_text = response.text.replace("```json", "").replace("```", "").strip()
                bucket_data = json.loads(clean_text)
                
                # Standardize output
                clusters = []
                if isinstance(bucket_data, list): clusters = bucket_data
                elif isinstance(bucket_data, dict): 
                    # Try to find the list
                    for k,v in bucket_data.items():
                        if isinstance(v, list): 
                            clusters = v
                            break
                
                # Re-hydrate with FULL TEXT for Stage 4
                for c in clusters:
                    full_dossier = ""
                    source_urls = []
                    
                    # Extract IDs flexibly
                    c_ids = c.get("article_ids") or c.get("ids") or []
                    for art_id in c_ids:
                        try: art_id = int(art_id)
                        except: continue
                        
                        if art_id in indexed_articles:
                            obj = indexed_articles[art_id]
                            full_dossier += f"HEADLINE: {obj['headline']}\nFACTS: {obj['full_text']}\n\n"
                            source_urls.append(obj['url'])
                    
                    if full_dossier:
                        final_clusters["clusters"].append({
                            "topic": c.get("topic", theme),
                            "file_slug": c.get("file_slug", "News"),
                            "dossier": full_dossier,
                            "source_urls": source_urls
                        })

        except Exception as e:
            print(f"      ⚠️ Failed to cluster {theme}: {e}")
            
    return final_clusters

# --- STAGE 4: THE PRODUCER ---
def generate_script(topic, dossier, cast_names):
    host1, host2 = cast_names
    prompt = f"""
    Write a detailed, in-depth 10-minute podcast script (approx 1,500 words).
    TOPIC: {topic}
    HOSTS: {host1} & {host2}.
    
    STYLE: Professional, conversational, deep-dive analysis.
    CONTENT: Use the provided DOSSIER.
    
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

# --- STAGE 5: THE SOUND ENGINEER (SMART DJ) ---
async def produce_audio(topic, script, filename):
    if not os.path.exists(TODAY_OUTPUT_DIR): os.makedirs(TODAY_OUTPUT_DIR)
    
    filepath = os.path.join(TODAY_OUTPUT_DIR, filename)
    print(f"   🎙️  Recording: {filename}")
    print(f"   📝 Script Length: {len(script)} lines")
    
    cast_dict = VOICE_CAST["Default"]
    for key in VOICE_CAST:
        if key.lower() in topic.lower():
            cast_dict = VOICE_CAST[key]
            break
            
    combined_vocals = AudioSegment.empty()
    
    for i, line in enumerate(script):
        speaker = line.get("speaker", "Host")
        text = line.get("text", "")
        
        print(f"      -> Line {i+1}/{len(script)} ({speaker}): {text[:30]}...") 

        voice = cast_dict.get(speaker, list(cast_dict.values())[0])

        try:
            communicate = edge_tts.Communicate(text, voice)
            temp_file = os.path.join(CACHE_DIR, f"temp_{i}.mp3")
            await communicate.save(temp_file)
            
            segment = AudioSegment.from_mp3(temp_file)
            combined_vocals += segment + AudioSegment.silent(duration=400)
            if os.path.exists(temp_file): os.remove(temp_file)
            
        except Exception as e:
            print(f"      ❌ Error on line {i+1}: {e}")
            continue

    print("   🎹 Mixing background tracks (Smart DJ Mode)...")
    
    music_mix = AudioSegment.empty()
    available_tracks = []
    if os.path.exists(MUSIC_DIR):
        available_tracks = [f for f in os.listdir(MUSIC_DIR) if f.endswith(".mp3")]

    if available_tracks:
        target_duration_ms = len(combined_vocals) + 10000
        while len(music_mix) < target_duration_ms:
            random_track_name = random.choice(available_tracks)
            track_path = os.path.join(MUSIC_DIR, random_track_name)
            try:
                track = AudioSegment.from_mp3(track_path)
                track = normalize_volume(track, target_dBFS=-35.0)
                if len(music_mix) == 0:
                    music_mix = track
                else:
                    cf_duration = min(len(music_mix), len(track), 3000)
                    music_mix = music_mix.append(track, crossfade=cf_duration)
            except Exception as e:
                if len(music_mix) == 0: break 
        
        if len(music_mix) > target_duration_ms:
            music_mix = music_mix[:target_duration_ms]
            
        final_mix = music_mix.fade_out(3000).overlay(combined_vocals)
    else:
        final_mix = combined_vocals
            
    try:
        final_mix.export(filepath, format="mp3")
        print(f"   ✅ Published: {filename}")
        return filename
    except Exception as e: 
        print(f"   ❌ Audio mix failed: {e}")
        return None

# --- STAGE 6: INDEX ---
def generate_index_html(clusters_metadata):
    if not os.path.exists(TODAY_OUTPUT_DIR): os.makedirs(TODAY_OUTPUT_DIR)
    html_filename = os.path.join(TODAY_OUTPUT_DIR, f"Podcast_Index_{TODAY_STR}.html")
    html_content = f"<html><head><title>Briefing {TODAY_STR}</title></head><body><h1>Daily Briefing {TODAY_STR}</h1>"
    for item in clusters_metadata:
        html_content += f"<h2>{item['topic']}</h2><audio controls src='{item['filename']}'></audio><ul>"
        for url in item['source_urls']: html_content += f"<li><a href='{url}'>{url}</a></li>"
        html_content += "</ul>"
    with open(html_filename, "w") as f: f.write(html_content + "</body></html>")
    print(f"📚 Index generated: {html_filename}")

# --- ARCHIVER ---
def archive_inputs(processed_files):
    if not processed_files: return
    if not os.path.exists(TODAY_INPUT_ARCHIVE): os.makedirs(TODAY_INPUT_ARCHIVE)
    print(f"📦 Archiving {len(processed_files)} input files...")
    for file_path in processed_files:
        try: shutil.move(file_path, TODAY_INPUT_ARCHIVE)
        except: pass

# --- MAIN ENTRY POINT ---
async def main():
    target = INPUT_DIR
    resume_mode = False
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "RESUME":
            resume_mode = True
        else:
            target = sys.argv[1]

    check_and_clean_storage()

    processed_files = []
    data = None

    if resume_mode:
        print("🔄 Resuming from cache...")
        data = load_state()
    else:
        urls, processed_files = load_all_inputs(target)
        if not urls: 
            print("❌ No valid URLs found to process.")
            return

        print("🔍 Scraping articles...")
        article_objects = []
        for u in urls:
            obj = fetch_article_text(u)
            if obj: article_objects.append(obj)
        
        if article_objects:
            data = ai_cluster_and_summarize(article_objects)
            if data: save_state(data)
    
    if not data or not data.get("clusters"):
        print("❌ No news clusters generated. Exiting.")
        return

    index_metadata = []
    for cluster in data.get("clusters", []):
        topic = cluster["topic"]
        slug = re.sub(r'[^\w]', '', cluster.get("file_slug", "Update"))[:20]
        safe_topic = re.sub(r'[^\w]', '_', topic)[:20]
        filename = f"{TODAY_STR}_{safe_topic}_{slug}.mp3"
        
        print(f"\n📺 Studio: Producing '{topic}'...")
        cast_dict = VOICE_CAST["Default"]
        for key in VOICE_CAST:
            if key.lower() in topic.lower(): cast_dict = VOICE_CAST[key]
        cast_names = list(cast_dict.keys())

        script = generate_script(topic, cluster["dossier"], cast_names)
        if script:
            final_file = await produce_audio(topic, script, filename)
            if final_file:
                index_metadata.append({"topic": topic, "filename": final_file, "source_urls": cluster.get("source_urls", [])})

    if index_metadata: 
        generate_index_html(index_metadata)
        if not resume_mode: archive_inputs(processed_files)

if __name__ == "__main__":
    asyncio.run(main())