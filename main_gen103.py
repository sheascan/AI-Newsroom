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
import hashlib
import requests
import csv
import difflib
from urllib.parse import urlparse, urlunparse
from email import policy
from pydub import AudioSegment
from newspaper import Article
from google import genai
from google.genai import types

# --- VERSION TRACKER ---
VERSION_ID = "Gen 102 (JSON Stabilizer & Filter Update)"

# --- OPTIONAL IMPORTS ---
try:
    from mutagen.mp3 import MP3
    from mutagen.id3 import ID3, APIC, TIT2, TPE1, TALB, error
    HAS_MUTAGEN = True
except ImportError:
    HAS_MUTAGEN = False
    print("⚠️  'mutagen' library not found. Cover art feature disabled.")

try:
    from gtts import gTTS
    HAS_GTTS = True
except ImportError:
    HAS_GTTS = False
    print("⚠️  'gTTS' library not found. Backup audio engine disabled. (pip install gTTS)")

# --- CONFIGURATION & PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_DIR = os.path.join(DATA_DIR, "inputs")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
LIBRARY_DIR = os.path.join(CACHE_DIR, "library")
SCRIPT_DIR = os.path.join(CACHE_DIR, "scripts")
MUSIC_DIR = os.path.join(BASE_DIR, "music")
MUSIC_STATS_FILE = os.path.join(DATA_DIR, "music_stats.csv")
CHECKPOINT_FILE = os.path.join(CACHE_DIR, "phase_a_checkpoint.json")

# Date Strings
TODAY_STR = datetime.datetime.now().strftime("%d%b")
TODAY_OUTPUT_DIR = os.path.join(OUTPUT_DIR, TODAY_STR)
TODAY_INPUT_ARCHIVE = os.path.join(INPUT_DIR, "archive", TODAY_STR)

# Storage Limit (MB)
STORAGE_LIMIT_MB = 1024

# --- FILTERING CONFIG ---
IGNORE_URL_TERMS = [
    "unsubscribe", "manage your emails", "view in browser", "privacy policy", 
    "terms of service", "manage_preferences", "opt_out", "login", "signup", "signin",
    "facebook.com", "twitter.com", "instagram.com", "linkedin.com", "youtube.com",
    "google.com", "apple.com", "spotify.com", "tiktok.com",
    "help center", "contact us", "advertisement", "click here",
    "arxiv.org",
]

# REMOVED: "sale", "off", "discount" as requested
IGNORE_HEADLINE_TERMS = [
    "webinar", "register now", "last chance", 
    "giveaway", "promo", "deal of the day", "subscribe", "newsletter",
    "metro sport", "metro uk (@metro.co.uk)", "instagram",
    "make your day", "metro uk newsletters", 
    "terms and conditions", "cookies policy", "terms of use", "preferences",
    "ai & tech newsletter", "courses on chatgpt", "techpssdec2", 
    "workos", "work with us",
    "privacy policy", "terms & conditions",
    "sign up for", "email newsletters", "ad choices",
    "crossword", "spelling bee", "connections — the", "strands: uncover words",
    "minichess", "sudoku", "wordle",
    "the guardian view on", "the long read", "today in focus",
    "new york times", "the new york",
    "the irish times",
    "one ai to", "walk mate", "visualize knowledge", "your kitchen assistant",
    "enable ai", "proofly", "free guide to",
    "threads, say more", "techpresso", "beanvest quality investing", "resurf"
]

# CHECK API KEY
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("❌ Error: GOOGLE_API_KEY not found.")
    sys.exit(1)

client = genai.Client()

# MODEL CONFIGURATION
MODEL_NAME = "gemini-2.5-flash-lite" 

BATCH_SIZE = 25     
SLEEP_TIME = 20     

# VOICE CAST
VOICE_CAST = {
    "Politics": {"Sarah": "en-GB-SoniaNeural", "Mike": "en-US-GuyNeural"},
    "Technology": {"Tasha": "en-AU-NatashaNeural", "Chris": "en-US-ChristopherNeural"},
    "Sport": {"Connor": "en-IE-ConnorNeural", "Ryan": "en-GB-RyanNeural"},
    "Default": {"Alice": "en-US-AriaNeural", "Bob": "en-GB-RyanNeural"} 
}

# --- HELPER: API RATE LIMIT WRAPPER ---
def generate_with_retry(model_name, contents, config, retries=5):
    """Wraps Gemini calls with Exponential Backoff."""
    base_wait = 30 
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config
            )
            return response
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait_time = base_wait * (2 ** attempt) 
                print(f"      ⚠️  Quota Hit (429). Attempt {attempt+1}/{retries}")
                if attempt == 0: 
                    print(f"      🔎  Diagnostic: {error_str[:200]}...") 
                print(f"      ⏳  Pausing for {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"      ⚠️ API Error: {e}. Retrying in 10s...")
                time.sleep(10)
    return None

# --- HELPER: AUDIO NORMALIZATION ---
def normalize_volume(sound, target_dBFS=-35.0):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

# --- HELPER: MUSIC TRACKER ---
def update_music_stats(track_filename):
    stats = {}
    if os.path.exists(MUSIC_STATS_FILE):
        try:
            with open(MUSIC_STATS_FILE, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    stats[row['Filename']] = {'Count': int(row['Count']), 'Last_Used': row['Last_Used']}
        except: pass

    if track_filename in stats:
        stats[track_filename]['Count'] += 1
        stats[track_filename]['Last_Used'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    else:
        stats[track_filename] = {'Count': 1, 'Last_Used': datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
        
    try:
        with open(MUSIC_STATS_FILE, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['Filename', 'Count', 'Last_Used']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for name, data in stats.items():
                writer.writerow({'Filename': name, 'Count': data['Count'], 'Last_Used': data['Last_Used']})
    except: pass

# --- HELPER: BRANDING MANAGER ---
def attach_cover_art(mp3_path, topic_title):
    if not HAS_MUTAGEN: return
    cover_path = os.path.join(BASE_DIR, "cover.jpg")
    if not os.path.exists(cover_path): return 

    try:
        audio = MP3(mp3_path, ID3=ID3)
        try: audio.add_tags()
        except error: pass

        with open(cover_path, 'rb') as albumart:
            audio.tags.add(APIC(encoding=3, mime='image/jpeg', type=3, desc=u'Cover', data=albumart.read()))
        
        audio.tags.add(TIT2(encoding=3, text=topic_title))
        audio.tags.add(TPE1(encoding=3, text="AI News Anchor"))
        audio.tags.add(TALB(encoding=3, text=f"Daily Briefing {TODAY_STR}"))
        audio.save()
        print(f"      🖼️  Cover art attached.")
    except Exception as e:
        print(f"      ⚠️ Failed to attach cover: {e}")

# --- HELPER: STORAGE MANAGER ---
def check_and_clean_storage():
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(DATA_DIR):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp): total_size += os.path.getsize(fp)
    
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
    with open(filepath, "w") as f: json.dump(data, f, indent=4)
    print(f"💾 State saved to cache.")

def load_state():
    filepath = os.path.join(CACHE_DIR, "latest_run.json")
    if os.path.exists(filepath):
        with open(filepath, "r") as f: return json.load(f)
    return None

# --- HELPER: GENERATE AUDIT REPORT ---
def generate_audit_report(all_articles_metadata):
    if not os.path.exists(TODAY_OUTPUT_DIR): os.makedirs(TODAY_OUTPUT_DIR)
    report_path = os.path.join(TODAY_OUTPUT_DIR, f"Filtering_Report_{TODAY_STR}.html")
    
    kept_count = sum(1 for a in all_articles_metadata if a['status'] == 'kept')
    junk_count = sum(1 for a in all_articles_metadata if a['status'] == 'junk')
    dupe_count = sum(1 for a in all_articles_metadata if a['status'] == 'duplicate')
    
    html = f"""<html><head><title>Podcast Filtering Report</title>
    <style>
        body {{ font-family: sans-serif; padding: 20px; }}
        .summary {{ margin-bottom: 20px; padding: 10px; background: #eee; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        tr.kept {{ background-color: #d4edda; }}
        tr.junk {{ background-color: #f8d7da; }}
        tr.duplicate {{ background-color: #fff3cd; }}
    </style></head><body>
    <h1>Filtering Report: {TODAY_STR}</h1>
    <div class='summary'>
        <strong>Kept:</strong> {kept_count} | <strong>Junk:</strong> {junk_count} | <strong>Duplicates:</strong> {dupe_count}
    </div>
    <table>
        <tr><th>Status</th><th>Reason</th><th>Headline</th><th>URL</th></tr>
    """
    
    for item in all_articles_metadata:
        row_class = item['status']
        html += f"<tr class='{row_class}'><td>{item['status'].upper()}</td><td>{item.get('reject_reason', '-')}</td><td>{item.get('headline', 'N/A')}</td><td><a href='{item['url']}'>Link</a></td></tr>"
    
    html += "</table></body></html>"
    
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(html)
    print(f"   📊 Audit Report generated: {report_path}")

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
                with open(filepath, "rb") as f: msg = email.message_from_binary_file(f, policy=policy.default)
                body = None
                if msg.get_body(preferencelist=('html')): body = msg.get_body(preferencelist=('html')).get_content()
                elif msg.get_body(preferencelist=('plain')): body = msg.get_body(preferencelist=('plain')).get_content()
                if body: content_urls = re.findall(r'href=["\'](https?://[^"\']+)["\']', body, re.IGNORECASE)
            elif filename.lower().endswith((".html", ".mhtml")):
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f: content = f.read()
                content_urls = re.findall(r'href="(https?://[^"]+)"', content, re.IGNORECASE)
            else:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content_urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            master_raw_urls.extend(content_urls)
            processed_files.append(filepath)
        except Exception as e: print(f"   ⚠️ Could not read {filename}: {e}")

    print(f"   -> Found {len(master_raw_urls)} total links. Cleaning...")
    clean_urls_set = set()
    final_urls = []
    
    # --- LEVEL 1: URL FILTERING ---
    for url in master_raw_urls:
        try:
            u_lower = url.lower()
            if any(t in u_lower for t in IGNORE_URL_TERMS): continue
            
            parsed = urlparse(url)
            clean_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))
            if clean_url.strip("/").count("/") < 3: continue
            
            if clean_url not in clean_urls_set:
                clean_urls_set.add(clean_url)
                final_urls.append(clean_url) 
        except: continue

    print(f"   -> Consolidated into {len(final_urls)} unique articles.")
    return final_urls, processed_files

# --- STAGE 2: THE SCRAPER & REFINERY ---
def fetch_article_text(url):
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
    if not os.path.exists(LIBRARY_DIR): os.makedirs(LIBRARY_DIR)
    cache_path = os.path.join(LIBRARY_DIR, f"{url_hash}.json")

    # 1. Check Library (Cache)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except: pass

    # 2. Scrape Live
    try:
        article = Article(url)
        article.download()
        article.parse()
        headline = article.title
        
        # --- LEVEL 2: HEADLINE FILTERING (Initial Pass) ---
        is_junk = False
        reject_reason = ""
        
        for term in IGNORE_HEADLINE_TERMS:
            if term.lower() in headline.lower():
                is_junk = True
                reject_reason = f"Keyword: {term}"
                break

        snippet = f"{headline}. {article.text[:300]}..."
        full_text = article.text[:2500].replace("\n", " ")
        
        data = {
            "url": url, 
            "headline": headline, 
            "snippet": snippet, 
            "full_text": full_text,
            "status": "junk" if is_junk else "scraped",
            "reject_reason": reject_reason
        }

        # 3. Save to Library
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
            
        return data
    except: return None

# --- STAGE 3: THE EDITOR ---
def ai_cluster_and_summarize(article_objects):
    
    # --- LEVEL 3: DEDUPLICATION, RE-FILTERING & REPORTING ---
    print("   -> Running Refining Pass (Removing Duplicates & Re-Checking Junk)...")
    
    all_metadata_for_report = []
    unique_articles = []
    seen_headlines = []
    
    for obj in article_objects:
        if not obj: continue
        
        status = obj.get('status', 'scraped')
        reason = obj.get('reject_reason', '')
        headline_lower = obj['headline'].lower()

        # --- RE-CHECK JUNK (Fix for Cached Items) ---
        if status != 'junk':
            for term in IGNORE_HEADLINE_TERMS:
                if term.lower() in headline_lower:
                    status = 'junk'
                    reason = f"Keyword (Post-Cache): {term}"
                    print(f"      🗑️  Junk Headline (Found in Cache): {obj['headline'][:30]}...")
                    break

        if status == 'junk':
            obj['status'] = status
            obj['reject_reason'] = reason
            all_metadata_for_report.append(obj)
            continue

        # --- DEDUPLICATION ---
        is_dupe = False
        for seen in seen_headlines:
            ratio = difflib.SequenceMatcher(None, headline_lower, seen).ratio()
            if ratio > 0.8:
                is_dupe = True
                status = 'duplicate'
                reason = f"Similar to: {seen[:30]}..."
                print(f"      🗑️  Duplicate: {obj['headline'][:30]}...")
                break
        
        if not is_dupe:
            seen_headlines.append(headline_lower)
            status = 'kept'
            unique_articles.append(obj)
        
        # Final Status Update
        obj['status'] = status
        obj['reject_reason'] = reason
        all_metadata_for_report.append(obj)

    # --- GENERATE REPORT ---
    generate_audit_report(all_metadata_for_report)
            
    print(f"🧠 Analyzing {len(unique_articles)} unique articles (was {len(article_objects)})...")
    indexed_articles = {i: obj for i, obj in enumerate(unique_articles)}
    
    safety_settings = [
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    ]

    # --- PHASE A: THE SORTER ---
    master_buckets = {}
    
    if os.path.exists(CHECKPOINT_FILE):
        print("   📂 Found Phase A Checkpoint! Loading sorted themes...")
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                master_buckets = json.load(f)
        except Exception as e:
            print(f"   ⚠️ Checkpoint corrupt ({e}). Restarting sort.")
            master_buckets = {}

    if not master_buckets:
        print(f"   -> Phase A: Sorting into broad themes ({BATCH_SIZE} items/batch)...")
        ids = list(indexed_articles.keys())
        
        for i in range(0, len(ids), BATCH_SIZE):
            batch_ids = ids[i:i+BATCH_SIZE]
            print(f"      Processing Batch {i//BATCH_SIZE + 1} ({len(batch_ids)} items)...")
            
            sort_input = ""
            for art_id in batch_ids:
                sort_input += f"ID [{art_id}]: {indexed_articles[art_id]['headline']}\n"
                
            sort_prompt = """
            You are a News Sorter. Assign each Article ID to ONE of these broad buckets:
            [World, Politics, Tech, Sport, Culture, Economy, Science].
            OUTPUT JSON: {"World": [1, 5], "Politics": [2]}
            HEADLINES:
            """ + sort_input
            
            response = generate_with_retry(
                MODEL_NAME, 
                sort_prompt, 
                types.GenerateContentConfig(response_mime_type="application/json", safety_settings=safety_settings)
            )
            
            if response and response.text:
                try:
                    clean_text = response.text.replace("```json", "").replace("```", "").strip()
                    batch_buckets = json.loads(clean_text)
                    for category, id_list in batch_buckets.items():
                        if category not in master_buckets:
                            master_buckets[category] = []
                        if isinstance(id_list, list):
                            master_buckets[category].extend(id_list)
                except Exception as e:
                    print(f"      ⚠️ Failed to sort batch: {e}")
            
            # 20s Sleep
            time.sleep(SLEEP_TIME)
        
        if master_buckets:
            with open(CHECKPOINT_FILE, 'w') as f:
                json.dump(master_buckets, f)
            print("   💾 Phase A Checkpoint Saved.")

    # --- PHASE B: THE PLANNER ---
    final_clusters = {"clusters": []}
    
    if not master_buckets:
        master_buckets = {"General News": list(indexed_articles.keys())}

    for theme, ids in master_buckets.items():
        valid_ids = []
        if isinstance(ids, list):
            for x in ids:
                try: valid_ids.append(int(x))
                except: pass
        
        valid_ids = list(set(valid_ids))
        if not valid_ids: continue
        
        print(f"   -> Phase B: Clustering '{theme}' ({len(valid_ids)} articles)...")
        
        if len(valid_ids) <= 10:
            instruction_mode = "Create a SINGLE comprehensive episode. Title it 'News Digest'. Do NOT split these stories."
        else:
            instruction_mode = "Group into 2-3 episodes. Merge related stories aggressively."

        bucket_input = ""
        for i in valid_ids:
            if i in indexed_articles: bucket_input += f"ID [{i}]: {indexed_articles[i]['snippet']}\n\n"
        
        bucket_prompt = f"""
        You are the Editor for the '{theme}' desk.
        {instruction_mode}
        
        OUTPUT JSON: {{ "clusters": [ {{ "topic": "{theme}: Title", "file_slug": "{theme}_Slug", "article_ids": [1, 2, 5] }} ] }}
        SNIPPETS:
        {bucket_input}
        """
        
        response = generate_with_retry(
            MODEL_NAME, 
            bucket_prompt, 
            types.GenerateContentConfig(response_mime_type="application/json", safety_settings=safety_settings)
        )
        
        if response and response.text:
            try:
                clean_text = response.text.replace("```json", "").replace("```", "").strip()
                bucket_data = json.loads(clean_text)
                clusters = []
                if isinstance(bucket_data, list): clusters = bucket_data
                elif isinstance(bucket_data, dict): 
                    for k,v in bucket_data.items():
                        if isinstance(v, list): clusters = v; break
                
                for c in clusters:
                    full_dossier = ""
                    source_urls = []
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
                time.sleep(SLEEP_TIME) 
            except Exception as e:
                print(f"      ⚠️ Failed to parse cluster {theme}: {e}")
            
    if final_clusters["clusters"] and os.path.exists(CHECKPOINT_FILE):
        try: os.remove(CHECKPOINT_FILE)
        except: pass
        print("   ✅ Phase B Complete. Checkpoint cleared.")

    return final_clusters

# --- STAGE 4: THE PRODUCER ---
def generate_script(topic, dossier, cast_names, source_count):
    # --- VAULT CHECK (Gen 102) ---
    safe_slug = re.sub(r'[^\w]+', '_', topic).strip('_')[:50]
    script_hash = hashlib.md5(dossier.encode('utf-8')).hexdigest()[:10]
    vault_filename = f"{TODAY_STR}_{safe_slug}_{script_hash}.json"
    if not os.path.exists(SCRIPT_DIR): os.makedirs(SCRIPT_DIR)
    vault_path = os.path.join(SCRIPT_DIR, vault_filename)

    # 1. Check Vault
    if os.path.exists(vault_path):
        print("      📜 Found Script in Vault! Skipping Generation.")
        try:
            with open(vault_path, 'r') as f: return json.load(f)
        except: pass

    # 2. Generate New
    host1, host2 = cast_names
    
    if source_count <= 2:
        duration_desc = "3-minute 'Flash Update'"
        word_count = "400 words"
        style_note = "Fast-paced, punchy. No filler."
    elif source_count <= 5:
        duration_desc = "7-minute 'Standard Briefing'"
        word_count = "1000 words"
        style_note = "Balanced, conversational."
    else:
        duration_desc = "12-minute 'Deep Dive'"
        word_count = "1600 words"
        style_note = "In-depth analysis, weaving stories together."

    print(f"      📝 Mode: {duration_desc} ({source_count} articles)")

    prompt = f"""
    Write a {duration_desc} podcast script (approx {word_count}).
    TOPIC: {topic}
    HOSTS: {host1} & {host2}.
    STYLE: {style_note}
    INSTRUCTIONS: Use the provided DOSSIER. Do NOT make up facts.
    OUTPUT JSON: [ {{"speaker": "{host1}", "text": "..."}}, {{"speaker": "{host2}", "text": "..."}} ]
    DOSSIER: {dossier}
    """
    
    response = generate_with_retry(
        MODEL_NAME, 
        prompt, 
        types.GenerateContentConfig(response_mime_type="application/json")
    )
    
    if response and response.text:
        time.sleep(SLEEP_TIME) 
        
        try:
            # --- JSON STABILIZER (Gen 102) ---
            # 1. Remove Markdown wrapper
            clean_text = response.text.replace("```json", "").replace("```", "").strip()
            
            # 2. Surgically extract the list [...] to ignore trailing commentary
            start_idx = clean_text.find('[')
            end_idx = clean_text.rfind(']')
            if start_idx != -1 and end_idx != -1:
                clean_text = clean_text[start_idx : end_idx + 1]
            
            script_data = json.loads(clean_text)
            
            # 3. Save to Vault
            with open(vault_path, 'w') as f:
                json.dump(script_data, f, indent=4)
                
            return script_data
        except Exception as e:
            print(f"      ⚠️ JSON Error: {e}. Raw text len: {len(response.text)}")
            return []

    return []

# --- STAGE 5: THE SOUND ENGINEER ---
async def produce_audio(topic, script, filename):
    if not os.path.exists(TODAY_OUTPUT_DIR): os.makedirs(TODAY_OUTPUT_DIR)
    filepath = os.path.join(TODAY_OUTPUT_DIR, filename)
    print(f"   🎙️  Recording: {filename}")
    print(f"   📝 Script Length: {len(script)} lines")
    
    cast_dict = VOICE_CAST["Default"]
    for key in VOICE_CAST:
        if key.lower() in topic.lower(): cast_dict = VOICE_CAST[key]; break
    
    # --- AUDIO ENGINE STATE ---
    # If set to True, we skip EdgeTTS entirely for the rest of this session
    fallback_mode_active = False 

    combined_vocals = AudioSegment.empty()
    for i, line in enumerate(script):
        speaker = line.get("speaker", "Host")
        text = line.get("text", "")
        print(f"      -> Line {i+1}/{len(script)} ({speaker}): {text[:30]}...") 
        voice = cast_dict.get(speaker, list(cast_dict.values())[0])
        
        segment = None
        
        # 1. Try Microsoft Edge (High Quality) - ONLY if fallback not active
        if not fallback_mode_active:
            for attempt in range(3):
                try:
                    communicate = edge_tts.Communicate(text, voice)
                    temp_file = os.path.join(CACHE_DIR, f"temp_{i}.mp3")
                    await communicate.save(temp_file)
                    segment = AudioSegment.from_mp3(temp_file)
                    if os.path.exists(temp_file): os.remove(temp_file)
                    break 
                except Exception as e:
                    print(f"      ⚠️ EdgeTTS Error (Attempt {attempt+1}): {e}")
                    time.sleep(3)
            
            # If segment is still None, Edge failed 3 times. Trigger PERMANENT fallback.
            if segment is None:
                print("      🚨 EdgeTTS unavailable. Switching to Google Fallback (gTTS) for remainder of generation.")
                fallback_mode_active = True

        # 2. Try Google Backup (Robotic but Reliable)
        # This runs if fallback was JUST activated, or was ALREADY active
        if fallback_mode_active and HAS_GTTS:
            try:
                # Basic cleaner for gTTS (sometimes chokes on asterisks)
                clean_text = text.replace("*", "").replace("#", "")
                tts = gTTS(text=clean_text, lang='en')
                temp_file = os.path.join(CACHE_DIR, f"temp_gtts_{i}.mp3")
                tts.save(temp_file)
                segment = AudioSegment.from_mp3(temp_file)
                if os.path.exists(temp_file): os.remove(temp_file)
            except Exception as e:
                print(f"      ❌ Google Fallback Failed: {e}")

        if segment:
            combined_vocals += segment + AudioSegment.silent(duration=400)
        else:
            print(f"      ❌ FAILED: Line {i+1} dropped (No Audio).")

    print("   🎹 Mixing background tracks (Smart DJ Mode)...")
    music_mix = AudioSegment.empty()
    available_tracks = []
    if os.path.exists(MUSIC_DIR): available_tracks = [f for f in os.listdir(MUSIC_DIR) if f.endswith(".mp3")]

    if available_tracks:
        target_duration_ms = len(combined_vocals) + 10000
        while len(music_mix) < target_duration_ms:
            random_track_name = random.choice(available_tracks)
            update_music_stats(random_track_name)
            track_path = os.path.join(MUSIC_DIR, random_track_name)
            try:
                track = AudioSegment.from_mp3(track_path)
                track = normalize_volume(track, target_dBFS=-35.0)
                if len(music_mix) == 0: music_mix = track
                else: music_mix = music_mix.append(track, crossfade=min(len(music_mix), len(track), 3000))
            except:
                if len(music_mix) == 0: break 
        if len(music_mix) > target_duration_ms: music_mix = music_mix[:target_duration_ms]
        final_mix = music_mix.fade_out(3000).overlay(combined_vocals)
    else: final_mix = combined_vocals
            
    try:
        final_mix.export(filepath, format="mp3")
        attach_cover_art(filepath, topic)
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
    print(f"ℹ️  Running Podcast Studio: {VERSION_ID}")
    
    # --- DIAGNOSTIC STARTUP ---
    key = os.environ.get("GOOGLE_API_KEY")
    if key:
        print(f"🔑 Active Key: {key[:5]}... (Check this matches your new account)")
    print(f"🤖 Active Model: {MODEL_NAME}")
    
    target = INPUT_DIR
    resume_mode = False

    # --- ARGUMENT PARSING (ROBUST) ---
    if len(sys.argv) > 1:
        # Check if "RESUME" is in any argument, regardless of casing or path mess
        arg_str = str(sys.argv[1]).upper()
        if "RESUME" in arg_str: 
            resume_mode = True
            print("🟢 Mode: RESUME (Loading cache)")
        else: 
            target = sys.argv[1]
            print(f"🔵 Mode: NEW RUN (Target: {target})")

    check_and_clean_storage()

    processed_files = []
    data = None

    if resume_mode:
        print("🔄 Resuming from cache...")
        data = load_state()
        if not data:
            print("⚠️  No cache file found in data/cache/latest_run.json")
            print("   (Did the previous run complete Phase A/B?)")
            return
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
        safe_topic = re.sub(r'[^\w]+', '_', topic).strip('_')[:25]
        safe_slug = re.sub(r'[^\w]+', '_', cluster.get("file_slug", "Update")).strip('_')[:25]
        filename = f"{TODAY_STR}_{safe_topic}_{safe_slug}.mp3"
        
        print(f"\n📺 Studio: Producing '{topic}'...")
        cast_dict = VOICE_CAST["Default"]
        for key in VOICE_CAST:
            if key.lower() in topic.lower(): cast_dict = VOICE_CAST[key]; break
        cast_names = list(cast_dict.keys())
        
        source_count = len(cluster.get("source_urls", []))
        script = generate_script(topic, cluster["dossier"], cast_names, source_count)
        if script:
            final_file = await produce_audio(topic, script, filename)
            if final_file:
                index_metadata.append({"topic": topic, "filename": final_file, "source_urls": cluster.get("source_urls", [])})

    if index_metadata: 
        generate_index_html(index_metadata)
        if not resume_mode: archive_inputs(processed_files)

if __name__ == "__main__":
    asyncio.run(main())