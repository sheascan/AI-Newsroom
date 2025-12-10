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
import concurrent.futures 
from urllib.parse import urlparse, urlunparse
from email import policy
from pydub import AudioSegment
from newspaper import Article
from google import genai
from google.genai import types

# --- VERSION TRACKER ---
VERSION_ID = "Gen 37 (Robust Parser & Syntax Fix)"

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
    print("⚠️  'gTTS' library not found. Backup audio engine disabled.")

# --- CONFIGURATION ---
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

TODAY_STR = datetime.datetime.now().strftime("%d%b")
TODAY_OUTPUT_DIR = os.path.join(OUTPUT_DIR, TODAY_STR)
TODAY_INPUT_ARCHIVE = os.path.join(INPUT_DIR, "archive", TODAY_STR)
STORAGE_LIMIT_MB = 1024

IGNORE_URL_TERMS = ["unsubscribe", "manage your emails", "view in browser", "privacy policy", "terms of service", "manage_preferences", "login", "signup", "signin", "facebook.com", "twitter.com", "instagram.com", "linkedin.com", "youtube.com", "google.com", "apple.com", "tiktok.com", "help center", "contact us", "advertisement", "click here", "arxiv.org"]
IGNORE_HEADLINE_TERMS = ["sale", "off", "discount", "webinar", "register now", "last chance", "giveaway", "promo", "deal of the day", "subscribe", "newsletter", "metro sport", "metro uk", "instagram", "make your day", "terms and conditions", "cookies policy", "terms of use", "preferences", "ai & tech newsletter", "courses on chatgpt", "techpssdec2", "workos", "work with us", "privacy policy", "sign up for", "email newsletters", "ad choices", "crossword", "spelling bee", "connections — the", "strands:", "minichess", "sudoku", "wordle", "the guardian view on", "the long read", "today in focus", "new york times", "the irish times", "one ai to", "walk mate", "visualize knowledge", "your kitchen assistant", "enable ai", "proofly", "free guide to", "threads, say more", "techpresso", "beanvest", "resurf"]

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("❌ Error: GOOGLE_API_KEY not found.")
    sys.exit(1)

client = genai.Client()
MODEL_NAME = "gemini-2.5-flash-lite" 
BATCH_SIZE = 75      
MERGE_THRESHOLD = 6  
SLEEP_TIME = 20     

VOICE_CAST = {
    "Politics": {"Sarah": "en-GB-SoniaNeural", "Mike": "en-US-GuyNeural"},
    "Technology": {"Tasha": "en-AU-NatashaNeural", "Chris": "en-US-ChristopherNeural"},
    "Sport": {"Connor": "en-IE-ConnorNeural", "Ryan": "en-GB-RyanNeural"},
    "Default": {"Alice": "en-US-AriaNeural", "Bob": "en-GB-RyanNeural"} 
}

def clean_json_response(text):
    return text.replace("```json", "").replace("```", "").strip()

def generate_with_retry(model_name, contents, config, retries=5):
    base_wait = 30 
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model_name, contents=contents, config=config
            )
            return response
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait_time = base_wait * (2 ** attempt) 
                print(f"      ⚠️  Quota Hit (429). Attempt {attempt+1}/{retries}. Pausing {wait_time}s...")
                time.sleep(wait_time)
            elif "503" in error_str:
                print(f"      ⚠️  Server Overload (503). Retrying in 10s...")
                time.sleep(10)
            else:
                print(f"      ⚠️ API Error: {e}. Retrying in 10s...")
                time.sleep(10)
    return None

def normalize_volume(sound, target_dBFS=-35.0):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

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
    except Exception as e: print(f"      ⚠️ Failed to attach cover: {e}")

def check_and_clean_storage():
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(DATA_DIR):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp): total_size += os.path.getsize(fp)
    total_mb = total_size / (1024 * 1024)
    print(f"\n💾 Storage: {total_mb:.2f} MB / {STORAGE_LIMIT_MB} MB")

def save_state(data):
    if not os.path.exists(CACHE_DIR): os.makedirs(CACHE_DIR)
    with open(os.path.join(CACHE_DIR, "latest_run.json"), "w") as f: json.dump(data, f, indent=4)

def load_state():
    fp = os.path.join(CACHE_DIR, "latest_run.json")
    if os.path.exists(fp):
        with open(fp, "r") as f: return json.load(f)
    return None

def generate_audit_report(all_articles_metadata):
    if not os.path.exists(TODAY_OUTPUT_DIR): os.makedirs(TODAY_OUTPUT_DIR)
    report_path = os.path.join(TODAY_OUTPUT_DIR, f"Filtering_Report_{TODAY_STR}.html")
    kept = sum(1 for a in all_articles_metadata if a['status'] == 'kept')
    junk = sum(1 for a in all_articles_metadata if a['status'] == 'junk')
    dupe = sum(1 for a in all_articles_metadata if a['status'] == 'duplicate')
    html = f"<html><body><h1>Report {TODAY_STR}</h1><p>Kept: {kept} | Junk: {junk} | Dupe: {dupe}</p><table>"
    for item in all_articles_metadata:
        html += f"<tr class='{item['status']}'><td>{item['status']}</td><td>{item.get('reject_reason','')}</td><td>{item.get('headline','')}</td><td><a href='{item['url']}'>Link</a></td></tr>"
    html += "</table></body></html>"
    with open(report_path, "w", encoding='utf-8') as f: f.write(html)
    print(f"   📊 Audit Report: {report_path}")

def load_all_inputs(target_path):
    files = []
    if os.path.isdir(target_path):
        for root, _, fs in os.walk(target_path):
            if "archive" not in root:
                for f in fs: 
                    if not f.startswith("."): files.append(os.path.join(root, f))
    print(f"   -> Processing {len(files)} files...")
    raw_urls = []
    for fp in files:
        try:
            with open(fp, "r", errors="ignore") as f: content = f.read()
            raw_urls.extend(re.findall(r'href=["\'](https?://[^"\']+)["\']', content))
        except: pass
    clean_urls_set = set()
    final_urls = []
    for url in raw_urls:
        if any(t in url.lower() for t in IGNORE_URL_TERMS): continue
        parsed = urlparse(url)
        clean = urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))
        if clean not in clean_urls_set:
            clean_urls_set.add(clean)
            final_urls.append(clean) 
    return final_urls, files

def fetch_single_article(url):
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
    if not os.path.exists(LIBRARY_DIR): os.makedirs(LIBRARY_DIR)
    cache_path = os.path.join(LIBRARY_DIR, f"{url_hash}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f: return json.load(f)
        except: pass
    try:
        article = Article(url)
        article.download()
        article.parse()
        headline = article.title
        is_junk = False
        reason = ""
        for term in IGNORE_HEADLINE_TERMS:
            if term.lower() in headline.lower():
                is_junk = True
                reason = f"Keyword: {term}"
                break
        data = {
            "url": url, "headline": headline, 
            "snippet": f"{headline}. {article.text[:300]}...",
            "full_text": article.text[:2500].replace("\n", " "),
            "status": "junk" if is_junk else "scraped",
            "reject_reason": reason
        }
        with open(cache_path, 'w') as f: json.dump(data, f)
        return data
    except: return None

def scrape_in_parallel(urls):
    print(f"🔍 Scraping {len(urls)} articles (Parallel Mode)...")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_single_article, url): url for url in urls}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res: results.append(res)
    return results

def ai_cluster_and_summarize(article_objects):
    print("   -> Refining (Deduping & Re-Filtering)...")
    unique_articles = []
    seen_headlines = []
    all_meta = []
    for obj in article_objects:
        if not obj: continue
        status = obj.get('status', 'scraped')
        reason = obj.get('reject_reason', '')
        if status != 'junk':
            for term in IGNORE_HEADLINE_TERMS:
                if term.lower() in obj['headline'].lower():
                    status = 'junk'; reason = f"Keyword: {term}"; break
        if status == 'junk':
            obj['status'] = status; obj['reject_reason'] = reason
            all_meta.append(obj); continue
        is_dupe = False
        for seen in seen_headlines:
            if difflib.SequenceMatcher(None, obj['headline'].lower(), seen).ratio() > 0.8:
                is_dupe = True; status = 'duplicate'; reason = f"Similar to: {seen[:30]}..."
                break
        if not is_dupe:
            seen_headlines.append(obj['headline'].lower())
            status = 'kept'
            unique_articles.append(obj)
        obj['status'] = status; obj['reject_reason'] = reason
        all_meta.append(obj)
    generate_audit_report(all_meta)
    print(f"🧠 Analyzing {len(unique_articles)} unique articles...")
    indexed_articles = {i: obj for i, obj in enumerate(unique_articles)}
    
    master_buckets = {}
    if os.path.exists(CHECKPOINT_FILE):
        try: 
            with open(CHECKPOINT_FILE, 'r') as f: master_buckets = json.load(f)
            print("   📂 Loaded Checkpoint.")
        except: pass

    if not master_buckets:
        print(f"   -> Phase A: Sorting ({BATCH_SIZE} items/batch)...")
        ids = list(indexed_articles.keys())
        for i in range(0, len(ids), BATCH_SIZE):
            batch_ids = ids[i:i+BATCH_SIZE]
            print(f"      Processing Batch {i//BATCH_SIZE + 1}...")
            prompt = "Sort into [World, Politics, Tech, Sport, Culture, Economy, Science]. JSON: {\"World\": [1, 2]}"
            inputs = "".join([f"ID [{x}]: {indexed_articles[x]['headline']}\n" for x in batch_ids])
            resp = generate_with_retry(MODEL_NAME, prompt + inputs, types.GenerateContentConfig(response_mime_type="application/json"))
            if resp:
                try:
                    clean = clean_json_response(resp.text)
                    for k,v in json.loads(clean).items():
                        master_buckets.setdefault(k, []).extend(v)
                except: pass
            time.sleep(SLEEP_TIME)
        with open(CHECKPOINT_FILE, 'w') as f: json.dump(master_buckets, f)
        print("   💾 Checkpoint Saved.")

    final_clusters = {"clusters": []}
    merged_buckets = {}
    digest_ids = []
    for theme, ids in master_buckets.items():
        valid_ids = []
        if isinstance(ids, list):
            for x in ids:
                if isinstance(x, (int, str)) and str(x).isdigit():
                    valid_ids.append(x)
                elif isinstance(x, list): # Handle nested lists
                    for sub in x:
                        if str(sub).isdigit(): valid_ids.append(str(sub))
        
        if len(valid_ids) < MERGE_THRESHOLD:
            print(f"      🔹 Merging '{theme}' ({len(valid_ids)} items) into Digest...")
            digest_ids.extend(valid_ids)
        else:
            merged_buckets[theme] = valid_ids
    if digest_ids:
        merged_buckets["News Digest"] = digest_ids

    for theme, ids in merged_buckets.items():
        if not ids: continue
        print(f"   -> Phase B: Clustering '{theme}' ({len(ids)})...")
        prompt = f"Cluster '{theme}' articles into episodes. JSON: {{ 'clusters': [ {{ 'topic': 'Title', 'file_slug': 'Slug', 'ids': [1, 2] }} ] }}"
        inputs = "".join([f"ID [{x}]: {indexed_articles[int(x)]['snippet']}\n" for x in ids if int(x) in indexed_articles])
        resp = generate_with_retry(MODEL_NAME, prompt + inputs, types.GenerateContentConfig(response_mime_type="application/json"))
        if resp:
            try:
                clean = clean_json_response(resp.text)
                for c in json.loads(clean).get("clusters", []):
                    dossier = ""
                    urls = []
                    for aid in c.get("ids", []):
                        if int(aid) in indexed_articles:
                            art = indexed_articles[int(aid)]
                            dossier += f"HEADLINE: {art['headline']}\nTEXT: {art['full_text']}\n\n"
                            urls.append(art['url'])
                    if dossier:
                        final_clusters["clusters"].append({"topic": c["topic"], "file_slug": c["file_slug"], "dossier": dossier, "source_urls": urls})
            except: pass
        time.sleep(SLEEP_TIME)
    
    if os.path.exists(CHECKPOINT_FILE): os.remove(CHECKPOINT_FILE)
    return final_clusters

def generate_script(topic, dossier, cast):
    safe_slug = re.sub(r'[^\w]+', '_', topic).strip('_')[:50]
    vh = hashlib.md5(dossier.encode('utf-8')).hexdigest()[:10]
    vpath = os.path.join(SCRIPT_DIR, f"{TODAY_STR}_{safe_slug}_{vh}.json")
    if os.path.exists(vpath):
        print("      📜 Loaded Script from Vault.")
        try: 
            with open(vpath) as f: return json.load(f)
        except: pass
    print(f"      📝 Writing Script...")
    prompt = f"Write podcast script. TOPIC: {topic}. HOSTS: {cast[0]}, {cast[1]}. JSON: [ {{'speaker': '{cast[0]}', 'text': '...'}} ]"
    resp = generate_with_retry(MODEL_NAME, prompt + "\nDOSSIER: " + dossier[:10000], types.GenerateContentConfig(response_mime_type="application/json"))
    if resp:
        try:
            clean = clean_json_response(resp.text)
            data = json.loads(clean)
            if not os.path.exists(SCRIPT_DIR): os.makedirs(SCRIPT_DIR)
            with open(vpath, 'w') as f: json.dump(data, f)
            return data
        except: pass
    return []

async def produce_audio(topic, script, filename):
    if not os.path.exists(TODAY_OUTPUT_DIR): os.makedirs(TODAY_OUTPUT_DIR)
    path = os.path.join(TODAY_OUTPUT_DIR, filename)
    print(f"   🎙️  Recording: {filename}")
    cast = VOICE_CAST["Default"]
    combined = AudioSegment.empty()
    edge_permanently_failed = False
    
    for i, line in enumerate(script):
        spk = line.get("speaker", "Host")
        txt = line.get("text", "")
        voice = cast.get(spk, list(cast.values())[0])
        print(f"      -> Line {i+1}: {txt[:30]}...")
        seg = None
        
        if not edge_permanently_failed:
            for attempt in range(3):
                try:
                    comm = edge_tts.Communicate(txt, voice)
                    tmp = os.path.join(CACHE_DIR, f"t{i}.mp3")
                    await comm.save(tmp)
                    seg = AudioSegment.from_mp3(tmp)
                    os.remove(tmp)
                    break
                except: time.sleep(2)
            if seg is None:
                print("      🚫 EdgeTTS Banned/Failed. Switching to gTTS for remaining lines.")
                edge_permanently_failed = True

        if (seg is None or edge_permanently_failed) and HAS_GTTS:
            try:
                tts = gTTS(txt, lang='en')
                tmp = os.path.join(CACHE_DIR, f"g{i}.mp3")
                tts.save(tmp)
                seg = AudioSegment.from_mp3(tmp)
                os.remove(tmp)
            except: pass
        if seg: combined += seg + AudioSegment.silent(duration=300)
    
    if os.path.exists(MUSIC_DIR):
        tracks = [os.path.join(MUSIC_DIR, x) for x in os.listdir(MUSIC_DIR) if x.endswith(".mp3")]
        if tracks:
            bg = AudioSegment.from_mp3(random.choice(tracks))
            bg = normalize_volume(bg, -35.0)
            if len(bg) < len(combined): bg = bg * (int(len(combined)/len(bg)) + 1)
            combined = combined.overlay(bg[:len(combined) + 5000].fade_out(3000))
    combined.export(path, format="mp3")
    attach_cover_art(path, topic)
    return filename

async def main():
    print(f"ℹ️  {VERSION_ID}")
    print(f"🔑 Key: {os.environ.get('GOOGLE_API_KEY', '')[:5]}... | Model: {MODEL_NAME}")
    if len(sys.argv) > 1 and sys.argv[1] == "RESUME":
        print("🔄 Resuming from cache...")
        data = load_state()
        files = [] 
    else:
        check_and_clean_storage()
        urls, files = load_all_inputs(INPUT_DIR)
        articles = scrape_in_parallel(urls)
        data = ai_cluster_and_summarize(articles)
        save_state(data)
    
    if not data or not data.get("clusters"):
        print("❌ No news clusters. Exiting.")
        return

    meta = []
    for c in data.get("clusters", []):
        print(f"\n📺 Producing '{c['topic']}'...")
        script = generate_script(c['topic'], c['dossier'], list(VOICE_CAST["Default"].keys()))
        if script:
            fname = f"{TODAY_STR}_{c['file_slug']}.mp3"
            res = await produce_audio(c['topic'], script, fname)
            if res: meta.append({"topic": c['topic'], "filename": res, "source_urls": c['source_urls']})
    generate_index_html(meta)
    if files: archive_inputs(files)

if __name__ == "__main__":
    asyncio.run(main())