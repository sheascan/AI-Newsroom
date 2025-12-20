import os
import sys
import json
import asyncio
import edge_tts
import time
import random
import re
import datetime
import shutil
import hashlib
import csv
import difflib
import numpy as np
import requests
from urllib.parse import urlparse, urlunparse
from pydub import AudioSegment
from newspaper import Article
from google import genai
from google.genai import types

# --- NEW DEPENDENCIES ---
try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_distances
except ImportError:
    print("❌ Error: Missing Science Libraries.")
    print("   Please run: pip install scikit-learn numpy requests")
    sys.exit(1)

# --- VERSION TRACKER ---
VERSION_ID = "Gen 134 (Restored Brain & Full Feature Set)"

# --- OPTIONAL IMPORTS ---
try:
    from mutagen.mp3 import MP3
    from mutagen.id3 import ID3, APIC, TIT2, TPE1, TALB, error
    HAS_MUTAGEN = True
except ImportError:
    HAS_MUTAGEN = False

try:
    from gtts import gTTS
    HAS_GTTS = True
except ImportError:
    HAS_GTTS = False

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
CHECKPOINT_FILE = os.path.join(CACHE_DIR, "gen119_clusters.json")
FAILURE_LOG_FILE = os.path.join(DATA_DIR, "scrape_failures.csv")

TODAY_STR = datetime.datetime.now().strftime("%d%b")
TODAY_OUTPUT_DIR = os.path.join(OUTPUT_DIR, TODAY_STR)
TODAY_INPUT_ARCHIVE = os.path.join(INPUT_DIR, "archive", TODAY_STR)

STORAGE_LIMIT_MB = 1024
EDGE_TTS_ALIVE = True 

# --- STATS TRACKER ---
STATS = {
    "files_scanned": 0,
    "raw_urls": 0,
    "unique_urls": 0,
    "scraped_success": 0,
    "scraped_cached": 0,
    "scraped_failed": 0,
    "junk_removed": 0,
    "dupes_removed": 0,
    "clusters_created": 0
}

# --- FILTERING ---
IGNORE_URL_TERMS = [
    "unsubscribe", "privacy policy", "terms", "opt_out", "login", "signup", 
    "facebook.com", "twitter.com", "instagram.com", "linkedin.com", 
    "youtube.com", "google.com", "apple.com", "tiktok.com", "arxiv.org",
    "www.w3.org", "liveintent", "nl.nytimes.com", "static.nyt.com", "static01.nyt.com"
]

IGNORE_HEADLINE_TERMS = ["webinar", "register now", "last chance", "giveaway", "subscribe", "newsletter", "terms and conditions", "privacy policy", "crossword", "spelling bee", "wordle", "live updates", "live blog"]

# EXTENSIONS TO SKIP
BAD_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp", ".ico",  
    ".mp3", ".wav", ".mp4", ".avi", ".mov", ".mkv", ".flv",            
    ".zip", ".tar", ".gz", ".rar", ".7z",                              
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",         
    ".css", ".js", ".xml", ".json", ".txt"                             
)

# CHECK API KEY
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("❌ Error: GOOGLE_API_KEY not found.")
    sys.exit(1)

# INITIALIZE CLIENT
client = genai.Client(api_key=api_key)

# --- MODEL SELECTION ---
MODEL_NAME = "gemini-2.5-flash-lite"
EMBEDDING_MODEL = "text-embedding-004" 

# --- TOPIC TRIGGER MAP ---
TOPIC_TRIGGERS = {
    "Politics": [
        "government", "minister", "election", "senate", "law", "parliament", "president", 
        "vote", "poll", "tax", "policy", "war", "conflict", "treaty", "ukraine", "russia", 
        "gaza", "israel", "china", "eu ", "european union", "white house", "congress", "mp"
    ],
    "Technology": [
        "tech", "ai ", "artificial intelligence", "crypto", "bitcoin", "space", "rocket", 
        "nasa", "spacex", "apple", "google", "microsoft", "amazon", "meta", "cyber", 
        "digital", "software", "startup", "innovation", "robot", "browser", "data", "web",
        "mobile", "phone", "app"
    ],
    "Sport": [
        "sport", "football", "rugby", "cricket", "tennis", "golf", "f1", "racing", 
        "league", "cup", "championship", "united", "city", "liverpool", "chelsea", 
        "arsenal", "leinster", "munster", "connacht", "ulster", "game", "match", 
        "score", "olympic", "athlete", "manager", "coach", "fifa", "uefa"
    ]
}

# VOICE CAST
VOICE_CAST = {
    "Politics": {"Sarah": "en-GB-SoniaNeural", "Mike": "en-US-GuyNeural"},
    "Technology": {"Tasha": "en-AU-NatashaNeural", "Chris": "en-US-ChristopherNeural"},
    "Sport": {"Connor": "en-IE-ConnorNeural", "Ryan": "en-GB-RyanNeural"},
    "Default": {"Alice": "en-US-AriaNeural", "Bob": "en-US-GuyNeural"} 
}

# --- HELPER: DETECT TOPIC ---
def detect_topic_category(headline):
    headline_lower = headline.lower()
    for category, keywords in TOPIC_TRIGGERS.items():
        for keyword in keywords:
            if keyword in headline_lower:
                return category
    return "Default"

# --- HELPER: CLEAN FILENAME ---
def clean_filename(text, max_length=50):
    if not text: return "untitled_topic"
    clean = re.sub(r'[^\w\s-]', '', text.lower())
    clean = re.sub(r'[-\s]+', '_', clean)
    return clean.strip()[:max_length]

# --- CLASS: CONTENT CLUSTERER ---
class ContentClusterer:
    def __init__(self, min_size=5, max_size=15): 
        self.min_size = min_size
        self.max_size = max_size

    def get_centroids(self, embeddings, labels):
        unique_labels = np.unique(labels)
        centroids = {}
        for label in unique_labels:
            indices = np.where(labels == label)[0]
            if len(indices) > 0:
                cluster_embeddings = embeddings[indices]
                centroids[label] = np.mean(cluster_embeddings, axis=0)
        return centroids

    def merge_orphans(self, embeddings, labels):
        refined_labels = labels.copy()
        
        # Pass 1: Polite Merge (Distance based)
        for _ in range(10):
            unique_labels, counts = np.unique(refined_labels, return_counts=True)
            sizes = dict(zip(unique_labels, counts))
            orphans = [l for l, s in sizes.items() if s < self.min_size]
            if not orphans: break 

            centroids = self.get_centroids(embeddings, refined_labels)
            orphans.sort(key=lambda l: sizes[l])
            
            orphan_label = orphans[0]
            orphan_centroid = centroids[orphan_label].reshape(1, -1)
            best_target = None
            min_dist = 0.45 
            
            for potential_target in unique_labels:
                if potential_target == orphan_label: continue
                target_centroid = centroids[potential_target].reshape(1, -1)
                dist = cosine_distances(orphan_centroid, target_centroid)[0][0]
                if (sizes[orphan_label] + sizes[potential_target]) <= self.max_size:
                    if dist < min_dist:
                        min_dist = dist
                        best_target = potential_target
            
            if best_target is not None:
                refined_labels[refined_labels == orphan_label] = best_target
            else: break
            
        return refined_labels

    def split_giant_clusters(self, articles, embeddings, labels):
        final_labels = labels.copy()
        while True:
            unique_labels, counts = np.unique(final_labels, return_counts=True)
            sizes = dict(zip(unique_labels, counts))
            giants = [l for l, s in sizes.items() if s > self.max_size]
            if not giants: break
            
            next_label_id = max(unique_labels) + 1
            for giant_label in giants:
                indices = np.where(final_labels == giant_label)[0]
                sub_embeddings = embeddings[indices]
                sub_clusterer = AgglomerativeClustering(n_clusters=2, metric='cosine', linkage='average')
                sub_labels = sub_clusterer.fit_predict(sub_embeddings)
                new_group_indices = indices[sub_labels == 1]
                final_labels[new_group_indices] = next_label_id
                next_label_id += 1
        return final_labels

    def run_clustering(self, articles, embeddings):
        n_articles = len(embeddings)
        n_clusters_initial = max(1, int(n_articles / 10))
        clusterer = AgglomerativeClustering(n_clusters=n_clusters_initial, metric='cosine', linkage='average')
        initial_labels = clusterer.fit_predict(embeddings)
        merged_labels = self.merge_orphans(embeddings, initial_labels)
        final_labels = self.split_giant_clusters(articles, embeddings, merged_labels)
        
        final_clusters_map = {}
        misc_articles = []
        for tag, arts in raw_clusters.items():
            if len(arts) < 3:
                misc_articles.extend(arts)
            else:
                final_clusters_map[tag] = arts
        if misc_articles:
            final_clusters_map["Group_Misc"] = misc_articles

        final_clusters = {"clusters": []}
        print("\n📊 --- FLIGHT MANIFEST ---")
        print(f"{'CLUSTER ID':<15} | {'COUNT':<5} | {'TOPIC NAME'}")
        print("-" * 50)
        
        for tag, arts in final_clusters_map.items():
            if not arts: continue
            
            if tag == "Group_Misc":
                topic_name = "News Roundup: Various Stories"
                safe_topic = "news_roundup_various_stories"
                detected_category = "Default"
            else:
                topic_name = arts[0]['headline']
                safe_topic = clean_filename(topic_name)
                detected_category = detect_topic_category(topic_name)
                
            display_name = topic_name
            if len(display_name) > 50: display_name = display_name[:47] + "..."

            print(f"{tag:<15} | {len(arts):<5} | {display_name} [{detected_category}]")
            
            full_dossier = ""
            cluster_sources = [] 
            for a in arts:
                full_dossier += f"HEADLINE: {a['headline']}\nFACTS: {a['full_text']}\n\n"
                cluster_sources.append({"title": a['headline'], "url": a['url']})
                
            final_clusters["clusters"].append({
                "topic": topic_name, 
                "file_slug": safe_topic, 
                "dossier": full_dossier, 
                "sources": cluster_sources,
                "is_misc": (tag == "Group_Misc")
            })
        print("-" * 50 + "\n")
        return final_clusters

# --- HELPER: API WRAPPER ---
def generate_with_retry(model_name, contents, config, retries=3):
    base_wait = 15  
    for attempt in range(retries):
        try:
            print(f"         Attempt {attempt+1}/{retries} sending to {model_name}...")
            return client.models.generate_content(model=model_name, contents=contents, config=config)
        except Exception as e:
            e_str = str(e).lower()
            if "429" in e_str or "resource_exhausted" in e_str or "503" in e_str or "overloaded" in e_str:
                wait = base_wait * (2 ** attempt)
                if wait > 60: wait = 60 
                print(f"         ⚠️  Rate Limit/Overload ({e_str[:20]}...).")
                for i in range(wait, 0, -1):
                    sys.stdout.write(f"\r         ⏳ Cooling down... {i}s   ")
                    sys.stdout.flush()
                    time.sleep(1)
                print("\n         🔄 Retrying now...")
            elif "404" in e_str and "not found" in e_str:
                print(f"         ❌ Error: {model_name} not found.")
                if model_name == "gemini-2.5-flash-lite":
                    print(f"         ➡️ Falling back to gemini-2.5-flash (Standard)...")
                    return generate_with_retry("gemini-2.5-flash", contents, config, retries-1)
                return None
            else:
                print(f"         ⚠️ API Error: {e}")
                time.sleep(5)
    print("         ❌ Failed after all retries.")
    return None

# --- HELPER: EMBEDDINGS ---
def get_embeddings_batch(texts):
    try:
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type="CLUSTERING")
        )
        return np.array([e.values for e in result.embeddings])
    except Exception as e:
        print(f"      ❌ Embedding Error: {e}")
        return None

# --- HELPER: AUDIO & FILES ---
def normalize_volume(sound, target_dBFS=-35.0):
    return sound.apply_gain(target_dBFS - sound.dBFS)

def attach_cover_art(mp3_path, topic_title):
    if not HAS_MUTAGEN: return
    try:
        audio = MP3(mp3_path, ID3=ID3)
        try: audio.add_tags()
        except error: pass
        if os.path.exists(os.path.join(BASE_DIR, "cover.jpg")):
            with open(os.path.join(BASE_DIR, "cover.jpg"), 'rb') as img:
                audio.tags.add(APIC(encoding=3, mime='image/jpeg', type=3, desc=u'Cover', data=img.read()))
        audio.tags.add(TIT2(encoding=3, text=topic_title))
        audio.save()
    except: pass

# --- LOGGING HELPERS ---
def log_failure(url, reason):
    try:
        with open(FAILURE_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([datetime.datetime.now().strftime("%H:%M:%S"), url, reason])
    except: pass

def analyze_failures():
    if not os.path.exists(FAILURE_LOG_FILE): return
    reasons = {}
    try:
        with open(FAILURE_LOG_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3: continue
                r = row[2]
                reasons[r] = reasons.get(r, 0) + 1
        print("\n🔎 --- SCRAPE FAILURE FORENSICS ---")
        sorted_reasons = sorted(reasons.items(), key=lambda x: x[1], reverse=True)
        for reason, count in sorted_reasons[:5]:
            print(f"   - {reason}: {count} occurrences")
        print(f"   👉 Full log: {FAILURE_LOG_FILE}\n")
    except: pass

# --- STAGE 3: CLUSTERER (RESTORED) ---
def ai_cluster_and_summarize(article_objects):
    print("\n🔍 --- FILTERING & LOGIC ---")
    valid_articles = []
    seen = []
    
    for obj in article_objects:
        if not obj: continue
        if obj['status'] == 'junk': 
            STATS["junk_removed"] += 1; continue
        is_dupe = False
        for s in seen:
            if difflib.SequenceMatcher(None, obj['headline'].lower(), s).ratio() > 0.8:
                is_dupe = True; break
        if is_dupe:
            STATS["dupes_removed"] += 1; continue
        seen.append(obj['headline'].lower())
        valid_articles.append(obj)

    if not valid_articles: return None

    print(f"   -> Embedding {len(valid_articles)} valid articles...")
    headlines = [f"{a['headline']}: {a['snippet']}" for a in valid_articles]
    embeddings = get_embeddings_batch(headlines)
    if embeddings is None: return None

    print("   -> Clustering (Split & Merge)...")
    cluster_engine = ContentClusterer(min_size=5, max_size=15)
    raw_clusters = cluster_engine.run_clustering(valid_articles, embeddings)

    # POST-CLUSTERING CLEANUP
    final_clusters_map = {}
    misc_articles = []
    
    for tag, arts in raw_clusters.items():
        if len(arts) < 3:
            misc_articles.extend(arts)
        else:
            final_clusters_map[tag] = arts
            
    if misc_articles:
        final_clusters_map["Group_Misc"] = misc_articles

    final_clusters = {"clusters": []}
    print("\n📊 --- FLIGHT MANIFEST ---")
    print(f"{'CLUSTER ID':<15} | {'COUNT':<5} | {'TOPIC NAME'}")
    print("-" * 50)
    
    for tag, arts in final_clusters_map.items():
        if not arts: continue
        
        if tag == "Group_Misc":
            topic_name = "News Roundup: Various Stories"
            safe_topic = "news_roundup_various_stories"
            detected_category = "Default"
        else:
            topic_name = arts[0]['headline']
            safe_topic = clean_filename(topic_name)
            detected_category = detect_topic_category(topic_name)
            
        display_name = topic_name
        if len(display_name) > 50: display_name = display_name[:47] + "..."

        print(f"{tag:<15} | {len(arts):<5} | {display_name} [{detected_category}]")
        
        full_dossier = ""
        cluster_sources = [] 
        for a in arts:
            full_dossier += f"HEADLINE: {a['headline']}\nFACTS: {a['full_text']}\n\n"
            cluster_sources.append({"title": a['headline'], "url": a['url']})
            
        final_clusters["clusters"].append({
            "topic": topic_name, 
            "file_slug": safe_topic, 
            "dossier": full_dossier, 
            "sources": cluster_sources,
            "is_misc": (tag == "Group_Misc")
        })
    print("-" * 50 + "\n")
    return final_clusters

# --- STAGE 1 & 2: SCRAPING ---
def load_all_inputs(target_path):
    files = []
    if os.path.isdir(target_path):
        for root, _, fs in os.walk(target_path):
            if "archive" not in root:
                files.extend([os.path.join(root, f) for f in fs if not f.startswith(".")])
    else: files.append(target_path)
    STATS["files_scanned"] = len(files)
    
    raw_urls = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
                found = re.findall(r'(https?://[^\s<>"]+|www\.[^\s<>"]+)', content)
                raw_urls.extend(found)
        except: pass
    STATS["raw_urls"] = len(raw_urls)
    
    clean_urls = set()
    final_urls = []
    
    for u in raw_urls:
        if u.endswith("="): continue 
        if u.endswith("="): u = u.split("?")[0]
        if any(x in u.lower() for x in IGNORE_URL_TERMS): continue
        try:
            path = urlparse(u).path.lower()
            if path.endswith(BAD_EXTENSIONS): continue
        except: pass
        clean = urlunparse(urlparse(u)._replace(query="", fragment=""))
        if clean not in clean_urls:
            clean_urls.add(clean)
            final_urls.append(clean)
            
    STATS["unique_urls"] = len(final_urls)
    return final_urls

def clean_text(text):
    if not text: return ""
    return re.sub(r'\s+', ' ', text).strip()

def fetch_article_text(url):
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
    if not os.path.exists(LIBRARY_DIR): os.makedirs(LIBRARY_DIR)
    cache_path = os.path.join(LIBRARY_DIR, f"{url_hash}.json")

    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f: 
                STATS["scraped_cached"] += 1
                return json.load(f)
        except: pass

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code >= 400:
            log_failure(url, f"HTTP {response.status_code}")
            STATS["scraped_failed"] += 1
            return None

        article = Article(url)
        article.set_html(response.text)
        article.parse()
        
        if not article.text or len(article.text) < 100:
             log_failure(url, "Empty Text (JS Block)")
             STATS["scraped_failed"] += 1
             return None

        is_junk = any(t in article.title.lower() for t in IGNORE_HEADLINE_TERMS)
        raw_text = clean_text(article.text)
        truncated_text = raw_text[:1000]
        if len(raw_text) > 1000: truncated_text += "..."

        data = {
            "url": url, 
            "headline": article.title, 
            "snippet": truncated_text[:200], 
            "full_text": truncated_text, 
            "status": "junk" if is_junk else "scraped"
        }
        with open(cache_path, 'w') as f: json.dump(data, f)
        STATS["scraped_success"] += 1
        return data

    except Exception as e:
        log_failure(url, f"Error: {str(e)[:50]}")
        STATS["scraped_failed"] += 1
        return None

# --- STAGE 4: PRODUCER ---
def generate_script(topic, dossier, cast_names, source_count):
    host1, host2 = cast_names 
    safe_dossier = dossier[:30000] 
    
    if source_count <= 5: mode = "Briefing"; words = "600 words"
    else: mode = "Deep Dive"; words = "1200 words"

    print(f"      📝 Draft: {mode} ({source_count} items) - Hosts: {host1} & {host2}")
    
    prompt = f"""
    Write a {mode} podcast script ({words}).
    TOPIC: {topic}
    HOSTS: {host1} & {host2}.
    OUTPUT JSON: [ {{"speaker": "{host1}", "text": "..."}}, {{"speaker": "{host2}", "text": "..."}} ]
    DOSSIER: {safe_dossier}
    """
    
    resp = generate_with_retry(MODEL_NAME, prompt, types.GenerateContentConfig(response_mime_type="application/json"))
    if resp:
        try:
            txt = resp.text.replace("```json", "").replace("```", "").strip()
            if "[" in txt: txt = txt[txt.find('['):txt.rfind(']')+1]
            return json.loads(txt)
        except: pass
    return []

# --- STAGE 5: AUDIO ---
async def produce_audio(topic, script, filename, cast_dict):
    global EDGE_TTS_ALIVE
    path = os.path.join(TODAY_OUTPUT_DIR, filename)
    combined = AudioSegment.empty()
    
    # --- FUZZY MATCHING SETUP ---
    clean_map = {}
    host_keys = list(cast_dict.keys()) # [FemaleName, MaleName]
    
    for k, v in cast_dict.items():
        clean_map[k.lower()] = v
        
    # Generic fallbacks
    clean_map["host 1"] = cast_dict[host_keys[0]]
    clean_map["host 2"] = cast_dict[host_keys[1]]
    clean_map["speaker 1"] = cast_dict[host_keys[0]]
    clean_map["speaker 2"] = cast_dict[host_keys[1]]
    
    # Dynamic Tracking
    unknown_map = {}
    unknown_count = 0
    seen_speakers = set()
    
    for line in script:
        raw_spk = line.get("speaker", "Host")
        txt = line.get("text", "")
        
        spk_clean = raw_spk.strip().lower().replace(":", "")
        voice = clean_map.get(spk_clean)
        
        if not voice:
            if host_keys[1].lower() in spk_clean:
                voice = cast_dict[host_keys[1]]
            elif host_keys[0].lower() in spk_clean:
                voice = cast_dict[host_keys[0]]

        if not voice:
            if raw_spk not in unknown_map:
                assigned_key = host_keys[unknown_count % 2]
                unknown_map[raw_spk] = cast_dict[assigned_key]
                unknown_count += 1
            voice = unknown_map[raw_spk]
            
        if raw_spk not in seen_speakers:
            seen_speakers.add(raw_spk)
            assigned_name = "Unknown"
            for k, v in cast_dict.items():
                if v == voice: assigned_name = k
            print(f"      🎤 Casting Debug: '{raw_spk}' -> Mapped to {assigned_name}")

        seg = None
        
        if EDGE_TTS_ALIVE:
            try:
                comm = edge_tts.Communicate(txt, voice)
                tmp = os.path.join(CACHE_DIR, f"tmp_{random.randint(0,999)}.mp3")
                await comm.save(tmp)
                seg = AudioSegment.from_mp3(tmp)
                os.remove(tmp)
            except: EDGE_TTS_ALIVE = False
            
        if not seg and HAS_GTTS:
            try:
                tts = gTTS(txt, lang='en')
                tmp = os.path.join(CACHE_DIR, f"tmp_g_{random.randint(0,999)}.mp3")
                tts.save(tmp)
                seg = AudioSegment.from_mp3(tmp)
                os.remove(tmp)
            except: pass
            
        if seg: combined += seg + AudioSegment.silent(duration=400)
    
    if os.path.exists(MUSIC_DIR):
        tracks = [f for f in os.listdir(MUSIC_DIR) if f.endswith(".mp3")]
        if tracks:
            bg = AudioSegment.empty()
            while len(bg) < len(combined) + 10000:
                bg += normalize_volume(AudioSegment.from_mp3(os.path.join(MUSIC_DIR, random.choice(tracks))))
            combined = bg[:len(combined)+2000].fade_out(3000).overlay(combined)
            
    combined.export(path, format="mp3")
    attach_cover_art(path, topic)
    return filename

# --- NEW: ARCHIVE FUNCTION ---
def archive_processed_inputs():
    print(f"\n📦 --- ARCHIVING INPUTS ---")
    if not os.path.exists(TODAY_INPUT_ARCHIVE):
        os.makedirs(TODAY_INPUT_ARCHIVE)
        
    files_moved = 0
    # List files in INPUT_DIR
    for filename in os.listdir(INPUT_DIR):
        file_path = os.path.join(INPUT_DIR, filename)
        
        # Skip directories (like the 'archive' folder itself!)
        if os.path.isdir(file_path): continue
        
        # Move file
        try:
            shutil.move(file_path, os.path.join(TODAY_INPUT_ARCHIVE, filename))
            files_moved += 1
        except Exception as e:
            print(f"   ⚠️ Failed to archive {filename}: {e}")
            
    print(f"   ✅ Moved {files_moved} files to: {TODAY_INPUT_ARCHIVE}")

# --- MAIN ---
async def main():
    print(f"ℹ️  Podcast Studio: {VERSION_ID}")
    
    if os.path.exists(FAILURE_LOG_FILE): os.remove(FAILURE_LOG_FILE)
    urls = load_all_inputs(INPUT_DIR)

    raw_urls_file = os.path.join(DATA_DIR, "raw_urls.txt")
    try:
        with open(raw_urls_file, "w", encoding="utf-8") as f:
            for u in urls: f.write(f"{u}\n")
        print(f"   💾 Saved URLs to: {raw_urls_file}")
    except Exception as e:
        print(f"   ⚠️ Failed to save raw URLs: {e}")
    
    print(f"   📂 Input Report:")
    print(f"      - Files Scanned: {STATS['files_scanned']}")
    print(f"      - Unique URLs:   {STATS['unique_urls']}")
    
    articles = [fetch_article_text(u) for u in urls]
    
    print(f"      - Cached:        {STATS['scraped_cached']}")
    print(f"      - Scraped OK:    {STATS['scraped_success']}")
    print(f"      - Scraped Fail:  {STATS['scraped_failed']}")
    
    analyze_failures()
    
    data = ai_cluster_and_summarize(articles)
    if data:
        index_items = []
        
        for c in data["clusters"]:
            fname = f"{TODAY_STR}_{c['file_slug']}.mp3"
            full_path = os.path.join(TODAY_OUTPUT_DIR, fname)
            if os.path.exists(full_path):
                print(f"      ⏩ Found existing audio: {fname} (Skipping Generation)")
                index_items.append({
                    "topic": c['topic'],
                    "filename": fname,
                    "sources": c['sources']
                })
                continue 
            
            print(f"\n📺 Studio: Producing '{c['topic'][:50]}...'")
            print("      ⏳ Rate Limit Safety: Pausing 10s...")
            time.sleep(10)
            
            # --- LUCKY DIP LOGIC ---
            topic_category = detect_topic_category(c['topic'])
            
            if topic_category == "Default":
                 # ROLL THE DICE
                 random_key = random.choice(list(VOICE_CAST.keys()))
                 selected_cast = VOICE_CAST[random_key]
                 print(f"      🎲 Random Cast Selected: {random_key} ({list(selected_cast.keys())})")
            else:
                 selected_cast = VOICE_CAST[topic_category]
                 print(f"      🎤 Cast Matches Topic: {topic_category} ({list(selected_cast.keys())})")

            host_names = list(selected_cast.keys())
            
            script = generate_script(c["topic"], c["dossier"], host_names, len(c["sources"]))
            if script:
                await produce_audio(c["topic"], script, fname, selected_cast)
                print(f"      ✅ Saved: {fname}")
                index_items.append({
                    "topic": c['topic'],
                    "filename": fname,
                    "sources": c['sources']
                })

        if index_items:
            html = f"""<html><head><title>Briefing {TODAY_STR}</title>
            <style>body{{font-family:sans-serif; padding:20px; max-width:800px; margin:0 auto;}} 
            .pod{{border:1px solid #ccc; padding:20px; margin-bottom:20px; border-radius:8px; background:#f9f9f9;}}
            .source-list{{margin-top:10px; font-size:0.9em; color:#555;}}
            .source-list a{{text-decoration:none; color:#007bff;}}
            </style></head><body><h1>Daily Briefing: {TODAY_STR}</h1>"""
            
            for item in index_items:
                html += f"""<div class='pod'>
                <h2>{item['topic']}</h2>
                <audio controls src='{item['filename']}'></audio>
                <div class='source-list'><strong>Reference Articles:</strong><ul>"""
                for s in item['sources']:
                    html += f"<li><a href='{s['url']}' target='_blank'>{s['title']}</a></li>"
                html += "</ul></div></div>"
            
            html += "</body></html>"
            with open(os.path.join(TODAY_OUTPUT_DIR, "index.html"), "w") as f: f.write(html)
            print(f"\n✅ Index Generated: {os.path.join(TODAY_OUTPUT_DIR, 'index.html')}")

    # --- RUN ARCHIVING AT THE END ---
    archive_processed_inputs()

if __name__ == "__main__":
    if not os.path.exists(TODAY_OUTPUT_DIR): os.makedirs(TODAY_OUTPUT_DIR)
    asyncio.run(main())