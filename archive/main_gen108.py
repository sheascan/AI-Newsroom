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
    print("   Please run: pip install scikit-learn numpy")
    sys.exit(1)

# --- VERSION TRACKER ---
VERSION_ID = "Gen 108 (The Lawnmower Edition)"

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
CHECKPOINT_FILE = os.path.join(CACHE_DIR, "gen108_clusters.json")

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
    "scraped_failed": 0,
    "junk_removed": 0,
    "dupes_removed": 0,
    "clusters_created": 0
}

# --- FILTERING ---
IGNORE_URL_TERMS = ["unsubscribe", "privacy policy", "terms", "opt_out", "login", "signup", "facebook.com", "twitter.com", "instagram.com", "linkedin.com", "youtube.com", "google.com", "apple.com", "tiktok.com", "arxiv.org"]
IGNORE_HEADLINE_TERMS = ["webinar", "register now", "last chance", "giveaway", "subscribe", "newsletter", "terms and conditions", "privacy policy", "crossword", "spelling bee", "wordle", "live updates", "live blog"]

# CHECK API KEY
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("❌ Error: GOOGLE_API_KEY not found.")
    sys.exit(1)

client = genai.Client()
MODEL_NAME = "gemini-2.0-flash-lite" 
EMBEDDING_MODEL = "text-embedding-004" 

# VOICE CAST
VOICE_CAST = {
    "Politics": {"Sarah": "en-GB-SoniaNeural", "Mike": "en-US-GuyNeural"},
    "Technology": {"Tasha": "en-AU-NatashaNeural", "Chris": "en-US-ChristopherNeural"},
    "Sport": {"Connor": "en-IE-ConnorNeural", "Ryan": "en-GB-RyanNeural"},
    "Default": {"Alice": "en-US-AriaNeural", "Bob": "en-GB-RyanNeural"} 
}

# --- CLASS: CONTENT CLUSTERER ---
class ContentClusterer:
    def __init__(self, min_size=5, max_size=20):
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
            min_dist = float('inf')
            
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

    def run_clustering(self, articles, embeddings):
        n_articles = len(embeddings)
        n_clusters_initial = max(1, int(n_articles / 12))
        
        # 1. Base Clustering
        clusterer = AgglomerativeClustering(n_clusters=n_clusters_initial, metric='cosine', linkage='average')
        initial_labels = clusterer.fit_predict(embeddings)
        
        # 2. Enforce Size Constraints
        final_labels = self.merge_orphans(embeddings, initial_labels)
        
        # 3. Organize Output
        clustered_data = {}
        unique_labels = np.unique(final_labels)
        for label in unique_labels:
            indices = np.where(final_labels == label)[0]
            cluster_articles = [articles[i] for i in indices]
            tag = f"Group_{label}" # Temporary tag
            clustered_data[tag] = cluster_articles
            
        STATS["clusters_created"] = len(clustered_data)
        return clustered_data

# --- HELPER: API WRAPPER ---
def generate_with_retry(model_name, contents, config, retries=3):
    base_wait = 5
    for attempt in range(retries):
        try:
            return client.models.generate_content(model=model_name, contents=contents, config=config)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = base_wait * (2 ** attempt)
                print(f"      ⚠️  Quota Hit (429). Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"      ⚠️ API Error: {e}")
                break
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

def check_and_clean_storage():
    if not os.path.exists(OUTPUT_DIR): return
    pass 

def save_state(data):
    if not os.path.exists(CACHE_DIR): os.makedirs(CACHE_DIR)
    with open(CHECKPOINT_FILE, "w") as f: json.dump(data, f, indent=4)

def load_state():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f: return json.load(f)
    return None

# --- STAGE 1: AGGREGATOR ---
def load_all_inputs(target_path):
    files = []
    if os.path.isdir(target_path):
        for root, _, fs in os.walk(target_path):
            if "archive" not in root:
                files.extend([os.path.join(root, f) for f in fs if not f.startswith(".")])
    else: files.append(target_path)
    STATS["files_scanned"] = len(files)
    
    raw_urls = []
    processed_files = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                found = re.findall(r'(https?://[^\s<>"]+|www\.[^\s<>"]+)', content)
                raw_urls.extend(found)
            processed_files.append(fp)
        except: pass
    STATS["raw_urls"] = len(raw_urls)
    
    clean_urls = set()
    final_urls = []
    for u in raw_urls:
        if any(x in u.lower() for x in IGNORE_URL_TERMS): continue
        clean = urlunparse(urlparse(u)._replace(query="", fragment=""))
        if clean not in clean_urls:
            clean_urls.add(clean)
            final_urls.append(clean)
    STATS["unique_urls"] = len(final_urls)
    return final_urls, processed_files

# --- STAGE 2: SCRAPER (THE LAWNMOWER) ---
def clean_text(text):
    """
    Aggressively strips whitespace, newlines, and HTML noise.
    """
    if not text: return ""
    # 1. Replace multiple spaces/newlines with single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def fetch_article_text(url):
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
    if not os.path.exists(LIBRARY_DIR): os.makedirs(LIBRARY_DIR)
    cache_path = os.path.join(LIBRARY_DIR, f"{url_hash}.json")

    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f: return json.load(f)

    try:
        article = Article(url)
        article.download()
        article.parse()
        
        is_junk = any(t in article.title.lower() for t in IGNORE_HEADLINE_TERMS)
        
        # --- THE LAWNMOWER LOGIC ---
        # 1. Get text
        raw_text = article.text
        # 2. Clean noise
        cleaned_text = clean_text(raw_text)
        # 3. Hard Limit (1000 chars approx 20 lines)
        truncated_text = cleaned_text[:1000]
        
        if len(cleaned_text) > 1000:
            truncated_text += "..."

        data = {
            "url": url, 
            "headline": article.title, 
            "snippet": truncated_text[:200], # For embedding (even shorter)
            "full_text": truncated_text,     # For Dossier (max 1000 chars)
            "status": "junk" if is_junk else "scraped"
        }
        with open(cache_path, 'w') as f: json.dump(data, f)
        STATS["scraped_success"] += 1
        return data
    except: 
        STATS["scraped_failed"] += 1
        return None

# --- STAGE 3: CLUSTERER & REPORTER ---
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

    print("   -> Clustering...")
    cluster_engine = ContentClusterer(min_size=5, max_size=20)
    raw_clusters = cluster_engine.run_clustering(valid_articles, embeddings)

    # NAME THE CLUSTERS
    final_clusters = {"clusters": []}
    print("\n📊 --- FLIGHT MANIFEST ---")
    print(f"{'CLUSTER ID':<15} | {'COUNT':<5} | {'TOPIC NAME'}")
    print("-" * 50)
    
    for tag, arts in raw_clusters.items():
        if not arts: continue
        
        # Naming (Use first 3 headlines for context)
        sample = "\n".join([f"- {a['headline']}" for a in arts[:3]])
        naming_prompt = f"Create a 3-word Topic Title and a short file_slug for these headlines:\n{sample}\nOUTPUT JSON: {{'topic': '...', 'file_slug': '...'}}"
        
        try:
            resp = generate_with_retry(MODEL_NAME, naming_prompt, types.GenerateContentConfig(response_mime_type="application/json"), retries=1)
            if resp:
                j = json.loads(resp.text.replace("```json","").replace("```",""))
                topic_name = j.get("topic", tag)
                file_slug = j.get("file_slug", tag)
            else: raise Exception("Quota")
        except:
            topic_name = arts[0]['headline'][:30] + "..."
            file_slug = "News_Update"

        print(f"{tag:<15} | {len(arts):<5} | {topic_name}")
        
        # Build Dossier (Now much lighter due to 1000 char limit)
        full_dossier = ""
        cluster_sources = [] # Store title+url for final HTML
        
        for a in arts:
            full_dossier += f"HEADLINE: {a['headline']}\nFACTS: {a['full_text']}\n\n"
            cluster_sources.append({"title": a['headline'], "url": a['url']})
            
        final_clusters["clusters"].append({
            "topic": topic_name, 
            "file_slug": file_slug,
            "dossier": full_dossier, 
            "sources": cluster_sources 
        })
    print("-" * 50 + "\n")
    return final_clusters

# --- STAGE 4: PRODUCER ---
def generate_script(topic, dossier, cast_names, source_count):
    host1, host2 = cast_names
    safe_dossier = dossier[:30000] # Even safer limit
    
    if source_count <= 5: mode = "Briefing"; words = "600 words"
    else: mode = "Deep Dive"; words = "1200 words"

    print(f"      📝 Draft: {mode} ({source_count} items) - Host: {host1}")
    
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
async def produce_audio(topic, script, filename):
    global EDGE_TTS_ALIVE
    path = os.path.join(TODAY_OUTPUT_DIR, filename)
    combined = AudioSegment.empty()
    
    cast = VOICE_CAST["Default"]
    for k in VOICE_CAST:
        if k.lower() in topic.lower(): cast = VOICE_CAST[k]; break
        
    for line in script:
        spk = line.get("speaker", "Host")
        txt = line.get("text", "")
        voice = cast.get(spk, list(cast.values())[0])
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

# --- MAIN ---
async def main():
    print(f"ℹ️  Podcast Studio: {VERSION_ID}")
    
    urls, files = load_all_inputs(INPUT_DIR)
    
    print(f"   📂 Input Report:")
    print(f"      - Files Scanned: {STATS['files_scanned']}")
    print(f"      - Raw URLs:      {STATS['raw_urls']}")
    print(f"      - Unique URLs:   {STATS['unique_urls']}")
    
    articles = [fetch_article_text(u) for u in urls]
    
    print(f"      - Scraped OK:    {STATS['scraped_success']}")
    print(f"      - Scraped Fail:  {STATS['scraped_failed']}")
    
    data = ai_cluster_and_summarize(articles)
    if not data: return

    # Production & HTML Generation
    index_items = []
    
    for c in data["clusters"]:
        fname = f"{TODAY_STR}_{c['file_slug']}.mp3"
        print(f"\n📺 Studio: Producing '{c['topic']}'")
        script = generate_script(c["topic"], c["dossier"], ["Alice", "Bob"], len(c["sources"]))
        if script:
            await produce_audio(c["topic"], script, fname)
            index_items.append({
                "topic": c['topic'],
                "filename": fname,
                "sources": c['sources']
            })

    # GENERATE HTML INDEX
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
        
        with open(os.path.join(TODAY_OUTPUT_DIR, "index.html"), "w") as f:
            f.write(html)
        print(f"\n✅ Index Generated: {os.path.join(TODAY_OUTPUT_DIR, 'index.html')}")

if __name__ == "__main__":
    if not os.path.exists(TODAY_OUTPUT_DIR): os.makedirs(TODAY_OUTPUT_DIR)
    asyncio.run(main())