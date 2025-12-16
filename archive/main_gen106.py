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
import numpy as np
from urllib.parse import urlparse, urlunparse
from email import policy
from pydub import AudioSegment
from newspaper import Article
from google import genai
from google.genai import types

# --- NEW DEPENDENCIES FOR GEN 105 ---
try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_distances
except ImportError:
    print("❌ Error: Missing Science Libraries.")
    print("   Please run: pip install scikit-learn numpy")
    sys.exit(1)

# --- VERSION TRACKER ---
VERSION_ID = "Gen 105.1 (Semantic Core + Fix)"

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
CHECKPOINT_FILE = os.path.join(CACHE_DIR, "gen105_clusters.json")

# Date Strings
TODAY_STR = datetime.datetime.now().strftime("%d%b")
TODAY_OUTPUT_DIR = os.path.join(OUTPUT_DIR, TODAY_STR)
TODAY_INPUT_ARCHIVE = os.path.join(INPUT_DIR, "archive", TODAY_STR)

# Storage Limit (MB)
STORAGE_LIMIT_MB = 1024

# --- CIRCUIT BREAKER STATE ---
EDGE_TTS_ALIVE = True 

# --- FILTERING CONFIG ---
IGNORE_URL_TERMS = [
    "unsubscribe", "manage your emails", "view in browser", "privacy policy", 
    "terms of service", "manage_preferences", "opt_out", "login", "signup", "signin",
    "facebook.com", "twitter.com", "instagram.com", "linkedin.com", "youtube.com",
    "google.com", "apple.com", "spotify.com", "tiktok.com",
    "help center", "contact us", "advertisement", "click here", "arxiv.org",
]

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
MODEL_NAME = "gemini-2.0-flash-lite" 
EMBEDDING_MODEL = "text-embedding-004" 

SLEEP_TIME = 5 

# VOICE CAST
VOICE_CAST = {
    "Politics": {"Sarah": "en-GB-SoniaNeural", "Mike": "en-US-GuyNeural"},
    "Technology": {"Tasha": "en-AU-NatashaNeural", "Chris": "en-US-ChristopherNeural"},
    "Sport": {"Connor": "en-IE-ConnorNeural", "Ryan": "en-GB-RyanNeural"},
    "Default": {"Alice": "en-US-AriaNeural", "Bob": "en-GB-RyanNeural"} 
}

# --- CLASS: CONTENT CLUSTERER (THE NEW BRAIN) ---
class ContentClusterer:
    """
    Uses Hierarchical Agglomerative Clustering to group articles.
    Enforces min_size=5 and max_size=20 via 'Gravity Merge'.
    """
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
        
        # Iterative cleanup
        for _ in range(10):
            unique_labels, counts = np.unique(refined_labels, return_counts=True)
            sizes = dict(zip(unique_labels, counts))
            
            # Find clusters that are too small
            orphans = [l for l, s in sizes.items() if s < self.min_size]
            if not orphans:
                break 

            centroids = self.get_centroids(embeddings, refined_labels)
            orphans.sort(key=lambda l: sizes[l]) # Deal with smallest first
            
            orphan_label = orphans[0]
            orphan_centroid = centroids[orphan_label].reshape(1, -1)
            
            best_target = None
            min_dist = float('inf')
            
            # Find best parent cluster
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
            else:
                break

        return refined_labels

    def run_clustering(self, articles, embeddings):
        n_articles = len(embeddings)
        n_clusters_initial = max(1, int(n_articles / 12))
        
        print(f"      ⚗️  Math: Grouping {n_articles} items into approx {n_clusters_initial} topics...")
        
        # 1. Base Clustering (Hierarchical)
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters_initial, 
            metric='cosine', 
            linkage='average'
        )
        initial_labels = clusterer.fit_predict(embeddings)
        
        # 2. Enforce Size Constraints
        final_labels = self.merge_orphans(embeddings, initial_labels)
        
        # 3. Organize Output
        clustered_data = {}
        unique_labels = np.unique(final_labels)
        
        for label in unique_labels:
            indices = np.where(final_labels == label)[0]
            cluster_articles = [articles[i] for i in indices]
            tag = f"Topic_Group_{label}"
            clustered_data[tag] = cluster_articles
            
        return clustered_data

# --- HELPER: API RATE LIMIT WRAPPER ---
def generate_with_retry(model_name, contents, config, retries=5):
    base_wait = 10
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
                print(f"      ⚠️  Quota Hit. Pausing for {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"      ⚠️ API Error: {e}. Retrying...")
                time.sleep(5)
    return None

# --- HELPER: EMBEDDING GENERATOR (FIXED) ---
def get_embeddings_batch(texts):
    """Generates embeddings for a list of texts using Google GenAI."""
    try:
        # Batching: We send the list directly.
        # This is faster and avoids the 'embedding' vs 'embeddings' attribute error.
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type="CLUSTERING")
        )
        
        # The response.embeddings is a list of ContentEmbedding objects.
        # We need to extract the '.values' from each one.
        all_embeddings = [e.values for e in result.embeddings]
        
        return np.array(all_embeddings)
    except Exception as e:
        print(f"      ❌ Embedding Error: {e}")
        return None

# --- HELPER: AUDIO & FILES ---
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
    except: pass

def check_and_clean_storage():
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(DATA_DIR):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp): total_size += os.path.getsize(fp)
    
    total_mb = total_size / (1024 * 1024)
    print(f"\n💾 Storage Status: {total_mb:.2f} MB / {STORAGE_LIMIT_MB} MB")
    
    if total_mb > STORAGE_LIMIT_MB:
        print("⚠️  Limit Exceeded! Cleaning old output folders...")
        day_folders = []
        if os.path.exists(OUTPUT_DIR):
            for d in os.listdir(OUTPUT_DIR):
                path = os.path.join(OUTPUT_DIR, d)
                if os.path.isdir(path): day_folders.append(path)
        
        day_folders.sort(key=os.path.getmtime)
        if day_folders:
            try: shutil.rmtree(day_folders[0])
            except: pass

def save_state(data):
    if not os.path.exists(CACHE_DIR): os.makedirs(CACHE_DIR)
    with open(CHECKPOINT_FILE, "w") as f: json.dump(data, f, indent=4)

def load_state():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f: return json.load(f)
    return None

def generate_audit_report(all_articles_metadata):
    if not os.path.exists(TODAY_OUTPUT_DIR): os.makedirs(TODAY_OUTPUT_DIR)
    report_path = os.path.join(TODAY_OUTPUT_DIR, f"Filtering_Report_{TODAY_STR}.html")
    
    kept = [a for a in all_articles_metadata if a['status'] == 'kept']
    junk = [a for a in all_articles_metadata if a['status'] == 'junk']
    
    html = f"""<html><body><h1>Report {TODAY_STR}</h1>
    <p>Kept: {len(kept)} | Junk: {len(junk)}</p>
    <table border=1><tr><th>Status</th><th>Headline</th></tr>"""
    for item in all_articles_metadata:
        html += f"<tr><td>{item['status']}</td><td>{item.get('headline','')}</td></tr>"
    html += "</table></body></html>"
    
    with open(report_path, "w", encoding='utf-8') as f: f.write(html)

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
    
    for filepath in files_to_process:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                urls = re.findall(r'(https?://[^\s<>"]+|www\.[^\s<>"]+)', content)
                master_raw_urls.extend(urls)
            processed_files.append(filepath)
        except: pass

    clean_urls_set = set()
    final_urls = []
    
    for url in master_raw_urls:
        try:
            u_lower = url.lower()
            if any(t in u_lower for t in IGNORE_URL_TERMS): continue
            parsed = urlparse(url)
            clean_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))
            if clean_url not in clean_urls_set:
                clean_urls_set.add(clean_url)
                final_urls.append(clean_url) 
        except: continue

    return final_urls, processed_files

# --- STAGE 2: THE SCRAPER ---
def fetch_article_text(url):
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
    if not os.path.exists(LIBRARY_DIR): os.makedirs(LIBRARY_DIR)
    cache_path = os.path.join(LIBRARY_DIR, f"{url_hash}.json")

    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f: return json.load(f)
        except: pass

    try:
        article = Article(url)
        article.download()
        article.parse()
        headline = article.title
        
        is_junk = False
        reject_reason = ""
        for term in IGNORE_HEADLINE_TERMS:
            if term.lower() in headline.lower():
                is_junk = True; reject_reason = term; break

        snippet = f"{headline}. {article.text[:300]}..."
        full_text = article.text[:2500].replace("\n", " ")
        
        data = {
            "url": url, "headline": headline, "snippet": snippet, "full_text": full_text,
            "status": "junk" if is_junk else "scraped", "reject_reason": reject_reason
        }
        with open(cache_path, 'w', encoding='utf-8') as f: json.dump(data, f, indent=4)
        return data
    except: return None

# --- STAGE 3: THE SEMANTIC CLUSTERER (MAJOR UPGRADE) ---
def ai_cluster_and_summarize(article_objects):
    
    # 1. Deduplication & Cleanup
    print("   -> Running Cleanup & Deduplication...")
    unique_articles = []
    seen_headlines = []
    all_metadata = []

    for obj in article_objects:
        if not obj: continue
        
        # Late-check junk
        if obj['status'] != 'junk':
            for term in IGNORE_HEADLINE_TERMS:
                if term.lower() in obj['headline'].lower():
                    obj['status'] = 'junk'; break
        
        if obj['status'] == 'junk':
            all_metadata.append(obj)
            continue

        # Dedupe
        is_dupe = False
        h_lower = obj['headline'].lower()
        for seen in seen_headlines:
            if difflib.SequenceMatcher(None, h_lower, seen).ratio() > 0.8:
                is_dupe = True; obj['status'] = 'duplicate'; break
        
        if not is_dupe:
            seen_headlines.append(h_lower)
            obj['status'] = 'kept'
            unique_articles.append(obj)
        all_metadata.append(obj)

    generate_audit_report(all_metadata)
    
    if not unique_articles: return None

    # 2. Embedding Generation (The "Eyes")
    print(f"   -> Generating Embeddings for {len(unique_articles)} articles...")
    headlines_for_embed = [f"{a['headline']}: {a['snippet']}" for a in unique_articles]
    
    # Call the fixed batch function
    embeddings = get_embeddings_batch(headlines_for_embed)
    
    if embeddings is None:
        print("   ❌ Failed to generate embeddings. Aborting.")
        return None

    # 3. Clustering (The "Brain")
    print("   -> Running Agglomerative Clustering...")
    cluster_engine = ContentClusterer(min_size=5, max_size=20)
    clustered_data = cluster_engine.run_clustering(unique_articles, embeddings)

    # 4. Cluster Naming & Formatting (The "Labeler")
    final_clusters = {"clusters": []}
    
    print(f"   -> Processing {len(clustered_data)} resulting groups...")
    
    for tag, articles in clustered_data.items():
        if not articles: continue
        
        # We need an LLM just to name the topic and create the file slug
        sample_headlines = "\n".join([f"- {a['headline']}" for a in articles[:5]])
        
        naming_prompt = f"""
        Review these headlines and generate a short, punchy Podcast Topic Title (max 4 words).
        Also generate a filename_slug.
        HEADLINES:
        {sample_headlines}
        OUTPUT JSON: {{"topic": "Tech Market Crash", "file_slug": "Tech_Crash"}}
        """
        
        topic_name = tag
        file_slug = tag
        
        try:
            resp = generate_with_retry(MODEL_NAME, naming_prompt, types.GenerateContentConfig(response_mime_type="application/json"))
            if resp and resp.text:
                j = json.loads(resp.text.replace("```json","").replace("```",""))
                topic_name = j.get("topic", tag)
                file_slug = j.get("file_slug", tag)
        except: pass

        # Build Dossier
        full_dossier = ""
        source_urls = []
        for a in articles:
            full_dossier += f"HEADLINE: {a['headline']}\nCONTENT: {a['full_text']}\n\n"
            source_urls.append(a['url'])
        
        final_clusters["clusters"].append({
            "topic": topic_name,
            "file_slug": file_slug,
            "dossier": full_dossier,
            "source_urls": source_urls
        })
    
    return final_clusters

# --- STAGE 4: THE PRODUCER ---
def generate_script(topic, dossier, cast_names, source_count):
    # Vault Check
    script_hash = hashlib.md5(dossier.encode('utf-8')).hexdigest()[:10]
    safe_topic = re.sub(r'[^\w]+', '_', topic)[:20]
    vault_path = os.path.join(SCRIPT_DIR, f"{TODAY_STR}_{safe_topic}_{script_hash}.json")
    
    if os.path.exists(vault_path):
        print("      📜 Found Script in Vault.")
        try:
            with open(vault_path, 'r') as f: return json.load(f)
        except: pass

    host1, host2 = cast_names
    
    # Adjust logic for cluster sizes (now guaranteed > 5 usually)
    if source_count <= 5:
        mode = "Quick Briefing"; words = "800 words"
    elif source_count <= 12:
        mode = "Standard Show"; words = "1200 words"
    else:
        mode = "Deep Dive Special"; words = "1800 words"

    print(f"      📝 Writing Script ({mode}, {source_count} sources)...")
    
    prompt = f"""
    Write a {mode} podcast script ({words}).
    TOPIC: {topic}
    HOSTS: {host1} (Main), {host2} (Color/Support).
    STYLE: Professional, engaging, slightly witty.
    INSTRUCTIONS: 
    - Synthesize the DOSSIER into a cohesive narrative. 
    - Don't just list news items; connect them.
    - If there are many items, group them by sub-theme.
    OUTPUT JSON: [ {{"speaker": "{host1}", "text": "..."}}, {{"speaker": "{host2}", "text": "..."}} ]
    DOSSIER: {dossier[:100000]} 
    """ # Truncate dossier just in case of context limits
    
    response = generate_with_retry(
        MODEL_NAME, 
        prompt, 
        types.GenerateContentConfig(response_mime_type="application/json")
    )
    
    if response and response.text:
        try:
            clean = response.text.replace("```json", "").replace("```", "").strip()
            # Stabilizer
            if "[" in clean and "]" in clean:
                clean = clean[clean.find('['):clean.rfind(']')+1]
            data = json.loads(clean)
            
            if not os.path.exists(SCRIPT_DIR): os.makedirs(SCRIPT_DIR)
            with open(vault_path, 'w') as f: json.dump(data, f, indent=4)
            return data
        except Exception as e:
            print(f"      ⚠️ JSON Error: {e}")
    return []

# --- STAGE 5: AUDIO ---
async def produce_audio(topic, script, filename):
    global EDGE_TTS_ALIVE 
    if not os.path.exists(TODAY_OUTPUT_DIR): os.makedirs(TODAY_OUTPUT_DIR)
    filepath = os.path.join(TODAY_OUTPUT_DIR, filename)
    
    cast_dict = VOICE_CAST["Default"]
    for key in VOICE_CAST:
        if key.lower() in topic.lower(): cast_dict = VOICE_CAST[key]; break
    
    combined = AudioSegment.empty()
    
    for i, line in enumerate(script):
        speaker = line.get("speaker", "Host")
        text = line.get("text", "")
        voice = cast_dict.get(speaker, list(cast_dict.values())[0])
        
        segment = None
        
        # EdgeTTS
        if EDGE_TTS_ALIVE:
            try:
                communicate = edge_tts.Communicate(text, voice)
                temp = os.path.join(CACHE_DIR, f"temp_{i}.mp3")
                await communicate.save(temp)
                segment = AudioSegment.from_mp3(temp)
                os.remove(temp)
            except:
                print("      🚫 EdgeTTS Failed. Circuit Breaker Open.")
                EDGE_TTS_ALIVE = False
        
        # Fallback
        if segment is None and HAS_GTTS:
            try:
                tts = gTTS(text=text.replace("*",""), lang='en')
                temp = os.path.join(CACHE_DIR, f"temp_g_{i}.mp3")
                tts.save(temp)
                segment = AudioSegment.from_mp3(temp)
                os.remove(temp)
            except: pass
            
        if segment: combined += segment + AudioSegment.silent(duration=400)
        else: print(f"      ❌ Line {i} dropped.")

    # Music Mix
    if os.path.exists(MUSIC_DIR):
        tracks = [f for f in os.listdir(MUSIC_DIR) if f.endswith(".mp3")]
        if tracks:
            bg_music = AudioSegment.empty()
            while len(bg_music) < len(combined) + 10000:
                t_name = random.choice(tracks)
                update_music_stats(t_name)
                t = AudioSegment.from_mp3(os.path.join(MUSIC_DIR, t_name))
                bg_music += normalize_volume(t)
            
            final = bg_music[:len(combined)+2000].fade_out(3000).overlay(combined)
            final.export(filepath, format="mp3")
            attach_cover_art(filepath, topic)
            return filename
    
    combined.export(filepath, format="mp3")
    return filename

# --- MAIN ---
async def main():
    print(f"ℹ️  Podcast Studio: {VERSION_ID}")
    check_and_clean_storage()

    target = INPUT_DIR
    resume = False
    if len(sys.argv) > 1 and "RESUME" in str(sys.argv[1]).upper(): resume = True
    elif len(sys.argv) > 1: target = sys.argv[1]

    data = None
    if resume:
        data = load_state()
    else:
        urls, processed_files = load_all_inputs(target)
        if urls:
            articles = [fetch_article_text(u) for u in urls]
            data = ai_cluster_and_summarize(articles)
            if data: save_state(data)

    if not data: return

    index = []
    for c in data.get("clusters", []):
        topic = c["topic"]
        fname = f"{TODAY_STR}_{c['file_slug']}.mp3"
        print(f"\n📺 Producing: {topic}")
        
        # Determine host names based on topic keyword
        cast_dict = VOICE_CAST["Default"]
        for key in VOICE_CAST:
            if key.lower() in topic.lower(): cast_dict = VOICE_CAST[key]; break
        cast_names = list(cast_dict.keys())

        script = generate_script(topic, c["dossier"], cast_names, len(c["source_urls"]))
        if script:
            out_file = await produce_audio(topic, script, fname)
            if out_file:
                index.append({"topic": topic, "filename": out_file, "source_urls": c["source_urls"]})

    # Generate Index HTML
    if index:
        html = f"<html><body><h1>Briefing {TODAY_STR}</h1>"
        for i in index:
            html += f"<h2>{i['topic']}</h2><audio controls src='{i['filename']}'></audio><br>"
        with open(os.path.join(TODAY_OUTPUT_DIR, "index.html"), "w") as f: f.write(html+"</body></html>")
        
    if not resume: archive_inputs(processed_files)

if __name__ == "__main__":
    asyncio.run(main())