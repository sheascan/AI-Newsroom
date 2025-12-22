import os
import shutil
import json
import glob
import subprocess
from datetime import datetime, timezone
from feedgen.feed import FeedGenerator

# --- CONFIGURATION ---
GITHUB_USER = "sheascan"
REPO_NAME = "my-daily-briefing"
PUBLIC_URL_BASE = f"https://{GITHUB_USER}.github.io/{REPO_NAME}/"
FEED_FILENAME = "briefing.xml"  # The filename we know works

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "outputs")
GIT_HOSTING_DIR = os.path.join(BASE_DIR, "hosting")

# Metadata
FEED_CONFIG = {
    "title": "My AI Daily Briefing",
    "link": PUBLIC_URL_BASE,
    "description": "Your daily executive summary of the most important news, generated automatically by AI.",
    "author": "Mike M",
    "email": "mike@test.com",
    "language": "en",
    # We use the GitHub logo because we know it works. 
    # You can change this later to a custom URL if you want.
    "image": "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"
}

def sync_and_deploy():
    print(f"🔄 Syncing Real Audio to Git Folder...")
    
    if not os.path.exists(GIT_HOSTING_DIR):
        print("❌ Error: 'hosting' folder not found.")
        return

    # 1. Create .nojekyll
    nojekyll_path = os.path.join(GIT_HOSTING_DIR, ".nojekyll")
    if not os.path.exists(nojekyll_path):
        open(nojekyll_path, 'w').close()

    # 2. Copy Files (The Real Data)
    for item in os.listdir(LOCAL_OUTPUT_DIR):
        src_path = os.path.join(LOCAL_OUTPUT_DIR, item)
        dst_path = os.path.join(GIT_HOSTING_DIR, item)
        
        if os.path.isdir(src_path):
            if os.path.exists(dst_path):
                # Update existing folders with new files
                for f in os.listdir(src_path):
                    if not os.path.exists(os.path.join(dst_path, f)):
                        shutil.copy2(os.path.join(src_path, f), os.path.join(dst_path, f))
            else:
                shutil.copytree(src_path, dst_path)

    # 3. Generate RSS
    generate_rss(GIT_HOSTING_DIR)

    # 4. Push to Internet
    print("🚀 Pushing to GitHub...")
    os.chdir(GIT_HOSTING_DIR)
    
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", f"Update {datetime.now().strftime('%d %b %H:%M')}"], check=False)
    subprocess.run(["git", "push"], check=True)
    
    print(f"✅ DONE! Feed updated at: {PUBLIC_URL_BASE}{FEED_FILENAME}")

def generate_rss(target_dir):
    print("🎙️  Building RSS with Dates & Metadata...")
    fg = FeedGenerator()
    fg.load_extension('podcast')
    
    # Metadata
    fg.title(FEED_CONFIG["title"])
    fg.link(href=FEED_CONFIG["link"], rel='alternate')
    fg.description(FEED_CONFIG["description"])
    fg.language(FEED_CONFIG["language"])
    
    # COMPLIANCE TAGS (The "Special Sauce")
    fg.link(href=f"{PUBLIC_URL_BASE}{FEED_FILENAME}", rel='self') # The Self Link
    fg.podcast.itunes_category('Technology')
    fg.podcast.itunes_explicit('no')
    fg.podcast.itunes_image(FEED_CONFIG["image"]) # The Image
    fg.podcast.itunes_owner(name=FEED_CONFIG["author"], email=FEED_CONFIG["email"])
    fg.podcast.itunes_author(FEED_CONFIG["author"])

    # Find Episodes
    manifest_pattern = os.path.join(target_dir, "*", "manifest.json")
    manifests = glob.glob(manifest_pattern)
    # Sort by file modification time so newest is first
    manifests.sort(key=os.path.getmtime, reverse=True)

    for m_file in manifests:
        try:
            with open(m_file, 'r') as f: data = json.load(f)
            folder_name = os.path.basename(os.path.dirname(m_file))
            
            # Get the folder creation time to use as the pubDate
            # (This ensures older episodes have older dates)
            folder_stats = os.stat(os.path.dirname(m_file))
            creation_time = datetime.fromtimestamp(folder_stats.st_mtime, timezone.utc)

            for ep in data.get("episodes", []):
                fe = fg.add_entry()
                fe.title(ep['title'])
                fe.id(ep['filename']) # ID
                
                # THE CRITICAL FIX: PUBLICATION DATE
                fe.published(creation_time)
                
                safe_filename = ep['filename'].replace(" ", "%20")
                public_url = f"{PUBLIC_URL_BASE}{folder_name}/{safe_filename}"
                
                fe.link(href=public_url)
                
                local_path = os.path.join(target_dir, folder_name, ep['filename'])
                if os.path.exists(local_path):
                    fe.enclosure(public_url, str(os.path.getsize(local_path)), 'audio/mpeg')
                    
                fe.description(f"Generated on {creation_time.strftime('%Y-%m-%d')}")

        except Exception as e: print(f"⚠️ RSS Error: {e}")

    fg.rss_file(os.path.join(target_dir, FEED_FILENAME))

if __name__ == "__main__":
    sync_and_deploy()