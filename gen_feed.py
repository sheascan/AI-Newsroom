import os
import subprocess
import time
import shutil
from datetime import datetime
from email.utils import formatdate

# --- CONFIGURATION ---
PROJECT_DIR = os.path.expanduser("~/Dropbox/Podcast_Studio")
DROPBOX_ROOT = os.path.expanduser("~/Dropbox")
RSS_FILE = os.path.join(PROJECT_DIR, "feed.xml")

PODCAST_TITLE = "The Daily Briefing"
PODCAST_DESC = "AI-generated news summaries."
PODCAST_URL = "http://localhost"
PODCAST_IMAGE = "https://via.placeholder.com/1400" 

def get_existing_links_map():
    """Runs 'maestral sharelink list' to cache existing links."""
    print("⏳ Fetching existing Maestral links...")
    links_map = {}
    try:
        output = subprocess.check_output(['maestral', 'sharelink', 'list'], stderr=subprocess.STDOUT).decode('utf-8').strip()
        for line in output.split('\n'):
            if "http" in line:
                parts = line.split()
                for part in parts:
                    if part.startswith("http"):
                        url = part.replace("dl=0", "dl=1")
                        path = line.split(part)[0].strip()
                        if not path.startswith("/"): path = "/" + path
                        links_map[path] = url
                        break
    except Exception:
        pass 
    return links_map

def wait_for_maestral_index(local_path, timeout=30):
    """
    Polls 'maestral filestatus' until the file is recognized.
    Returns True if found, False if timed out.
    """
    print(f"         ↳ Waiting for Maestral to index...", end="", flush=True)
    for _ in range(timeout):
        try:
            status = subprocess.check_output(
                ['maestral', 'filestatus', local_path], 
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip().lower()
            
            # If status is valid (syncing or up to date), we are good
            if "up to date" in status or "syncing" in status:
                print(" ✅ Ready.")
                time.sleep(2) # Safety buffer
                return True
        except subprocess.CalledProcessError:
            pass # File not found yet by daemon
            
        time.sleep(1)
        print(".", end="", flush=True)
        
    print(" ❌ Timeout.")
    return False

def heal_zombie_file(full_path):
    """
    Clones file to force new ID, then WAITS for indexing.
    """
    directory = os.path.dirname(full_path)
    filename = os.path.basename(full_path)
    name, ext = os.path.splitext(filename)
    
    # Simple clean rename logic
    if "_cloned" in name:
        new_name = f"{name}{ext}" 
    else:
        new_name = f"{name}_cloned{ext}"
        
    new_full_path = os.path.join(directory, new_name)
    
    print(f"      🚑 ZOMBIE LINK DETECTED.")
    print(f"         ↳ Cloning to: {new_name}")
    
    try:
        shutil.copy2(full_path, new_full_path)
        os.remove(full_path) 
        
        # CRITICAL: Wait for Maestral to actually see the new file
        if wait_for_maestral_index(new_full_path):
            return new_full_path
        else:
            print(f"      ❌ Error: Maestral took too long to see the new file.")
            return None
            
    except OSError as e:
        print(f"      ❌ Clone failed: {e}")
        return None

def get_maestral_link(full_path, existing_links):
    rel_path = os.path.relpath(full_path, DROPBOX_ROOT)
    dropbox_path = "/" + rel_path if not rel_path.startswith("/") else rel_path
    
    if dropbox_path in existing_links:
        return existing_links[dropbox_path], full_path

    print(f"      🔗 Creating new link for: {dropbox_path}")
    try:
        result = subprocess.check_output(['maestral', 'sharelink', 'create', dropbox_path], stderr=subprocess.STDOUT).decode('utf-8').strip()
        if "http" in result:
            return result.replace("dl=0", "dl=1"), full_path
    except subprocess.CalledProcessError as e:
        err_msg = e.output.decode('utf-8').strip()
        
        if "already exists" in err_msg:
            # TRIGGER CLONE & WAIT
            new_path = heal_zombie_file(full_path)
            if new_path:
                return get_maestral_link(new_path, existing_links)
            
        print(f"      ❌ Create Error: {err_msg}")
            
    return None, full_path

def generate_rss():
    print(f"📂 Scanning Recursively: {PROJECT_DIR}")
    if not os.path.exists(PROJECT_DIR): return

    existing_links = get_existing_links_map()
    print(f"   ℹ️  Found {len(existing_links)} existing shared links.")

    episodes = []
    
    for root, dirs, files in os.walk(PROJECT_DIR):
        for file in files:
            if file.endswith(".mp3"):
                full_path = os.path.join(root, file)
                
                link, final_path = get_maestral_link(full_path, existing_links)
                
                if link:
                    stat = os.stat(final_path)
                    final_filename = os.path.basename(final_path)
                    print(f"   ✅ Linked: {final_filename}")
                    
                    # Clean Title Display
                    clean_title = final_filename.replace(".mp3", "").replace("_", " ")
                    clean_title = clean_title.replace("cloned", "").replace("final", "").replace("v2", "").strip()
                    
                    episodes.append({
                        "title": clean_title,
                        "link": link,
                        "size": stat.st_size,
                        "date": formatdate(stat.st_mtime),
                    })
                else:
                    print(f"   ⚠️ Failed to Link: {file}")

    if not episodes:
        print("⚠️ No episodes found.")
        return

    episodes.sort(key=lambda x: datetime.strptime(x['date'], '%a, %d %b %Y %H:%M:%S %z'), reverse=True)

    rss = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
  <channel>
    <title>{PODCAST_TITLE}</title>
    <description>{PODCAST_DESC}</description>
    <link>{PODCAST_URL}</link>
    <language>en-us</language>
    <itunes:image href="{PODCAST_IMAGE}"/>
"""
    for ep in episodes:
        rss += f"""    <item>
      <title>{ep['title']}</title>
      <enclosure url="{ep['link']}" length="{ep['size']}" type="audio/mpeg"/>
      <guid>{ep['link']}</guid>
      <pubDate>{ep['date']}</pubDate>
    </item>
"""
    rss += "  </channel>\n</rss>"

    with open(RSS_FILE, "w") as f:
        f.write(rss)
    
    print(f"\n🎉 RSS Updated with {len(episodes)} episodes.")
    print(f"   XML File: {RSS_FILE}")

if __name__ == "__main__":
    generate_rss()