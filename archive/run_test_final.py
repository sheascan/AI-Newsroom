import os
import shutil
import subprocess
from datetime import datetime
from feedgen.feed import FeedGenerator

# --- CONFIGURATION ---
GITHUB_USER = "sheascan"
REPO_NAME = "my-daily-briefing"
TEST_FOLDER_NAME = "test_v3_final" 
XML_FILENAME = "success.xml"  # <--- NEW NAME to bypass cache

# Calculated Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GIT_HOSTING_DIR = os.path.join(BASE_DIR, "hosting")
TEST_DIR = os.path.join(GIT_HOSTING_DIR, TEST_FOLDER_NAME)
PUBLIC_URL_BASE = f"https://{GITHUB_USER}.github.io/{REPO_NAME}/{TEST_FOLDER_NAME}/"
FULL_FEED_URL = f"{PUBLIC_URL_BASE}{XML_FILENAME}"

# We point to a REAL file that already exists in your repo
# (From your screenshot: 21Dec_0855_best_offer_ever.mp3)
REAL_AUDIO_URL = f"https://{GITHUB_USER}.github.io/{REPO_NAME}/21Dec/21Dec_0855_best_offer_ever.mp3"

def run_experiment():
    print("🧪 STARTING FINAL REAL-AUDIO TEST...")
    
    # 1. Prepare the Test Folder
    if not os.path.exists(GIT_HOSTING_DIR):
        print("❌ Error: 'hosting' folder missing.")
        return

    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)
    
    # 2. Generate XML Feed
    print("   XML Building feed pointing to REAL audio...")
    fg = FeedGenerator()
    fg.load_extension('podcast')
    
    # Metadata
    fg.title(f"FINAL TEST {datetime.now().strftime('%H:%M')}")
    fg.link(href=PUBLIC_URL_BASE, rel='alternate')
    fg.description("Testing with real audio content to bypass spam filters.")
    fg.language("en")
    
    # Compliance Tags
    fg.podcast.itunes_category('Technology')
    fg.podcast.itunes_explicit('no')
    fg.podcast.itunes_image('https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png')
    fg.podcast.itunes_owner(name="Mike M", email="mike@test.com")
    fg.podcast.itunes_author("Mike M")
    
    # Add Episode (Pointing to REAL Audio)
    fe = fg.add_entry()
    fe.title("Real Audio Test Episode")
    fe.id(f"final_test_{datetime.now().timestamp()}")
    fe.link(href=REAL_AUDIO_URL)
    
    # We fake the size/type since we know the file is good
    fe.enclosure(REAL_AUDIO_URL, "9818514", 'audio/mpeg') 
    
    # Save
    fg.rss_file(os.path.join(TEST_DIR, XML_FILENAME))

    # 3. Push to GitHub
    print("   🚀 Pushing to GitHub...")
    os.chdir(GIT_HOSTING_DIR)
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", "Deploy Final Test"], check=False)
    subprocess.run(["git", "push"], check=True)
    
    print("\n✅ DEPLOYED!")
    print(f"   🔗 URL: {FULL_FEED_URL}")
    print("   ✋ STOP! Do not paste this into Pocket Casts yet.")
    print("   ⏳ You MUST wait 5 minutes for GitHub to build.")

if __name__ == "__main__":
    run_experiment()