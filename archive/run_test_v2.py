import os
import shutil
import subprocess
from datetime import datetime
from feedgen.feed import FeedGenerator
from pydub import AudioSegment

# --- CONFIGURATION ---
GITHUB_USER = "sheascan"
REPO_NAME = "my-daily-briefing"
TEST_FOLDER_NAME = "test_experiment_v2"  # New folder to ensure clean start
XML_FILENAME = "debug_compliant.xml"

# Calculated Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GIT_HOSTING_DIR = os.path.join(BASE_DIR, "hosting")
TEST_DIR = os.path.join(GIT_HOSTING_DIR, TEST_FOLDER_NAME)
PUBLIC_URL_BASE = f"https://{GITHUB_USER}.github.io/{REPO_NAME}/{TEST_FOLDER_NAME}/"
FULL_FEED_URL = f"{PUBLIC_URL_BASE}{XML_FILENAME}"

def run_experiment():
    print("🧪 STARTING COMPLIANT CLEAN ROOM TEST...")
    
    # 1. Prepare the Test Folder in Git
    if not os.path.exists(GIT_HOSTING_DIR):
        print("❌ Error: 'hosting' folder missing. Please clone repo first.")
        return

    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)
    print(f"   📂 Created test folder: {TEST_DIR}")

    # 2. Generate Synthetic Audio
    print("   🎵 Generating synthetic audio...")
    audio_filename = "test_audio_compliant.mp3"
    audio_path = os.path.join(TEST_DIR, audio_filename)
    
    silence = AudioSegment.silent(duration=2000) 
    silence.export(audio_path, format="mp3")
    file_size = os.path.getsize(audio_path)

    # 3. Generate Validated XML Feed
    print("   XML Building compliant XML feed...")
    fg = FeedGenerator()
    fg.load_extension('podcast')
    
    # --- METADATA (VALIDATOR REQUIRED) ---
    fg.title(f"VALIDATOR TEST {datetime.now().strftime('%H:%M')}")
    fg.link(href=PUBLIC_URL_BASE, rel='alternate')
    # REQUIRED: Description > 50 chars
    fg.description("This is a technical test feed designed to pass strict RSS validation standards including Apple Podcasts and Pocket Casts requirements.")
    fg.language("en")
    
    # REQUIRED: Explicit Tag
    fg.podcast.itunes_explicit('no')
    
    # REQUIRED: Category
    fg.podcast.itunes_category('Technology')
    
    # REQUIRED: Image
    fg.podcast.itunes_image('https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png')
    
    # REQUIRED: Owner & Author (The missing piece!)
    fg.author(name="Debug Bot", email="debug@test.com")
    fg.podcast.itunes_author("Debug Bot")
    fg.podcast.itunes_owner(name="Debug Bot", email="debug@test.com")

    # REQUIRED: Atom Self Link
    fg.link(href=FULL_FEED_URL, rel='self')

    # Add Episode
    fe = fg.add_entry()
    fe.title("Compliance Check: Audio Test")
    fe.id(f"test_id_{datetime.now().timestamp()}")
    
    public_mp3_url = f"{PUBLIC_URL_BASE}{audio_filename}"
    fe.link(href=public_mp3_url)
    fe.enclosure(public_mp3_url, str(file_size), 'audio/mpeg')
    
    # Save
    fg.rss_file(os.path.join(TEST_DIR, XML_FILENAME))

    # 4. Push to GitHub
    print("   🚀 Pushing experiment to GitHub...")
    os.chdir(GIT_HOSTING_DIR)
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", "Deploy Compliant Test v2"], check=False)
    subprocess.run(["git", "push"], check=True)
    
    print("\n✅ EXPERIMENT DEPLOYED")
    print(f"   🔗 YOUR TEST URL: {FULL_FEED_URL}")
    print("   ⏳ Wait 60 seconds, then use this URL in Podba.se validator.")

if __name__ == "__main__":
    run_experiment()