import os
import shutil
import subprocess
from datetime import datetime
from feedgen.feed import FeedGenerator
from pydub import AudioSegment

# --- CONFIGURATION ---
# We use a specific subfolder so we don't touch your real podcast
GITHUB_USER = "sheascan"
REPO_NAME = "my-daily-briefing"
TEST_FOLDER_NAME = "test_experiment"

# Calculated Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GIT_HOSTING_DIR = os.path.join(BASE_DIR, "hosting")
TEST_DIR = os.path.join(GIT_HOSTING_DIR, TEST_FOLDER_NAME)
PUBLIC_URL_BASE = f"https://{GITHUB_USER}.github.io/{REPO_NAME}/{TEST_FOLDER_NAME}/"

def run_experiment():
    print("🧪 STARTING DISCRETE TEST EXPERIMENT...")
    
    # 1. Prepare the Test Folder in Git
    if not os.path.exists(GIT_HOSTING_DIR):
        print("❌ Error: 'hosting' folder missing. Please clone repo first.")
        return

    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)
    print(f"   📂 Created test folder: {TEST_DIR}")

    # 2. Generate Synthetic Audio (1 Second of Silence)
    # We use pydub because we know you have it installed from main_gen139.py
    print("   🎵 Generating synthetic audio...")
    audio_filename = "test_audio_v1.mp3"
    audio_path = os.path.join(TEST_DIR, audio_filename)
    
    silence = AudioSegment.silent(duration=2000) # 2 seconds
    silence.export(audio_path, format="mp3")
    file_size = os.path.getsize(audio_path)

    # 3. Generate a Fresh XML Feed
    # We call it 'debug.xml' so Pocket Casts has never seen it before
    print("   XML Building fresh XML feed...")
    fg = FeedGenerator()
    fg.load_extension('podcast')
    
    # Metadata - intentionally different from your main feed
    fg.title(f"DEBUG TEST {datetime.now().strftime('%H:%M')}")
    fg.link(href=PUBLIC_URL_BASE, rel='alternate')
    fg.description("This is a temporary technical test feed.")
    fg.language("en")
    
    # Required Tags
    fg.podcast.itunes_category('Technology')
    fg.podcast.itunes_explicit('no')
    fg.podcast.itunes_image('https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png')
    fg.podcast.itunes_owner(name="Debug Bot", email="debug@test.com")

    # Add the single episode
    fe = fg.add_entry()
    fe.title("System Check: Audio Test")
    fe.id(f"test_id_{datetime.now().timestamp()}") # 100% Unique ID
    
    public_mp3_url = f"{PUBLIC_URL_BASE}{audio_filename}"
    fe.link(href=public_mp3_url)
    fe.enclosure(public_mp3_url, str(file_size), 'audio/mpeg')
    
    # Save as 'debug.xml'
    fg.rss_file(os.path.join(TEST_DIR, "debug.xml"))

    # 4. Push to GitHub
    print("   🚀 Pushing experiment to GitHub...")
    os.chdir(GIT_HOSTING_DIR)
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", "Deploy Clean Room Test"], check=False)
    subprocess.run(["git", "push"], check=True)
    
    print("\n✅ EXPERIMENT SUCCESSFUL")
    print(f"   🔗 YOUR TEST URL: {PUBLIC_URL_BASE}debug.xml")
    print("   (Wait 60 seconds, then search this URL in Pocket Casts)")

if __name__ == "__main__":
    run_experiment()