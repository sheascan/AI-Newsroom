#!/bin/bash

# --- CONFIGURATION ---
STUDIO_DIR="$HOME/podcast_studio"
GENERATOR_SCRIPT="$STUDIO_DIR/main.py"
RSS_SCRIPT="$STUDIO_DIR/gen_feed.py"

# NOW SIMPLE AND CLEAN:
DROPBOX_DIR="$HOME/Dropbox/Podcast_Studio"

# Date Format
TODAY=$(date +"%d%b")
LOCAL_OUTPUT_DIR="$STUDIO_DIR/data/outputs/$TODAY"
DROPBOX_TARGET="$DROPBOX_DIR/$TODAY"

echo "========================================"
echo "🎙️  STARTING STUDIO PIPELINE: $TODAY"
echo "========================================"

# 0. ENSURE MAESTRAL IS RUNNING
maestral start > /dev/null 2>&1

# 1. ACTIVATE PYTHON
source "$STUDIO_DIR/venv/bin/activate"

# 2. RUN PRODUCTION
echo "--> 🏗️  Running Generator..."
python3 "$GENERATOR_SCRIPT"

# 3. CHECK FOR OUTPUTS
if [ ! -d "$LOCAL_OUTPUT_DIR" ]; then
    echo "⚠️  No output folder found for today."
    exit 0
fi

# 4. TRANSFER TO DROPBOX
echo "--> 🚀 Transferring to Dropbox..."
mkdir -p "$DROPBOX_TARGET"

count=0
for file in "$LOCAL_OUTPUT_DIR"/*.mp3; do
    if [ -e "$file" ]; then
        filename=$(basename "$file")
        cp "$file" "$DROPBOX_TARGET/$filename"
        echo "    ✅ Copied: $filename"
        ((count++))
    fi
done

if [ $count -eq 0 ]; then
    echo "⚠️  No MP3s found."
    exit 0
fi

# 5. SYNC PAUSE (Short wait for the local move)
echo "--> ⏳ Syncing..."
sleep 5

# 6. UPDATE RSS
echo "--> 📡 Updating RSS Feed..."
python3 "$RSS_SCRIPT"

echo "========================================"
echo "✅ SUCCESS! $count episode(s) live."
echo "========================================"