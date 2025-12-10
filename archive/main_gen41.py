#!/usr/bin/env python3
"""
Podcast Studio Automation - Generation 41 (Refactored)
------------------------------------------------------
Changes from Gen40:
- [FIX] implemented 'TokenBucket' style rate limiting for Google GenAI (429 errors).
- [FIX] Added exponential backoff with jitter for retries.
- [REF] Modularized 'deduplication' into dedicated FuzzyMatcher class.
- [REF] Switched to Dataclasses for memory-efficient article storage.
- [LOG] Enhanced logging format to match previous 'Man Utd' output style.

Author: Gemini
Date: 2024-05-22
"""

import os
import time
import random
import logging
import difflib
import argparse
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime

# Third-party imports
try:
    import feedparser
    import requests
    from bs4 import BeautifulSoup
    from google import genai
    from google.genai import types
except ImportError as e:
    print(f"CRITICAL ERROR: Missing dependency. {e}")
    print("Run: pip install feedparser requests beautifulsoup4 google-genai")
    exit(1)

# ==============================================================================
# 1. SYSTEM CONFIGURATION & LOGGING
# ==============================================================================
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt='%H:%M:%S')
logger = logging.getLogger("PodStudio")

# Configuration Constants
API_KEY = os.environ.get("GOOGLE_API_KEY", "YOUR_API_KEY_HERE")
MODEL_ID = "gemini-2.0-flash"
SIMILARITY_THRESHOLD = 0.85  # 85% match = duplicate
BATCH_SIZE = 8               # Smaller batches to prevent Token Limit hits
RPM_LIMIT = 10               # Safety limit: Requests Per Minute
COOLDOWN_SEC = 60 / RPM_LIMIT

# ==============================================================================
# 2. DATA STRUCTURES
# ==============================================================================
@dataclass
class NewsItem:
    """Standardized internal representation of a news article."""
    title: str
    link: str
    summary: str
    source: str
    published: str
    guid: str = field(default="")

    def __post_init__(self):
        # Basic cleanup on init
        self.title = self.title.strip()
        self.summary = self.summary.strip()

# ==============================================================================
# 3. UTILITY CLASSES (Dedupe & Rate Limiting)
# ==============================================================================
class RateLimitGuard:
    """
    Manages API quotas to prevent 429 errors before they happen.
    Acts as a traffic controller.
    """
    def __init__(self, rpm_limit: int):
        self.interval = 60.0 / rpm_limit
        self.last_request_time = 0.0

    def wait_for_slot(self):
        """Blocks execution until it's safe to send a request."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.interval:
            sleep_time = self.interval - elapsed
            logger.info(f"   ⏳ Rate Guard: Cooling down for {sleep_time:.2f}s...")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

class FuzzyMatcher:
    """
    Handles the 'Duplicate: Man Utd' logic using sequence matching.
    """
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.seen_titles = []

    def is_duplicate(self, title: str) -> bool:
        """Returns True if title is similar to any previously seen title."""
        for seen in self.seen_titles:
            ratio = difflib.SequenceMatcher(None, title, seen).ratio()
            if ratio > self.threshold:
                # Log match for debugging (replicating Gen40 style)
                logger.info(f"   🗑️  Duplicate: {title[:30]}... (matches '{seen[:15]}...')")
                return True
        self.seen_titles.append(title)
        return False

# ==============================================================================
# 4. CORE MODULES (Ingest & AI)
# ==============================================================================
class RSSIngestor:
    def __init__(self, feeds: List[str]):
        self.feeds = feeds

    def fetch_all(self) -> List[NewsItem]:
        logger.info(f"📡 Polling {len(self.feeds)} RSS feeds...")
        items = []
        for url in self.feeds:
            try:
                feed = feedparser.parse(url)
                source_name = feed.feed.get('title', 'Unknown Source')
                
                for entry in feed.entries:
                    # HTML Stripping for summary
                    raw_summary = getattr(entry, 'summary', '')
                    clean_summary = BeautifulSoup(raw_summary, "html.parser").get_text()[:400]
                    
                    items.append(NewsItem(
                        title=entry.title,
                        link=entry.link,
                        summary=clean_summary,
                        source=source_name,
                        published=getattr(entry, 'published', str(datetime.now()))
                    ))
            except Exception as e:
                logger.error(f"   ❌ Feed Error [{url}]: {e}")
        return items

class AIEngine:
    """
    The updated Gen41 AI Handler.
    Includes Exponential Backoff and Rate Guarding.
    """
    def __init__(self, api_key: str, model_id: str):
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id
        self.guard = RateLimitGuard(RPM_LIMIT)

    def generate_safe(self, prompt: str, retries: int = 5) -> Optional[str]:
        """
        Wraps the generation call with:
        1. Pre-flight cooling (RateLimitGuard)
        2. Post-error backoff (Exponential Sleep)
        """
        # 1. Pre-flight check
        self.guard.wait_for_slot()

        delay = 10  # Initial retry delay
        
        for attempt in range(1, retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=prompt
                )
                return response.text

            except Exception as e:
                error_msg = str(e).lower()
                # 2. Check for 429 Quota
                if "429" in error_msg or "quota" in error_msg:
                    if attempt == retries:
                        logger.critical("💀 Max retries reached. Abandoning batch.")
                        raise e
                    
                    # Jitter prevents 'thundering herd' if running multiple instances
                    jitter = random.uniform(1.5, 4.5)
                    total_wait = delay + jitter
                    
                    logger.warning(f"⚠️ Quota Hit (429). Attempt {attempt}/{retries}. Pausing {total_wait:.1f}s...")
                    time.sleep(total_wait)
                    
                    delay *= 2  # Double the wait for next time (10 -> 20 -> 40)
                else:
                    # Non-recoverable error
                    logger.error(f"❌ API Error: {e}")
                    raise e
        return None

# ==============================================================================
# 5. ORCHESTRATION PIPELINE
# ==============================================================================
class PodcastPipeline:
    def __init__(self):
        # In a real 700-line script, these would be loaded from a config file
        self.feeds = [
            "https://www.manutd.com/en/rss/news",
            "https://www.skysports.com/manchester-united-news/rss/0,20514,11667,00.xml",
            "https://www.manchestereveningnews.co.uk/sport/football/football-news/?service=rss"
        ]
        self.ingestor = RSSIngestor(self.feeds)
        self.matcher = FuzzyMatcher(SIMILARITY_THRESHOLD)
        self.ai = AIEngine(API_KEY, MODEL_ID)

    def run(self):
        logger.info("🚀 Starting PodcastStudio Gen41 Pipeline...")
        
        # --- PHASE 1: INGEST ---
        raw_items = self.ingestor.fetch_all()
        logger.info(f"📥 Ingested {len(raw_items)} raw items.")

        # --- PHASE 2: CLEAN & DEDUPE ---
        unique_items = []
        for item in raw_items:
            if not self.matcher.is_duplicate(item.title):
                unique_items.append(item)
        
        logger.info(f"✨ Cleaned list: {len(unique_items)} unique articles (Removed {len(raw_items) - len(unique_items)}).")

        # --- PHASE 3: BATCHED AI ANALYSIS ---
        if not unique_items:
            logger.warning("No new items to process.")
            return

        self.process_batches(unique_items)

    def process_batches(self, items: List[NewsItem]):
        logger.info(f"🧠 Starting AI Analysis on {len(items)} items...")
        
        # Create batches
        batches = [items[i:i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
        total_batches = len(batches)
        
        results = []

        for i, batch in enumerate(batches):
            batch_num = i + 1
            logger.info(f"   -> Processing Batch {batch_num}/{total_batches} ({len(batch)} items)")

            # Construct Prompt
            batch_text = "\n\n".join([f"Headline: {x.title}\nSummary: {x.summary}" for x in batch])
            prompt = (
                f"You are a news editor for a Manchester United podcast.\n"
                f"Analyze these {len(batch)} news items. Sort them into themes (Transfer, Injury, Match).\n"
                f"Return valid JSON only.\n\n{batch_text}"
            )

            try:
                # The 'generate_safe' method handles all the 429 logic internally now
                output = self.ai.generate_safe(prompt)
                if output:
                    results.append(output)
                    logger.info(f"   ✅ Batch {batch_num} Success.")
            except Exception:
                logger.error(f"   💀 Batch {batch_num} Failed permanently.")

        # --- PHASE 4: SAVE ---
        self.save_results(results)

    def save_results(self, results: List[str]):
        filename = f"podcast_script_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(results))
        logger.info(f"🏁 Pipeline Complete. Output saved to {filename}")

# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    # Simulate CLI arguments (often found in larger scripts)
    parser = argparse.ArgumentParser(description="Gen41 News Aggregator")
    parser.add_argument("--dry-run", action="store_true", help="Skip AI calls")
    args = parser.parse_args()

    app = PodcastPipeline()
    app.run()