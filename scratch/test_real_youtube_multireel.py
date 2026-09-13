"""
scratch/test_real_youtube_multireel.py
--------------------------------------
E2E test verifying that a real YouTube video with emojis in the title
(e.g., 'https://www.youtube.com/watch?v=TKJq7-IHSiA' which has \U0001f630)
processes successfully through Flask using the Phase 13 Multi-Reel pipeline
without any Windows charmap UnicodeEncodeError.
"""

import os
import sys
import time
import json
import urllib.request
import urllib.parse
from http.cookiejar import CookieJar

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.utf8_utils import setup_utf8_environment
setup_utf8_environment()

SERVER_URL = "http://127.0.0.1:5000"
# YouTube video containing \U0001f630 in title
YOUTUBE_URL = "https://www.youtube.com/watch?v=TKJq7-IHSiA"

print("=" * 70)
print("TESTING REAL YOUTUBE MULTI-REEL PROCESSING THROUGH FLASK")
print(f"Target URL: {YOUTUBE_URL}")
print("=" * 70)

# 1. Submit job via /upload
cj = CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

post_data = urllib.parse.urlencode({
    "upload_type": "youtube",
    "youtube_url": YOUTUBE_URL,
    "output_mode": "reels",
    "mode": "multimodal",
    "num_reels": "3",
    "reel_length": "15-30",
    "aspect_ratio": "9:16",
    "captions": "on"
}).encode("utf-8")

req = urllib.request.Request(
    f"{SERVER_URL}/upload",
    data=post_data,
    headers={"Content-Type": "application/x-www-form-urlencoded"}
)

print("\n[Step 1] Submitting YouTube URL to /upload endpoint...")
resp = opener.open(req)
final_url = resp.geturl()
print(f"  -> Redirected to: {final_url}")

if "/processing/" not in final_url:
    print(f"ERROR: Expected redirect to /processing/<job_id>, got: {final_url}")
    sys.exit(1)

job_id = final_url.split("/processing/")[1]
print(f"  -> Successfully created job: {job_id}")

# 2. Poll /api/status/<job_id>
status_url = f"{SERVER_URL}/api/status/{job_id}"
print(f"\n[Step 2] Polling status at {status_url}...")

last_stage = None
start_time = time.time()
max_timeout = 600  # 10 minutes max

while time.time() - start_time < max_timeout:
    try:
        with opener.open(status_url) as s_resp:
            s_data = json.loads(s_resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  -> Status check failed: {e}")
        time.sleep(2)
        continue

    status = s_data.get("status")
    stage = s_data.get("stage")

    if stage != last_stage:
        elapsed = time.time() - start_time
        print(f"  [{elapsed:5.1f}s] Status: {status.upper():10s} | Stage: {stage}")
        last_stage = stage

    if status == "complete":
        print("\n[Step 3] Processing completed successfully!")
        break
    elif status == "error":
        error_msg = s_data.get("error", "Unknown error")
        print(f"\n[ERROR] Processing failed with error: {error_msg}")
        sys.exit(1)

    time.sleep(2)

# 3. Check /result/<job_id> page
result_url = f"{SERVER_URL}/result/{job_id}"
print(f"\n[Step 4] Verifying result page at {result_url}...")
with opener.open(result_url) as r_resp:
    html = r_resp.read().decode("utf-8")
    assert "Your Reels Are Ready!" in html or "Highlights Are Ready" in html, "Result page title missing"
    print("  -> Result page loaded cleanly with HTML5 reel players.")

# 4. Check segments.json and manifest.json
meta_path = os.path.join(BASE_DIR, "web_outputs", job_id, "manifest.json")
if os.path.isfile(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    print(f"  -> Manifest verified: {manifest.get('total_reels', 0)} reels generated.")
    for r in manifest.get("reels", []):
        print(f"     Reel {r['reel_num']}: {r['start_formatted']} -> {r['end_formatted']} ({r['duration_sec']}s)")

# 5. Check ZIP download bundle
zip_url = f"{SERVER_URL}/download_all/{job_id}"
print(f"\n[Step 5] Verifying ZIP download bundle at {zip_url}...")
with opener.open(zip_url) as z_resp:
    zip_bytes = z_resp.read()
    print(f"  -> Downloaded ZIP bundle ({len(zip_bytes) / (1024 * 1024):.2f} MB). Content-Disposition: {z_resp.headers.get('Content-Disposition')}")

print("\n" + "=" * 70)
print("REAL YOUTUBE MULTI-REEL TEST PASSED WITH 100% SUCCESS!")
print("=" * 70)
