"""
scratch/test_youtube_e2e.py
---------------------------
Real End-to-End Testing of YouTube Highlight Generation through the live Flask application:
1. YouTube Video 1: "Python in 100 Seconds" in Multimodal Mode (Visual + Speech)
2. YouTube Video 1: "Python in 100 Seconds" in Baseline Mode (Visual Only)
3. YouTube Video 2: "Cookie Monster Sings C is for Cookie" in Multimodal Mode (Music/Singing)
4. Error Handling: Invalid URLs, Non-YouTube sources
"""

import os
import sys
import time
import json
import urllib.request
import urllib.parse

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FLASK_BASE_URL = "http://127.0.0.1:5000"


def submit_youtube_job(youtube_url: str, mode: str = "multimodal", budget: int = 15):
    print("\n" + "=" * 75)
    print(f"SUBMITTING YOUTUBE JOB VIA FLASK: {youtube_url}")
    print(f"Mode: {mode.upper()} | Budget: {budget}%")
    print("=" * 75)

    post_data = urllib.parse.urlencode({
        "upload_type": "youtube",
        "youtube_url": youtube_url,
        "mode": mode,
        "budget": str(budget)
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{FLASK_BASE_URL}/upload",
        data=post_data,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )

    # Note: urllib follows redirects by default or we can inspect response URL
    try:
        resp = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        # If redirected to /processing/<job_id>
        resp = e

    final_url = resp.geturl()
    print(f"Redirected to: {final_url}")
    job_id = final_url.split("/")[-1]
    print(f"Assigned Job ID: {job_id}")

    # Poll status API until completion
    print("\nPolling /api/status/<job_id> during live execution...")
    start_time = time.time()
    last_stage = None
    completed = False
    result_data = None

    while time.time() - start_time < 360:  # 6 minute max timeout
        status_url = f"{FLASK_BASE_URL}/api/status/{job_id}"
        with urllib.request.urlopen(status_url) as s_resp:
            data = json.loads(s_resp.read().decode("utf-8"))

        status = data.get("status")
        stage = data.get("stage")

        if stage != last_stage:
            print(f"  [{time.time() - start_time:5.1f}s] Stage: {stage}")
            last_stage = stage

        if status == "complete":
            completed = True
            break
        elif status == "error":
            raise RuntimeError(f"Job failed: {data.get('error')}")

        time.sleep(2.0)

    elapsed_time = time.time() - start_time
    assert completed, f"Job {job_id} timed out!"
    print(f"\nProcessing completed successfully in {elapsed_time:.1f} seconds!")

    # Verify result page
    res_url = f"{FLASK_BASE_URL}/result/{job_id}"
    with urllib.request.urlopen(res_url) as r_resp:
        html = r_resp.read().decode("utf-8")
        assert "Your Highlights Are Ready!" in html
        assert f"/media/{job_id}/highlight.mp4" in html
        assert f"/media/{job_id}/scores.png" in html
        assert f"/download/{job_id}" in html

    # Verify download route
    dl_url = f"{FLASK_BASE_URL}/download/{job_id}"
    with urllib.request.urlopen(dl_url) as dl_resp:
        video_bytes = dl_resp.read()
        content_disp = dl_resp.headers.get("Content-Disposition", "")
        file_size_mb = len(video_bytes) / (1024 * 1024)
        print(f"Download verified: {file_size_mb:.2f} MB (Content-Disposition: {content_disp[:60]}...)")

    # Read saved metadata
    meta_path = os.path.join(BASE_DIR, "web_outputs", job_id, "segments.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return {
        "job_id": job_id,
        "elapsed_seconds": round(elapsed_time, 1),
        "mode": mode,
        "meta": meta,
        "file_size_mb": round(file_size_mb, 2)
    }


def test_invalid_url():
    print("\n" + "=" * 75)
    print("TESTING ERROR HANDLING: INVALID URL")
    print("=" * 75)

    post_data = urllib.parse.urlencode({
        "upload_type": "youtube",
        "youtube_url": "https://vimeo.com/not_youtube",
        "mode": "multimodal",
        "budget": "15"
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{FLASK_BASE_URL}/upload",
        data=post_data,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )

    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode("utf-8")
        assert "Invalid YouTube URL" in html
        print("Success: Non-YouTube URL was rejected with friendly error notification!")


if __name__ == "__main__":
    url_tech = "https://www.youtube.com/watch?v=x7X9w_GIm1s"
    url_music = "https://www.youtube.com/watch?v=Ye8mB6VsUHw"

    # Test 1: Video 1 in Multimodal Mode
    res_multi = submit_youtube_job(url_tech, mode="multimodal", budget=15)

    # Test 2: Video 1 in Baseline Mode
    res_base = submit_youtube_job(url_tech, mode="baseline", budget=15)

    # Test 3: Video 2 in Multimodal Mode
    res_music = submit_youtube_job(url_music, mode="multimodal", budget=15)

    # Test 4: Error Handling
    test_invalid_url()

    print("\n" + "=" * 75)
    print("ALL REAL YOUTUBE END-TO-END TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 75)
