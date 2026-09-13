"""
scratch/test_web_pipeline_e2e.py
--------------------------------
Simulates end-to-end web browser interaction through Flask test client
for both TVSum video (akI8YFjEmUw.mp4) and custom video (FS_demo.mp4).
"""

import os
import sys
import time
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app, JOBS_STORE, JOBS_LOCK


def test_e2e_video(video_path: str, budget_pct: int = 15):
    print("\n" + "=" * 70)
    print(f"TESTING END-TO-END FLASK PIPELINE: {os.path.basename(video_path)}")
    print(f"Video Path: {video_path}")
    print(f"Budget: {budget_pct}%")
    print("=" * 70)

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    client = app.test_client()

    # Step 1: Upload video via POST /upload
    print("\n[Step 1] Uploading video to Flask app...")
    with open(video_path, "rb") as vf:
        data = {
            "video_file": (vf, os.path.basename(video_path)),
            "budget": str(budget_pct)
        }
        resp = client.post("/upload", data=data, content_type="multipart/form-data")

    assert resp.status_code == 302, f"Expected 302 redirect, got {resp.status_code}"
    loc = resp.headers.get("Location", "")
    print(f"  -> Upload accepted! Redirect URL: {loc}")

    job_id = loc.split("/")[-1]
    print(f"  -> Assigned Job ID: {job_id}")

    # Step 2: Poll /api/status/<job_id> until completion
    print("\n[Step 2] Polling /api/status/<job_id> during ML processing...")
    start_time = time.time()
    last_stage = None
    completed = False

    while time.time() - start_time < 300:  # 5 min timeout
        status_resp = client.get(f"/api/status/{job_id}")
        assert status_resp.status_code == 200, f"Status poll returned {status_resp.status_code}"
        payload = status_resp.get_json()
        status = payload.get("status")
        stage = payload.get("stage")

        if stage != last_stage:
            print(f"  -> [{time.time() - start_time:.1f}s] Stage: {stage} (status={status})")
            last_stage = stage

        if status == "complete":
            completed = True
            break
        elif status == "error":
            raise RuntimeError(f"Job failed with error: {payload.get('error')}")

        time.sleep(1.5)

    assert completed, "Job timed out before completion!"
    print(f"  -> Processing finished in {time.time() - start_time:.2f}s!")

    # Step 3: Verify outputs in web_outputs/<job_id>/
    print("\n[Step 3] Verifying generated web outputs on disk...")
    job_dir = os.path.join(BASE_DIR, "web_outputs", job_id)
    out_mp4 = os.path.join(job_dir, "highlight.mp4")
    out_png = os.path.join(job_dir, "scores.png")
    out_meta = os.path.join(job_dir, "segments.json")

    assert os.path.isfile(out_mp4), f"Missing highlight.mp4 in {job_dir}"
    assert os.path.isfile(out_png), f"Missing scores.png in {job_dir}"
    assert os.path.isfile(out_meta), f"Missing segments.json in {job_dir}"

    mp4_size_mb = os.path.getsize(out_mp4) / (1024 * 1024)
    png_size_kb = os.path.getsize(out_png) / 1024
    print(f"  -> highlight.mp4 size: {mp4_size_mb:.2f} MB")
    print(f"  -> scores.png size:     {png_size_kb:.1f} KB")

    with open(out_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)

    print(f"  -> Original duration:  {meta['original_duration']}s")
    print(f"  -> Highlight duration: {meta['highlight_duration']}s")
    print(f"  -> Number of clips:    {meta['num_clips']}")
    for c in meta["clips"]:
        print(f"     Clip {c['clip_num']}: {c['start_formatted']} -> {c['end_formatted']} ({c['duration_sec']}s)")

    # Step 4: Verify Result Page GET /result/<job_id>
    print("\n[Step 4] Verifying Result Page rendering...")
    res_resp = client.get(f"/result/{job_id}")
    assert res_resp.status_code == 200, f"Result page failed with {res_resp.status_code}"
    html = res_resp.get_data(as_text=True)
    assert "Your Highlights Are Ready!" in html
    assert f"/media/{job_id}/highlight.mp4" in html
    assert f"/media/{job_id}/scores.png" in html
    assert f"/download/{job_id}" in html
    print("  -> Result page HTML rendered with video player, clip list, score plot, and download button!")

    # Step 5: Verify Download route GET /download/<job_id>
    print("\n[Step 5] Verifying Download route...")
    dl_resp = client.get(f"/download/{job_id}")
    assert dl_resp.status_code == 200
    assert dl_resp.mimetype == "video/mp4"
    assert "attachment" in dl_resp.headers.get("Content-Disposition", "")
    print(f"  -> Download endpoint successfully served {len(dl_resp.data)/(1024*1024):.2f} MB MP4 attachment!")

    print(f"\n>>> TEST PASSED FOR: {os.path.basename(video_path)} <<<\n")
    return job_id


if __name__ == "__main__":
    # Test 1: TVSum video
    tvsum_video = os.path.join(BASE_DIR, "video", "akI8YFjEmUw.mp4")
    job1 = test_e2e_video(tvsum_video, budget_pct=15)

    # Test 2: Custom video
    custom_video = r"C:\Users\mahin\Downloads\FS_demo.mp4"
    job2 = test_e2e_video(custom_video, budget_pct=15)

    print("\n" + "=" * 70)
    print("ALL END-TO-END FLASK PIPELINE TESTS COMPLETED SUCCESSFULLY!")
    print(f"TVSum Job ID:  {job1}")
    print(f"Custom Job ID: {job2}")
    print("=" * 70)
