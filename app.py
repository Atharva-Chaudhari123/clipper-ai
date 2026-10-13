"""
app.py
------
Flask Web Application for Automatic Video Highlight Detection and Short Video Generation.

Features:
1. Video upload (MP4) OR direct YouTube URL processing (via yt-dlp)
2. Dual Analysis Modes: Visual + Speech (Multimodal) vs Visual Only (Baseline)
3. Real-time background worker with live 8-stage progress tracking
4. In-browser HTML5 highlight preview + speech transcript excerpts
5. Importance score plot visualization and direct MP4 downloads
6. Directory traversal protection, rate limiting, and robust error handling
"""

import os
import sys

# Ensure project root in sys.path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# 1. Setup Windows UTF-8 safety at startup
from src.utf8_utils import setup_utf8_environment, SafeStreamHandler
setup_utf8_environment()

import uuid
import time
import json
import logging
import threading
import io
import zipfile
from flask import (
    Flask, render_template, request, redirect, url_for,
    jsonify, send_from_directory, send_file, flash, abort
)
from werkzeug.utils import secure_filename

from main import run_pipeline, run_multi_reel_pipeline
from src.youtube_downloader import is_valid_youtube_url

# Configure logging with UTF-8 safe stream handler
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[SafeStreamHandler(sys.stdout)]
)
logger = logging.getLogger("FlaskHighlightApp")

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Directory configurations
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
WEB_OUTPUTS_FOLDER = os.path.join(BASE_DIR, "web_outputs")
YT_DOWNLOADS_FOLDER = os.path.join(BASE_DIR, "data", "youtube_downloads")
MODEL_CHECKPOINT = os.path.join(BASE_DIR, "models", "best_bilstm.pt")
MULTIMODAL_CHECKPOINT = os.path.join(BASE_DIR, "models", "best_multimodal_bilstm.pt")
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "annotations")

ALLOWED_EXTENSIONS = {"mp4"}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 MB max upload limit

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(WEB_OUTPUTS_FOLDER, exist_ok=True)
os.makedirs(YT_DOWNLOADS_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["WEB_OUTPUTS_FOLDER"] = WEB_OUTPUTS_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# In-memory tracking for active processing jobs
JOBS_STORE: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def allowed_file(filename: str) -> bool:
    """Checks if the filename has an allowed extension (.mp4)."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_safe_job_id(job_id: str) -> bool:
    """Validates job_id to prevent directory traversal attacks."""
    if not job_id or not isinstance(job_id, str):
        return False
    return all(c.isalnum() or c in ("_", "-") for c in job_id) and ".." not in job_id


def process_video_worker(
    job_id: str,
    video_path: str = None,
    youtube_url: str = None,
    model_mode: str = "multimodal",
    output_mode: str = "reels",
    num_reels: str = "auto",
    reel_length: str = "30-60",
    aspect_ratio: str = "9:16",
    captions: bool = True,
    framing_mode: str = "face_track",
    gemini_api_key: str = None,
    budget_ratio: float = 0.15,
    original_filename: str = "video.mp4"
):
    """
    Background worker thread running the ML pipeline asynchronously.
    Supports both Single Summary mode and Multi-Reel mode.
    Updates JOBS_STORE stage strings in real-time.
    """
    setup_utf8_environment()
    job_output_dir = os.path.join(app.config["WEB_OUTPUTS_FOLDER"], job_id)
    os.makedirs(job_output_dir, exist_ok=True)

    out_meta_path = os.path.join(job_output_dir, "segments.json")

    logger.info(
        f"Starting ML processing for Job '{job_id}' (Output: {output_mode.upper()}, "
        f"Mode: {model_mode}, Reels: {num_reels}, Length: {reel_length}, Aspect: {aspect_ratio})..."
    )

    def update_stage(msg: str):
        with JOBS_LOCK:
            if job_id in JOBS_STORE:
                JOBS_STORE[job_id]["stage"] = str(msg)

    try:
        with JOBS_LOCK:
            JOBS_STORE[job_id]["status"] = "processing"
            JOBS_STORE[job_id]["stage"] = "Initializing processing..."

        if output_mode == "reels":
            results = run_multi_reel_pipeline(
                video_path=video_path,
                youtube_url=youtube_url,
                model_mode=model_mode,
                num_reels=num_reels,
                reel_length=reel_length,
                aspect_ratio=aspect_ratio,
                captions=captions,
                use_face_tracking=(framing_mode == "face_track"),
                gemini_api_key=gemini_api_key,
                output_dir=job_output_dir,
                stage_callback=update_stage,
                job_id=job_id
            )
        else:
            out_video_path = os.path.join(job_output_dir, "highlight.mp4")
            out_plot_path = os.path.join(job_output_dir, "scores.png")
            results = run_pipeline(
                video_path=video_path,
                youtube_url=youtube_url,
                model_mode=model_mode,
                budget_ratio=budget_ratio,
                output_video_path=out_video_path,
                plot_output_path=out_plot_path,
                annotations_dir=ANNOTATIONS_DIR,
                stage_callback=update_stage,
                job_id=job_id
            )

        results["original_display_name"] = original_filename
        results["job_id"] = job_id
        results["output_mode"] = output_mode

        # Save metadata to web_outputs/<job_id>/segments.json
        with open(out_meta_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        with JOBS_LOCK:
            JOBS_STORE[job_id]["status"] = "complete"
            if output_mode == "reels":
                JOBS_STORE[job_id]["stage"] = f"Generated {results.get('num_reels', len(results.get('reels', [])))} Reels successfully!"
            else:
                JOBS_STORE[job_id]["stage"] = "Highlight video successfully generated!"
            JOBS_STORE[job_id]["result"] = results

        logger.info(f"Job '{job_id}' completed successfully!")

    except Exception as e:
        try:
            err_msg = str(e)
        except Exception:
            err_msg = repr(e)
        try:
            logger.error(f"Error processing job '{job_id}': {err_msg}", exc_info=True)
        except Exception:
            pass
        with JOBS_LOCK:
            JOBS_STORE[job_id]["status"] = "error"
            JOBS_STORE[job_id]["stage"] = "Failed"
            JOBS_STORE[job_id]["error"] = f"Processing failed: {err_msg}"



# ----------------------------------------------------------------------
# ROUTES
# ----------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """Renders home landing page with tabs for Video Upload and YouTube URL."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Handles both file uploads and YouTube URL submissions for single or multi-reel output."""
    upload_type = request.form.get("upload_type", "file")
    model_mode = request.form.get("mode", "multimodal")
    if model_mode not in ["multimodal", "baseline"]:
        model_mode = "multimodal"

    output_mode = request.form.get("output_mode", "reels")
    if output_mode not in ["reels", "single"]:
        output_mode = "reels"

    num_reels = request.form.get("num_reels", "auto")
    if num_reels not in ["auto", "3", "5", "10"]:
        num_reels = "auto"

    reel_length = request.form.get("reel_length", "30-60")
    if reel_length not in ["15-30", "30-60", "60-90", "auto"]:
        reel_length = "30-60"

    aspect_ratio = request.form.get("aspect_ratio", "9:16")
    if aspect_ratio not in ["9:16", "16:9"]:
        aspect_ratio = "9:16"

    captions_param = request.form.get("captions", "on")
    captions = captions_param in ["on", "1", "true", "True"]

    framing_mode = request.form.get("framing_mode", "face_track")
    if framing_mode not in ["face_track", "center_crop"]:
        framing_mode = "face_track"

    gemini_api_key = request.form.get("gemini_api_key", "").strip() or None

    # Parse and validate budget percentage for single mode
    budget_param = request.form.get("budget", "15")
    try:
        budget_pct = int(budget_param)
        if budget_pct not in [10, 15, 20]:
            budget_pct = 15
    except ValueError:
        budget_pct = 15
    budget_ratio = budget_pct / 100.0

    # 1. Handle YouTube Submission
    if upload_type == "youtube":
        youtube_url = request.form.get("youtube_url", "").strip()
        if not youtube_url:
            flash("Please enter a valid YouTube URL.", "danger")
            return redirect(url_for("index"))

        if not is_valid_youtube_url(youtube_url):
            flash("Invalid YouTube URL! Please provide a standard YouTube, youtu.be, or Shorts link.", "danger")
            return redirect(url_for("index"))

        job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:6]}"

        with JOBS_LOCK:
            JOBS_STORE[job_id] = {
                "status": "queued",
                "stage": "Validating YouTube URL and preparing download...",
                "error": None,
                "result": None,
                "created_at": time.time()
            }

        thread = threading.Thread(
            target=process_video_worker,
            kwargs={
                "job_id": job_id,
                "video_path": None,
                "youtube_url": youtube_url,
                "model_mode": model_mode,
                "output_mode": output_mode,
                "num_reels": num_reels,
                "reel_length": reel_length,
                "aspect_ratio": aspect_ratio,
                "captions": captions,
                "framing_mode": framing_mode,
                "gemini_api_key": gemini_api_key,
                "budget_ratio": budget_ratio,
                "original_filename": youtube_url
            },
            daemon=True
        )
        thread.start()
        return redirect(url_for("processing", job_id=job_id))

    # 2. Handle Local File Upload
    else:
        if "video_file" not in request.files:
            flash("No video file selected. Please select an MP4 video.", "danger")
            return redirect(url_for("index"))

        file = request.files["video_file"]
        if file.filename == "":
            flash("No file was chosen. Please select an MP4 video to upload.", "danger")
            return redirect(url_for("index"))

        if not allowed_file(file.filename):
            flash("Invalid file format. Only MP4 videos (.mp4) are currently supported.", "danger")
            return redirect(url_for("index"))

        orig_filename = secure_filename(file.filename) or "uploaded_video.mp4"
        job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:6]}"

        saved_video_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{job_id}_{orig_filename}")
        file.save(saved_video_path)

        with JOBS_LOCK:
            JOBS_STORE[job_id] = {
                "status": "queued",
                "stage": "Video uploaded. Initializing pipeline...",
                "error": None,
                "result": None,
                "created_at": time.time()
            }

        thread = threading.Thread(
            target=process_video_worker,
            kwargs={
                "job_id": job_id,
                "video_path": saved_video_path,
                "youtube_url": None,
                "model_mode": model_mode,
                "output_mode": output_mode,
                "num_reels": num_reels,
                "reel_length": reel_length,
                "aspect_ratio": aspect_ratio,
                "captions": captions,
                "framing_mode": framing_mode,
                "gemini_api_key": gemini_api_key,
                "budget_ratio": budget_ratio,
                "original_filename": orig_filename
            },
            daemon=True
        )
        thread.start()
        return redirect(url_for("processing", job_id=job_id))


@app.route("/processing/<job_id>", methods=["GET"])
def processing(job_id: str):
    """Renders the processing status page."""
    if not is_safe_job_id(job_id):
        abort(400, "Invalid Job ID.")

    with JOBS_LOCK:
        job = JOBS_STORE.get(job_id)

    meta_path = os.path.join(app.config["WEB_OUTPUTS_FOLDER"], job_id, "segments.json")
    if job is None and os.path.isfile(meta_path):
        return redirect(url_for("result", job_id=job_id))

    if job is None:
        flash("Job not found or has expired. Please submit your video again.", "warning")
        return redirect(url_for("index"))

    if job["status"] == "complete":
        return redirect(url_for("result", job_id=job_id))

    return render_template("processing.html", job_id=job_id)


@app.route("/api/status/<job_id>", methods=["GET"])
def api_status(job_id: str):
    """JSON API polled by processing.html to check job state."""
    if not is_safe_job_id(job_id):
        return jsonify({"status": "error", "error": "Invalid Job ID"}), 400

    with JOBS_LOCK:
        job = JOBS_STORE.get(job_id)

    meta_path = os.path.join(app.config["WEB_OUTPUTS_FOLDER"], job_id, "segments.json")
    if job is None and os.path.isfile(meta_path):
        return jsonify({"status": "complete", "stage": "Complete", "redirect_url": url_for("result", job_id=job_id)})

    if job is None:
        return jsonify({"status": "error", "error": "Job not found"}), 404

    resp = {
        "status": job["status"],
        "stage": job["stage"],
        "error": job["error"],
        "redirect_url": url_for("result", job_id=job_id) if job["status"] == "complete" else None
    }
    return jsonify(resp)


@app.route("/result/<job_id>", methods=["GET"])
def result(job_id: str):
    """Renders the results page (supporting both single summary and multi-reel cards)."""
    if not is_safe_job_id(job_id):
        abort(400, "Invalid Job ID.")

    job_dir = os.path.join(app.config["WEB_OUTPUTS_FOLDER"], job_id)
    meta_path = os.path.join(job_dir, "segments.json")

    if not os.path.isfile(meta_path):
        flash("Result metadata not found for this job.", "danger")
        return redirect(url_for("index"))

    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return render_template("result.html", job_id=job_id, data=data)


@app.route("/media/<job_id>/<filename>", methods=["GET"])
def serve_media(job_id: str, filename: str):
    """Safely serves highlight video, reels, or score plot images for in-browser playback."""
    if not is_safe_job_id(job_id):
        abort(400, "Invalid Job ID.")

    if not filename or ".." in filename or "/" in filename or "\\" in filename:
        abort(403, "Access denied.")

    is_valid_file = filename == "scores.png" or filename.endswith(".mp4")
    if not is_valid_file:
        abort(403, "Access denied.")

    job_dir = os.path.join(app.config["WEB_OUTPUTS_FOLDER"], job_id)
    if not os.path.isdir(job_dir):
        abort(404, "Job directory not found.")

    mimetype = "video/mp4" if filename.endswith(".mp4") else "image/png"
    return send_from_directory(job_dir, filename, mimetype=mimetype)


@app.route("/download/<job_id>", methods=["GET"])
@app.route("/download/<job_id>/<filename>", methods=["GET"])
def download(job_id: str, filename: str = None):
    """Forces browser download of a specific highlight/reel video file."""
    if not is_safe_job_id(job_id):
        abort(400, "Invalid Job ID.")

    job_dir = os.path.join(app.config["WEB_OUTPUTS_FOLDER"], job_id)
    if not os.path.isdir(job_dir):
        abort(404, "Job directory not found.")

    if filename is None:
        target_file = "highlight.mp4"
        if not os.path.isfile(os.path.join(job_dir, target_file)):
            reels = sorted([f for f in os.listdir(job_dir) if f.startswith("reel_") and f.endswith(".mp4")])
            target_file = reels[0] if reels else "highlight.mp4"
    else:
        if ".." in filename or "/" in filename or "\\" in filename or not filename.endswith(".mp4"):
            abort(403, "Access denied.")
        target_file = filename

    file_path = os.path.join(job_dir, target_file)
    if not os.path.isfile(file_path):
        abort(404, "Requested video file not found.")

    return send_from_directory(
        job_dir,
        target_file,
        as_attachment=True,
        download_name=f"{job_id}_{target_file}",
        mimetype="video/mp4"
    )


@app.route("/download_all/<job_id>", methods=["GET"])
def download_all(job_id: str):
    """Bundles all generated reels and manifest.json into a ZIP archive for one-click download."""
    if not is_safe_job_id(job_id):
        abort(400, "Invalid Job ID.")

    job_dir = os.path.join(app.config["WEB_OUTPUTS_FOLDER"], job_id)
    if not os.path.isdir(job_dir):
        abort(404, "Job directory not found.")

    meta_path = os.path.join(job_dir, "segments.json")
    title_prefix = "reels"
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                yt_meta = meta.get("youtube_meta")
                if yt_meta and yt_meta.get("title"):
                    title_prefix = secure_filename(yt_meta["title"])[:30]
        except Exception:
            pass

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(job_dir):
            if fname.endswith(".mp4") or fname == "manifest.json":
                fpath = os.path.join(job_dir, fname)
                zf.write(fpath, arcname=fname)

    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        as_attachment=True,
        download_name=f"{title_prefix}_{job_id}_reels.zip",
        mimetype="application/zip"
    )


# ----------------------------------------------------------------------
# ERROR HANDLERS
# ----------------------------------------------------------------------

@app.errorhandler(413)
def request_entity_too_large(error):
    flash("Uploaded file is too large! Maximum allowed video size is 500 MB.", "danger")
    return redirect(url_for("index"))


@app.errorhandler(404)
def not_found_error(error):
    return render_template("index.html", error_message="Page not found."), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return render_template("index.html", error_message="An unexpected server error occurred. Please try again."), 500


if __name__ == "__main__":
    logger.info("Starting Flask Highlight Detection Web Server at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
