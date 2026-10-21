# Automatic Video Highlight Detection & Viral Short Video Generation (HighlightAI / Clipper-AI)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%20CUDA%2012.6-ee4c2c.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-black.svg)](https://flask.palletsprojects.com/)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-libass%20Supported-green.svg)](https://ffmpeg.org/)
[![faster--whisper](https://img.shields.io/badge/Speech-faster--whisper%20(CTranslate2)-orange.svg)](https://github.com/SYSTRAN/faster-whisper)
[![OpenCV](https://img.shields.io/badge/Vision-OpenCV%20YuNet%20Face%20Track-red.svg)](https://github.com/opencv/opencv_zoo)

An end-to-end Machine Learning, Computer Vision, and Speech Intelligence system that transforms long-form videos (local MP4 uploads or YouTube links) into **viral, short-form vertical Reels/Shorts (9:16)** with **AI face-aware tracking** and **word-by-word animated karaoke subtitles**, or a stitched **single highlight summary (16:9)**.

---

## 🌟 Key Features

1. **Dual Video Ingestion**:
   - **Local Video Upload**: Drag-and-drop MP4 video files up to 500 MB.
   - **YouTube URL Processing**: Via `yt-dlp` supporting standard `watch?v=...`, shortened `youtu.be/`, and `youtube.com/shorts/` URLs.
2. **Two-Stage Analysis Architecture (5.7× Speedup)**:
   - **Stage 1 (Pre-Screening)**: High-speed visual frame sampling at 0.5 FPS with pretrained ResNet-50 + ultra-fast windowed RMS audio energy Voice Activity Detection (VAD in $<0.5$s).
   - **Stage 2 (Candidate-Only Transcription)**: Uses `faster-whisper` (CTranslate2 on GPU) to transcribe *only* high-potential candidate intervals with word-level timestamps (`word_timestamps=True`), cutting speech transcription time by **>95%**.
3. **Multimodal Bi-LSTM Fusion**:
   - Fuses 2048-D ResNet-50 visual feature projections with 384-D `all-MiniLM-L6-v2` speech embeddings to predict highlight importance scores between 0 and 1.
4. **Smart Multi-Reel Selection**:
   - Temporal **Non-Maximum Suppression (NMS)** prevents content duplication.
   - **Sentence-Boundary Snapping** avoids mid-sentence cuts.
   - Target duration presets (15–30s, 30–60s, 60–90s, Auto).
5. **AI Face-Aware 9:16 Tracking (YuNet ONNX)**:
   - Deep neural network face detector (`face_detection_yunet.onnx`) running at 60+ FPS.
   - Dynamically tracks speakers across the frame with Exponential Moving Average (EMA) smoothing to produce cinema-grade portrait reframing without sudden jerks.
6. **Dynamic Word-by-Word Karaoke Subtitles (ASS Engine)**:
   - Advanced SubStation Alpha (`.ass` v4.0+) styling with pixel-perfect resolution (`PlayResX: 1080`, `PlayResY: 1920`).
   - Sits in the **lower third** (`MarginV: 240px`) to never obstruct speaker faces.
   - Real-time active word highlighting: words light up in **bold vibrant yellow (`&H0000FFFF`)** as spoken, while inactive words stay crisp white with a bold black outline.
7. **Dual AI Virality Engine**:
   - **Offline NLP Engine (Default)**: Fast lexical heuristic evaluation of hook sentences, curiosity triggers, and sentiment with zero external API calls.
   - **Google Gemini API (Optional)**: Free-tier Gemini Flash integration for creative social media titles and virality reasoning.
8. **Modern Responsive Web Interface**:
   - Intuitive stepped workflow (Step 1: Video Source, Step 2: Output Format, Step 3: Reel Customization, Step 4: AI Analysis Pipeline, Step 5: Virality Engine).
   - Real-time 11-stage progress tracking mapped to client-side UI via REST status polling.
   - In-browser HTML5 video players, subtitle previews, and a **one-click ZIP bundle download** (`/download_all/<job_id>`).
9. **Full GPU Acceleration & Windows UTF-8 Safety**:
   - NVIDIA CUDA 12.6 acceleration on PyTorch and CTranslate2.
   - Safe UTF-8 stream handling preventing Windows `charmap` Unicode errors.

---

## 🏛️ System Architecture

```
                                  [ Input Video ]
                        (Local MP4 Upload or YouTube URL)
                                         |
                +------------------------+------------------------+
                |                                                 |
                v                                                 v
    [ Stage 1: Visual Pre-Screen ]                   [ Stage 1: Audio Pre-Screen ]
    0.5 FPS ResNet-50 Feature Matrix                  16 kHz Mono WAV Extraction
                |                                                 |
      Baseline Bi-LSTM Scoring                       Fast RMS Energy / ZCR VAD
                |                                                 |
                +------------------------+------------------------+
                                         |
                                         v
                          [ Candidate Window Generation ]
                          (Peak Detection & Overlap Merge)
                                         |
                                         v
                         [ Stage 2: Fast Transcription ]
                        faster-whisper on Candidates ONLY
                         (CTranslate2 GPU + Word Timestamps)
                                         |
                                         v
                          [ MiniLM Text Embedding Fusion ]
                           (384-D Candidate Projections)
                                         |
                                         v
                        [ Multimodal Bi-LSTM Model Scoring ]
                        (models/best_multimodal_bilstm.pt)
                                         |
                                         v
                         [ Multi-Reel Selection & NMS ]
                       Sentence-Boundary Boundary Snapping
                                         |
                                         v
                        [ Dual Virality Scoring Engine ]
                       (Google Gemini / Local Offline NLP)
                                         |
                                         v
                 [ OpenCV YuNet Smart Face Tracking & Reframing ]
                               (1080x1920 9:16 Crop)
                                         |
                                         v
                     [ Dynamic ASS Karaoke Subtitle Burning ]
                    (Lower-Third Placement + Yellow Word Glow)
                                         |
                                         v
                      [ Flask Web UI & One-Click ZIP Download ]
```

---

## 🚀 Setup & Installation Guide

### 1. Prerequisites

- **Operating System**: Windows 10/11 (64-bit), Linux, or macOS.
- **Python**: Version `3.10` to `3.12`.
- **FFmpeg with `libass`**: Must be installed and available in system `PATH`.
  - *Verify via terminal:* `ffmpeg -version`
- **NVIDIA GPU (Recommended)**: NVIDIA RTX/GTX GPU with CUDA support for accelerated inference. CPU fallback is fully supported.

### 2. Clone the Repository

```bash
git clone https://github.com/Atharva-Chaudhari123/clipper-ai.git
cd clipper-ai
```

### 3. Create and Activate Virtual Environment

**On Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**On Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **For PyTorch CUDA GPU Acceleration:**
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
> ```

---

## 💻 How to Run the Project

### Option A: Run the Flask Web Application (Recommended)

1. Start the server:
   ```bash
   python app.py
   ```
   *(Or double-click `run.bat` on Windows)*

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

3. **Using the Web UI**:
   - Choose between **Upload Local Video** or **Paste YouTube URL**.
   - Select Output Format:
     - **Multiple Viral Reels (9:16)** *(Default)*: Configure reel count (Auto, 3, 5, 10), duration (30–60s), framing mode (`Smart Face Tracking` or `Center Crop`), and burned-in captions (`ON`/`OFF`).
     - **Single Highlight Summary (16:9)**: Choose duration budget (10%, 15%, 20%).
   - *(Optional)* Paste a free Google Gemini API key for enhanced viral titles (or leave blank for offline NLP).
   - Click **Generate Viral Reels (9:16)**.
   - Monitor the real-time 11-stage progress tracker.
   - View results, stream vertical reels with synchronized karaoke captions, and download reels individually or as a **ZIP bundle**.

---

### Option B: Run via Command-Line Interface (CLI)

```bash
# 1. Generate Multiple 9:16 Vertical Reels from YouTube
python main.py --youtube "https://www.youtube.com/watch?v=x7X9w_GIm1s" --output-mode reels --num-reels 3 --reel-length 30-60 --aspect-ratio 9:16

# 2. Generate Multiple Reels from a Local Video File
python main.py --video "path/to/video.mp4" --output-mode reels --num-reels auto --reel-length 30-60

# 3. Generate Single 16:9 Highlight Summary
python main.py --video "path/to/video.mp4" --output-mode single --budget 0.15 --mode multimodal
```

---

## 🧪 Running Automated Tests

```bash
# Run the complete Phase 13 Multi-Reel test suite
python tests/test_phase13_multireel.py

# Run Windows UTF-8 / charmap encoding safety regression tests
python tests/test_utf8_encoding.py

# Run Flask Web Application route and security tests
python tests/test_flask_app.py
```

---

## 📊 Benchmark & Performance Comparison

Evaluated on a real 22-minute video (*"GPT 6 Astra - End of Developers?"*, 1314.0s / 21.9 min) running on an **NVIDIA GeForce RTX 4050 Laptop GPU**:

| Metric | Old Baseline Pipeline | New Two-Stage Multi-Reel Pipeline | Improvement |
| :--- | :---: | :---: | :---: |
| **Whisper Audio Transcribed** | 1314.0s (100%) | 547.0s (Candidates Only) | **58.4% audio reduction** |
| **Speech Transcription Time** | >840.0s (>14 min) | 39.47s (faster-whisper GPU) | **95.3% reduction** |
| **Total Pipeline Processing Time** | ~1,200.0s (~20 min) | **211.49s (3.52 min)** | **5.7× faster (82.4% reduction)** |
| **Outputs Produced** | 1 landscape video | **8 vertical 9:16 Reels with subtitles** | **8× more short-form content** |
| **Framing Intelligence** | Static Center Crop | **YuNet Deep Face Tracking (EMA)** | **Always centered on speaker** |
| **Subtitles** | Unscaled Raw Text | **Dynamic Word-by-Word Karaoke ASS** | **Lower-third, yellow active highlight** |

---

## 📁 Repository Structure

```
clipper-ai/
├── app.py                      # Flask web application & async worker queue
├── main.py                     # Unified CLI & pipeline entrypoint
├── run.bat                     # Windows one-click start script
├── requirements.txt            # Python package dependencies
├── README.md                   # Comprehensive project documentation
│
├── models/                     # Trained deep learning checkpoints
│   ├── best_bilstm.pt          # Baseline visual Bi-LSTM checkpoint
│   ├── best_multimodal_bilstm.pt # Multimodal (visual + speech) Bi-LSTM checkpoint
│   └── training_history.json   # Training convergence & loss curves
│
├── src/                        # Core algorithms and pipeline modules
│   ├── face_tracker.py         # OpenCV YuNet smart face tracking with smoothing
│   ├── face_detection_yunet.onnx # Pretrained YuNet face detection ONNX model
│   ├── virality_llm.py         # Dual virality engine (Gemini Flash + Offline NLP)
│   ├── reel_renderer.py        # 9:16 reframing & dynamic ASS karaoke captions
│   ├── reel_selector.py        # Multimodal Bi-LSTM scoring & temporal NMS
│   ├── fast_transcriber.py     # CTranslate2 faster-whisper GPU transcriber
│   ├── text_embedder.py        # MiniLM-L6-v2 384-D sentence embeddings
│   ├── audio_pre_screener.py   # High-speed windowed RMS audio VAD
│   ├── fast_visual_screener.py # Batched ResNet-50 visual feature extraction
│   ├── candidate_generator.py  # Window generation & overlap merging
│   ├── youtube_downloader.py   # Safe yt-dlp acquisition and validation
│   ├── utf8_utils.py           # Windows UTF-8 and codepage safety utilities
│   ├── lstm_model.py           # Baseline PyTorch Bi-LSTM architecture
│   ├── multimodal_model.py     # Multimodal PyTorch Bi-LSTM architecture
│   └── renderer.py             # Single summary concatenation renderer
│
├── templates/                  # Modern responsive HTML5 templates
│   ├── index.html              # Stepped landing page & reel customization
│   ├── processing.html         # Live 11-stage progress tracker
│   └── result.html             # Multi-reel grid player & ZIP bundle export
│
├── static/                     # Web UI styling & interactions
│   ├── css/style.css           # Modern dark glassmorphic responsive design
│   └── js/script.js            # Live status polling & interactive controls
│
├── tests/                      # Automated unit & integration test suites
│   ├── test_phase13_multireel.py
│   ├── test_utf8_encoding.py
│   └── test_flask_app.py
│
├── annotations/                # TVSum benchmark dataset ground truth
└── data/                       # Cached audio/video working directories
```

---

## 📄 License & Academic Attribution

Developed as a Final Year Computer Science / AI Project on **Automatic Video Highlight Detection and Short Video Generation**.

Utilizes pretrained models and datasets:
- **ResNet-50** (He et al., Microsoft Research)
- **YuNet Face Detector** (OpenCV Model Zoo)
- **TVSum Dataset** (Song et al., Title-based Video Summarization)
- **faster-whisper** (Guillaume Klein et al., SYSTRAN)
- **all-MiniLM-L6-v2** (Sentence-Transformers, Reimers & Gurevych)
- **Google Gemini 1.5 Flash** (Google DeepMind)
