# Automatic Video Highlight Detection & Short Video Generation (Final Year Project)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%20CUDA-ee4c2c.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-black.svg)](https://flask.palletsprojects.com/)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-Supported-green.svg)](https://ffmpeg.org/)
[![faster--whisper](https://img.shields.io/badge/Speech-faster--whisper%20(CTranslate2)-orange.svg)](https://github.com/SYSTRAN/faster-whisper)

An end-to-end Machine Learning and Computer Vision system that takes long-form videos (local MP4 uploads or direct YouTube URLs) and automatically produces **multiple short-form vertical Reels/YouTube Shorts (9:16)** with burned-in subtitles, or a concise **single highlight summary (16:9)**.

---

## 🌟 Key Features

1. **Dual Video Ingestion**:
   - **Local Video Upload** (drag-and-drop MP4 up to 500 MB).
   - **YouTube URL Processing** via `yt-dlp` (standard watch links, shortened `youtu.be`, and Shorts).
2. **Two-Stage Analysis Architecture (5.7× Speedup)**:
   - **Stage 1 (Pre-Screening)**: High-speed 0.5 FPS visual frame sampling with frozen ResNet-50 + ultra-fast windowed RMS audio energy Voice Activity Detection (VAD in $<0.5$s).
   - **Stage 2 (Candidate-Only Transcription)**: Uses `faster-whisper` (CTranslate2 on GPU) to transcribe *only* high-potential candidate intervals rather than transcribing the entire long video, cutting transcription time by **>95%**.
3. **Multimodal Bi-LSTM Fusion**:
   - Combines 2048-D ResNet-50 visual feature projections with 384-D `all-MiniLM-L6-v2` speech embeddings to evaluate video importance scores.
4. **Smart Multi-Reel Selection**:
   - Temporal **Non-Maximum Suppression (NMS)** prevents repetitive moments.
   - **Sentence-Boundary Snapping** avoids cutting speech mid-sentence.
   - Target duration constraints (15–30s, 30–60s, 60–90s).
5. **Automated Vertical 9:16 Video Rendering**:
   - FFmpeg center-crop conversion to `1080x1920` 9:16 vertical format.
   - Burned-in, high-contrast styled captions with automatic subtitle time-shifting.
6. **Modern Flask Web Interface**:
   - Real-time 11-stage background progress tracking.
   - In-browser HTML5 video players for every generated reel with transcripts and importance scores.
   - Individual reel downloads and a **one-click ZIP download bundle** (`/download_all/<job_id>`).
7. **Windows UTF-8 & Charmap Safe**:
   - Comprehensive character encodings handling emoji and non-ASCII titles (e.g., `😰 🎬 ✅`) without crashing on Windows.

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
     Baseline Bi-LSTM Scoring                        Fast RMS Energy / ZCR VAD
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
                         (CTranslate2 int8_float16 GPU)
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
                         [ FFmpeg 9:16 Vertical Rendering ]
                         (Center Crop + Burned-In Captions)
                                         |
                                         v
                      [ Flask Web UI & One-Click ZIP Download ]
```

---

## 🚀 Setup & Installation Guide

### 1. Prerequisites

- **Operating System**: Windows 10/11, Linux, or macOS.
- **Python**: Version `3.10` or higher.
- **FFmpeg**: Must be installed and added to your system `PATH`.
  - *Verify via terminal:* `ffmpeg -version`
- **NVIDIA GPU (Recommended)**: NVIDIA GeForce GTX 1650 or higher with CUDA 11.8 / 12.x for accelerated neural inference. CPU fallback is automatically supported.

### 2. Clone the Repository

```bash
git clone https://github.com/anshu34253/FinalYearProject.git
cd FinalYearProject
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

> **Note for CUDA Users:**  
> If using GPU acceleration on Windows with CTranslate2 and PyTorch, install the official PyTorch wheel with CUDA:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
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
     - **Multiple Short Reels (9:16)** *(Default)*: Set number of reels (Auto, 3, 5, 10), target length (30–60s), and burned-in captions (ON/OFF).
     - **Single Highlight Summary (16:9)**: Set duration budget (10%, 15%, 20%).
   - Click **Generate Reels (9:16)**.
   - Monitor the 11 real-time progress stages.
   - View results, play reels with subtitles in-browser, or click **Download All Reels (.ZIP)**.

---

### Option B: Run via Command-Line Interface (CLI)

You can run highlight extraction directly from the command line using `main.py`:

#### 1. Generate Multiple 9:16 Vertical Reels from YouTube:
```bash
python main.py --youtube-url "https://www.youtube.com/watch?v=x7X9w_GIm1s" --output-mode reels --num-reels 3 --reel-length 15-30 --aspect-ratio 9:16
```

#### 2. Generate Multiple Reels from a Local Video:
```bash
python main.py --video-path "path/to/video.mp4" --output-mode reels --num-reels auto --reel-length 30-60
```

#### 3. Generate Single 16:9 Summary Video:
```bash
python main.py --video-path "path/to/video.mp4" --output-mode single --budget 0.15 --model-mode multimodal
```

---

## 🧪 Running Automated Tests

The repository includes comprehensive automated test suites covering all modules:

```bash
# Run the complete Phase 13 Multi-Reel test suite (15 tests)
python tests/test_phase13_multireel.py

# Run Windows UTF-8 / charmap encoding safety regression tests
python tests/test_utf8_encoding.py

# Run Flask Web Application route and security tests
python tests/test_flask_app.py
```

---

## 📊 Benchmark & Performance Comparison

Evaluated on a real 22-minute video (*"GPT 6 Astra - End of Developers?"*, 1314.0s / 21.9 min) running on an **NVIDIA GeForce GTX 1650 (4GB VRAM)**:

| Metric | Old Baseline Pipeline | New Two-Stage Multi-Reel Pipeline | Improvement |
|:---|:---:|:---:|:---:|
| **Whisper Audio Transcribed** | 1314.0s (100%) | 547.0s (Candidates Only) | **58.4% audio reduction** |
| **Speech Transcription Time** | >840.0s (>14 min) | 39.47s (faster-whisper GPU) | **95.3% reduction** |
| **Total Pipeline Processing Time** | ~1,200.0s (~20 min) | **211.49s (3.52 min)** | **5.7× faster (82.4% reduction)** |
| **Outputs Produced** | 1 landscape video | **8 vertical 9:16 Reels with subtitles** | **8× more short-form content** |
| **VRAM Peak Usage** | >2.5 GB | **451.3 MB** | **Lightweight memory footprint** |

---

## 📁 Repository Structure

```
FinalYearProject/
├── app.py                      # Flask web application & asynchronous worker
├── main.py                     # Unified CLI & pipeline entrypoint
├── run.bat                     # Windows one-click start script
├── requirements.txt            # Python package dependencies
├── README.md                   # Project documentation and guide
│
├── models/                     # Trained ML checkpoints
│   ├── best_bilstm.pt          # Baseline visual Bi-LSTM checkpoint
│   ├── best_multimodal_bilstm.pt # Multimodal (visual + speech) Bi-LSTM checkpoint
│   └── training_history.json   # Training epochs & loss convergence logs
│
├── src/                        # Core algorithms and pipeline modules
│   ├── audio_pre_screener.py   # High-speed windowed RMS audio VAD
│   ├── fast_visual_screener.py # Batched ResNet-50 feature extraction
│   ├── candidate_generator.py  # Window generation & overlap merging
│   ├── fast_transcriber.py     # CTranslate2 faster-whisper GPU transcription
│   ├── text_embedder.py        # MiniLM-L6-v2 384-D sentence embeddings
│   ├── reel_selector.py        # Multimodal Bi-LSTM scoring & temporal NMS
│   ├── reel_renderer.py        # FFmpeg 9:16 crop & burned-in captions
│   ├── youtube_downloader.py   # Safe yt-dlp acquisition and validation
│   ├── utf8_utils.py           # Windows UTF-8 and codepage safety utilities
│   ├── lstm_model.py           # Baseline PyTorch Bi-LSTM architecture
│   ├── multimodal_model.py     # Multimodal PyTorch Bi-LSTM architecture
│   └── renderer.py             # Single summary concatenation renderer
│
├── templates/                  # HTML5 Jinja2 web templates
│   ├── index.html              # Landing page with multi-reel controls
│   ├── processing.html         # Live 11-stage progress tracker
│   └── result.html             # Multi-reel grid preview & ZIP download
│
├── static/                     # Web UI styling & interactions
│   ├── css/style.css           # Modern dark-theme responsive design
│   └── js/script.js            # Live status polling & dynamic toggles
│
├── tests/                      # Automated unit & integration test suites
│   ├── test_phase13_multireel.py
│   ├── test_utf8_encoding.py
│   └── test_flask_app.py
│
├── annotations/                # TVSum benchmark dataset ground truth
└── scripts/                    # Validation and evaluation scripts
```

---

## 📄 License & Academic Attribution

Developed as a Final Year Computer Science / AI Project on **Automatic Video Highlight Detection and Short Video Generation**.

Utilizes pretrained backbones and datasets:
- **ResNet-50** (He et al., Microsoft Research)
- **TVSum Dataset** (Song et al., Title-based Video Summarization)
- **faster-whisper** (Guillaume Klein et al., SYSTRAN)
- **all-MiniLM-L6-v2** (Sentence-Transformers, Reimers & Gurevych)
