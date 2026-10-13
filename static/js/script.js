/**
 * static/js/script.js
 * -------------------
 * Interactive behaviors for tab switching, video upload / YouTube URL validation,
 * output mode switching, API key visibility toggle, and dynamic multi-stage processing polling.
 */

document.addEventListener("DOMContentLoaded", () => {
  initFormHandler();
  initProcessingPoller();
});

/**
 * Switches between Local File Upload and YouTube URL tabs.
 */
function switchSourceTab(type) {
  const uploadTypeInput = document.getElementById("uploadType");
  const filePanel = document.getElementById("fileUploadPanel");
  const youtubePanel = document.getElementById("youtubePanel");
  const tabUploadBtn = document.getElementById("tabUploadBtn");
  const tabYoutubeBtn = document.getElementById("tabYoutubeBtn");

  if (!uploadTypeInput) return;

  uploadTypeInput.value = type;

  if (type === "youtube") {
    if (filePanel) filePanel.style.display = "none";
    if (youtubePanel) youtubePanel.style.display = "block";
    if (tabUploadBtn) tabUploadBtn.classList.remove("active");
    if (tabYoutubeBtn) tabYoutubeBtn.classList.add("active");
  } else {
    if (youtubePanel) youtubePanel.style.display = "none";
    if (filePanel) filePanel.style.display = "block";
    if (tabYoutubeBtn) tabYoutubeBtn.classList.remove("active");
    if (tabUploadBtn) tabUploadBtn.classList.add("active");
  }
}

/**
 * Switches between Multi-Reel and Single Summary output modes on index page.
 */
function switchOutputMode(mode) {
  const multiPanel = document.getElementById("multiReelOptionsPanel");
  const singlePanel = document.getElementById("singleSummaryBudgetPanel");
  const btnText = document.getElementById("submitBtnText");

  if (mode === "reels") {
    if (multiPanel) multiPanel.style.display = "block";
    if (singlePanel) singlePanel.style.display = "none";
    if (btnText) btnText.textContent = "Generate Viral Reels (9:16)";
  } else {
    if (multiPanel) multiPanel.style.display = "none";
    if (singlePanel) singlePanel.style.display = "block";
    if (btnText) btnText.textContent = "Generate Highlight Summary";
  }
}

/**
 * Toggles visibility of the optional Gemini API key password input.
 */
function toggleKeyVisibility() {
  const keyInput = document.getElementById("geminiApiKey");
  if (!keyInput) return;
  if (keyInput.type === "password") {
    keyInput.type = "text";
  } else {
    keyInput.type = "password";
  }
}

/**
 * Initializes upload form validation for both File and YouTube inputs.
 */
function initFormHandler() {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("videoFile");
  const youtubeInput = document.getElementById("youtubeUrl");
  const selectedBanner = document.getElementById("selectedFileBanner");
  const selectedName = document.getElementById("selectedFileName");
  const highlightForm = document.getElementById("highlightForm");
  const submitBtn = document.getElementById("submitBtn");
  const uploadTypeInput = document.getElementById("uploadType");

  if (dropzone && fileInput) {
    dropzone.addEventListener("click", () => fileInput.click());

    ["dragenter", "dragover"].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add("dragover");
      }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove("dragover");
      }, false);
    });

    dropzone.addEventListener("drop", (e) => {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files.length > 0) {
        fileInput.files = files;
        handleFileSelected(files[0]);
      }
    });

    fileInput.addEventListener("change", () => {
      if (fileInput.files.length > 0) {
        handleFileSelected(fileInput.files[0]);
      }
    });
  }

  function handleFileSelected(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (ext !== "mp4") {
      alert("Please select a valid MP4 video file (.mp4).");
      fileInput.value = "";
      if (selectedBanner) selectedBanner.style.display = "none";
      return;
    }

    const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
    if (file.size > 500 * 1024 * 1024) {
      alert("File size exceeds 500 MB limit. Please select a smaller video.");
      fileInput.value = "";
      if (selectedBanner) selectedBanner.style.display = "none";
      return;
    }

    if (selectedName) selectedName.textContent = `${file.name} (${sizeMB} MB)`;
    if (selectedBanner) selectedBanner.style.display = "flex";
  }

  if (highlightForm) {
    highlightForm.addEventListener("submit", (e) => {
      const currentType = uploadTypeInput ? uploadTypeInput.value : "file";

      if (currentType === "youtube") {
        const url = youtubeInput ? youtubeInput.value.trim() : "";
        if (!url) {
          e.preventDefault();
          alert("Please enter a YouTube video link.");
          if (youtubeInput) youtubeInput.focus();
          return;
        }

        const ytRegex = /^(https?:\/\/)?(www\.|m\.)?(youtube\.com\/(watch\?v=|embed\/|v\/|shorts\/)|youtu\.be\/)([\w-]{11})([^\s]*)$/;
        if (!ytRegex.test(url)) {
          e.preventDefault();
          alert("Invalid YouTube URL! Please provide a standard link (youtube.com/watch?v=...), short link (youtu.be/...), or Shorts link.");
          if (youtubeInput) youtubeInput.focus();
          return;
        }
      } else {
        if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
          e.preventDefault();
          alert("Please select an MP4 video file to upload.");
          return;
        }
      }

      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = `
          <svg class="animate-spin" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-opacity="0.25"></circle>
            <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" stroke-linecap="round"></path>
          </svg>
          Initializing Processing...
        `;
      }
    });
  }
}

/**
 * Polls the job status endpoint in processing.html and maps to real pipeline stages.
 */
function initProcessingPoller() {
  const pollerElement = document.getElementById("processingPoller");
  if (!pollerElement) return;

  const jobId = pollerElement.getAttribute("data-job-id");
  const stageStatusText = document.getElementById("stageStatusText");
  const errorMessageContainer = document.getElementById("errorMessageContainer");
  const stageItems = document.querySelectorAll(".stage-item");

  if (!jobId) return;

  let pollInterval = setInterval(checkStatus, 1500);

  function checkStatus() {
    fetch(`/api/status/${jobId}`)
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        if (data.status === "processing" || data.status === "queued") {
          if (stageStatusText) {
            stageStatusText.textContent = data.stage || "Processing video...";
          }
          updateStageIndicators(data.stage);
        } else if (data.status === "complete") {
          clearInterval(pollInterval);
          if (stageStatusText) {
            stageStatusText.textContent = "Highlight video ready! Redirecting...";
          }
          setTimeout(() => {
            window.location.href = data.redirect_url || `/result/${jobId}`;
          }, 800);
        } else if (data.status === "error") {
          clearInterval(pollInterval);
          showError(data.error || "An error occurred during video processing.");
        }
      })
      .catch(err => {
        console.warn("Polling status error:", err);
      });
  }

  function updateStageIndicators(currentStage) {
    if (!currentStage) return;
    const s = currentStage.toLowerCase();

    // Map stages across the 11-stage list
    let activeIndex = 0;
    if (s.includes("getting youtube") || s.includes("validat") || s.includes("upload") || s.includes("init")) {
      activeIndex = 0;
    } else if (s.includes("fast video analysis") || s.includes("visual pre-screening") || s.includes("0.5 fps")) {
      activeIndex = 1;
    } else if (s.includes("speech activity") || s.includes("rms energy") || s.includes("audio pre-screening")) {
      activeIndex = 2;
    } else if (s.includes("candidate moment") || s.includes("candidate window") || s.includes("finding candidate")) {
      activeIndex = 3;
    } else if (s.includes("transcribing selected") || s.includes("candidate speech") || s.includes("faster-whisper")) {
      activeIndex = 4;
    } else if (s.includes("understanding candidate") || s.includes("minilm") || s.includes("text embed")) {
      activeIndex = 5;
    } else if (s.includes("ranking important") || s.includes("scoring candidate") || s.includes("predict")) {
      activeIndex = 6;
    } else if (s.includes("creating multiple reels") || s.includes("sentence-snapping") || s.includes("nms") || s.includes("select")) {
      activeIndex = 7;
    } else if (s.includes("converting to 9:16") || s.includes("9:16") || s.includes("center crop")) {
      activeIndex = 8;
    } else if (s.includes("adding captions") || s.includes("subtitle") || s.includes("caption")) {
      activeIndex = 9;
    } else if (s.includes("finalizing") || s.includes("manifest") || s.includes("render")) {
      activeIndex = 10;
    }

    stageItems.forEach((item, idx) => {
      if (idx < activeIndex) {
        item.classList.remove("active");
        item.classList.add("completed");
        const ind = item.querySelector(".stage-indicator");
        if (ind) ind.textContent = "✓";
      } else if (idx === activeIndex) {
        item.classList.add("active");
        item.classList.remove("completed");
        const ind = item.querySelector(".stage-indicator");
        if (ind) ind.textContent = "●";
      } else {
        item.classList.remove("active", "completed");
        const ind = item.querySelector(".stage-indicator");
        if (ind) ind.textContent = `${idx + 1}`;
      }
    });
  }

  function showError(msg) {
    if (errorMessageContainer) {
      errorMessageContainer.style.display = "block";
      errorMessageContainer.innerHTML = `
        <div class="alert alert-danger">
          <strong>Processing Error:</strong> ${msg}
        </div>
        <div style="margin-top: 1.5rem;">
          <a href="/" class="btn-secondary">← Back to Home</a>
        </div>
      `;
    }
    const spinner = document.querySelector(".spinner-outer");
    if (spinner) spinner.style.display = "none";
    if (stageStatusText) stageStatusText.style.display = "none";
  }
}
