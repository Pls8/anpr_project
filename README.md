<div align="center">
  <img src="https://github.com/Pls8/anpr_project/blob/main/Repo-anpr_002_00.jpg?raw=true" >
</div>

# ANPR System for Omani License Plates

Automatic Number Plate Recognition system using YOLO for detection and EfficientNet for OCR.

## Quick Start (For Users)

```bash
# 1. Clone the repo
git clone https://github.com/Pls8/anpr_project.git
cd anpr_project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the web app
python anpr_project/src/web_app.py

# 4. Open http://localhost:5000 in your browser
```

## Requirements

### Hardware
- AMD GPU (RX 7900 XT 20GB recommended) - optional, runs on CPU
- ~24GB disk space

### Software
- Python 3.10+
- PyTorch
- Ultralytics (YOLO)
- OpenCV

## Model Performance

| Model | Metric | Value |
|-------|--------|-------|
| YOLOv8s Detection | mAP50 | 0.991 |
| EfficientNet-B0 OCR (v5) | Val Accuracy | **79.78%** |
| EfficientNet-B0 Augmented | Val Accuracy | 79.27% |
| EfficientNet-B0 v7 | Val Accuracy | 79.53% |

## Available OCR Models

| File | Description | Accuracy |
|------|-------------|----------|
| `best_improved_ocr_v5.pth` | Main model (recommended) | 79.78% |
| `best_improved_ocr_v5_augmented.pth` | Better generalization | 79.27% |
| `best_improved_ocr_v7.pth` | Continued from augmented | 79.53% |

**Note**: EfficientNet-B1 was tested but did not outperform B0. See PROJECT_README.md for details.

## Plate Format

Supports Oman license plates:
- **OGTRC**: 1-5 digits + 1-2 suffix letters (e.g., "9466B", "7347BR")
- **Regular**: 1-5 digits + 1-2 suffix letters (e.g., "81080W")

## Files Included

| File | Description |
|------|-------------|
| `best_improved_ocr_v5.pth` | Main OCR model (79.78% accuracy) - RECOMMENDED |
| `best_improved_ocr_v5_augmented.pth` | OCR with data augmentation (better generalization) |
| `best_improved_ocr_v7.pth` | Trained from augmented model (79.53%) |
| `runs/detect/.../best.pt` | YOLOv8s plate detector (mAP50: 0.991) |

## Features

- **Image Upload**: Upload car images for plate detection
- **Video Streaming**: Live camera feed for real-time detection (2 options)
- **Ensemble Prediction**: Uses both OCR models for better accuracy

## Web App Routes

- `http://localhost:5000/` - Image upload
- `http://localhost:5000/video` - Live video stream (OpenCV-based)
- `http://localhost:5000/video2` - Live video stream (Browser getUserMedia - recommended)

## For Training (Optional)

See `anpr_project/src/train_improved_ocr_v5.py` for OCR training.

## Troubleshooting

### GPU Not Detected (Training Only)
- Install ROCm: `sudo apt install rocm-libs hipblaslt`
- Set environment: `export HSA_OVERRIDE_GFX_VERSION=11.0.0`

---

## Architecture & How It Works

### System Overview

This ANPR system uses a **two-stage pipeline**:

```
Input Image/Video → [YOLO Detector] → Crop Plate → [EfficientNet OCR] → Plate Text
```

### Components

| Component | Model | Purpose |
|-----------|-------|---------|
| **Detector** | YOLOv8s | Finds license plate location in image |
| **OCR** | EfficientNet-B0 | Reads text from cropped plate image |

### Core Module: `anpr_yolo_app.py`

The `ANPR` class (`anpr_project/src/anpr_yolo_app.py`) is the core module that handles:
- Loading YOLO and OCR models
- Plate detection using YOLO
- Image preprocessing (CLAHE for contrast enhancement)
- OCR prediction with ensemble (uses 2 OCR models)
- Post-processing (removes extra spaces, handles Oman plate format)

```python
from anpr_yolo_app import ANPR

anpr = ANPR()  # Loads models
result = anpr.predict("car_image.jpg")  # Returns plate text
```

---

## Workflows

### Workflow 1: Image Upload (/)

```
┌──────────┐    ┌─────────────┐    ┌──────────────┐    ┌────────────┐
│ User     │    │ Flask       │    │ ANPR Class   │    │ Response  │
│ selects  │───▶│ receives    │───▶│ (YOLO+OCR)  │───▶│ with plate │
│ image    │    │ POST /upload│    │ processes   │    │ text      │
└──────────┘    └─────────────┘    └──────────────┘    └────────────┘
```

**Steps:**
1. User selects image file in browser
2. Browser sends POST request with image to `/upload`
3. Flask saves image temporarily
4. `ANPR.predict()` processes image:
   - YOLO detects plate location
   - Plate is cropped
   - CLAHE preprocessing applied
   - EfficientNet OCR reads text
   - Ensemble prediction (2 models)
   - Post-processing (format fix)
5. Returns plate text to browser

---

### Workflow 2: Video Streaming - OpenCV (/video)

```
┌──────────┐    ┌─────────────┐    ┌──────────────┐    ┌────────────┐
│ Browser  │    │ Flask       │    │ Video       │    │ Browser   │
│ starts   │───▶│ opens       │───▶│ capture     │───▶│ displays  │
│ camera   │    │ VideoCapture│    │ thread      │    │ MJPEG     │
└──────────┘    └─────────────┘    └──────────────┘    └────────────┘
                     │                    │
                     │                    ▼
                     │             ┌──────────────┐
                     └────────────▶│ ANPR Class   │
                                  │ (YOLO+OCR)  │
                                  └──────────────┘
```

**Steps:**
1. User clicks "Start Camera" in browser
2. Browser sends POST to `/video/start`
3. Flask opens OpenCV VideoCapture (camera index 0)
4. Background thread continuously:
   - Captures frames from camera
   - Sends to ANPR for processing
   - Draws result on frame
5. Flask streams frames as MJPEG via `/video/feed`

**Issue:** May have camera access problems in some browsers.

---

### Workflow 3: Video Streaming - Browser getUserMedia (/video2) [Recommended]

```
┌──────────┐    ┌─────────────┐    ┌──────────────┐    ┌────────────┐
│ Browser  │    │ Browser     │    │ Canvas      │    │ Browser   │
│ accesses │───▶│ getUserMedia│───▶│ captures    │───▶│ displays  │
│ /video2  │    │ API         │    │ frame       │    │ video     │
└──────────┘    └─────────────┘    └──────────────┘    └────────────┘
                                           │
                                           ▼
                                    ┌──────────────┐
                                    │ POST /upload │
                                    │ (sends blob) │
                                    └──────────────┘
                                           │
                                           ▼
                                    ┌──────────────┐
                                    │ ANPR Class   │
                                    │ (YOLO+OCR)   │
                                    └──────────────┘
```

**Steps:**
1. User opens `/video2` page
2. Browser uses native `getUserMedia` API to access camera
3. Every 500ms, browser:
   - Captures frame to canvas
   - Converts to blob
   - Sends POST to `/upload`
4. Server processes and returns plate text
5. Browser displays result

**Why it's recommended:** Uses browser's built-in camera API, works better on most systems.

---

### Why Video Is Laggy (And How to Improve)

**Current bottleneck:**
- Each frame → HTTP POST → Server processes → HTTP response → Display
- Even on localhost, HTTP adds overhead per frame

**Solutions:**

| Option | Implementation | Latency |
|--------|---------------|---------|
| **Reduce frame rate** | Process every 5th frame | Medium |
| **WebSocket** | Keep connection open | Faster |
| **WebRTC** | P2P streaming | Fastest |
| **Client-side OCR** | Run model in browser (TensorFlow.js) | Fastest but complex |

For now, `/video2` processes every 500ms which should be smoother. If still laggy, we can:
1. Reduce processing frequency
2. Add a simpler/faster model
3. Consider local-only option (like Tkinter)

---

### Training Workflow

```
┌──────────┐    ┌─────────────┐    ┌──────────────┐    ┌────────────┐
│ Training │    │ Load       │    │ Train       │    │ Save best  │
│ script   │───▶│ dataset    │───▶│ EfficientNet │───▶│ model      │
│          │    │ (JSONL)    │    │ (GPU/CPU)   │    │ (.pth)     │
└──────────┘    └─────────────┘    └──────────────┘    └────────────┘
```

See `anpr_project/src/train_improved_ocr_v5.py` for training code.

---

### File Structure

```
anpr_project/
├── src/
│   ├── anpr_yolo_app.py      # Core ANPR class (YOLO + OCR)
│   ├── web_app.py            # Flask web server
│   ├── train_improved_ocr_v5.py  # OCR training script
│   └── templates/
│       ├── index.html        # Main upload page
│       ├── video.html        # OpenCV video page
│       └── video2.html       # Browser getUserMedia video
├── test_images/              # Test images
static/uploads/               # Uploaded images (runtime)
```

### Key Environment Variables (for AMD GPU/ROCm)

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export LD_PRELOAD=/opt/rocm/lib/libamdhip64.so
```
