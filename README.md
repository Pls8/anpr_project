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
