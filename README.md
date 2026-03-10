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
| EfficientNet-B0 OCR | Val Accuracy | 79.78% |

## Plate Format

Supports Oman license plates:
- **OGTRC**: 1-5 digits + 1-2 suffix letters (e.g., "9466B", "7347BR")
- **Regular**: 1-5 digits + 1-2 suffix letters (e.g., "81080W")

## Files Included

| File | Description |
|------|-------------|
| `best_improved_ocr_v5.pth` | Main OCR model (79.78% accuracy) |
| `best_improved_ocr_v5_augmented.pth` | OCR with data augmentation |
| `runs/detect/.../best.pt` | YOLOv8s plate detector |

## For Training (Optional)

See `anpr_project/src/train_improved_ocr_v5.py` for OCR training.

## Troubleshooting

### GPU Not Detected (Training Only)
- Install ROCm: `sudo apt install rocm-libs hipblaslt`
- Set environment: `export HSA_OVERRIDE_GFX_VERSION=11.0.0`
