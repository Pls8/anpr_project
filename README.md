# ANPR System for Omani License Plates

Automatic Number Plate Recognition system using YOLO for detection and EfficientNet for OCR.

## Requirements

### Hardware
- AMD GPU (RX 7900 XT 20GB recommended)
- WSL2 with ROCm installed
- ~24GB disk space

### Software
- Ubuntu on WSL2
- ROCm 5.7+
- Python 3.10+
- PyTorch with ROCm support
- Ultralytics (YOLO)
- OpenCV

## Installation

```bash
# Install ROCm (in WSL)
sudo apt update
sudo apt install rocm-libs hipblaslt

# Install Python dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
pip install ultralytics opencv-python pillow

# Set environment variables
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export LD_PRELOAD=/opt/rocm/lib/libamdhip64.so
```

## Project Structure

```
Test3OC/
├── best_improved_ocr_v5.pth          # Trained OCR model (79.78% accuracy)
├── runs/detect/                      # YOLO trained models
│   └── oman_plate_detector_yolov8s/weights/best.pt
├── anpr_project/
│   ├── src/
│   │   ├── anpr_yolo_app.py         # Main ANPR application
│   │   ├── train_improved_ocr_v5.py # OCR training script
│   │   └── web_app.py               # Flask web interface
│   └── test_images/                 # Test images
└── .gitignore
```

## Usage

### Python API
```python
from anpr_yolo_app import ANPR

anpr = ANPR()
result = anpr.predict('path/to/image.jpg')
print(result)  # e.g., "81080W"
```

### Web App
```bash
cd anpr_project/src
python web_app.py
# Open http://localhost:5000
```

### CLI Test
```bash
cd anpr_project/src
python -c "
from anpr_yolo_app import ANPR
anpr = ANPR()
print(anpr.predict('../test_images/OGTRC-1.png'))
"
```

## Model Performance

| Model | Metric | Value |
|-------|--------|-------|
| YOLOv8s Detection | mAP50 | 0.991 |
| EfficientNet-B0 OCR | Val Accuracy | 79.78% |

## Plate Format

Supports Oman license plates:
- **OGTRC**: 1-5 digits + 1-2 suffix letters (e.g., "9466B", "7347BR")
- **Regular**: 1-5 digits + 1-2 suffix letters (e.g., "81080W")

## Training

### OCR Training
```bash
cd anpr_project/src
python train_improved_ocr_v5.py
```

### YOLO Training
```bash
yolo detect train data=plate_data.yaml model=yolov8s.yaml
```

## Troubleshooting

### GPU Not Detected
- Ensure ROCm is installed: `rocminfo`
- Checkhip version: `hipconfig`
- Try: `export HSA_OVERRIDE_GFX_VERSION=11.0.0`

### TDR (Timeout Detection and Recovery)
If training causes GPU hang:
- Add registry key: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers\TdrDelay` = 8

## Files Description

| File | Description |
|------|-------------|
| `best_improved_ocr_v5.pth` | Current best OCR model |
| `best_improved_ocr_v5_augmented.pth` | OCR with augmentation |
| `runs/detect/.../best.pt` | YOLOv8s detector |
