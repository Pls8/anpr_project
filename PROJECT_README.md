# ANPR System for Omani License Plates

## Project Overview

Automatic Number Plate Recognition (ANPR) system for detecting and reading Omani license plates using AMD GPU (RX 7900 XT 20GB) with WSL2 and ROCm.

---

## Hardware & Environment

| Component           | Details                          |
| ------------------- | -------------------------------- |
| **OS**              | Windows 11 + WSL2 (Ubuntu 22.04) |
| **GPU**             | AMD Radeon RX 7900 XT 20GB       |
| **ROCm Version**    | 6.1.3                            |
| **Python Version**  | 3.10                             |
| **PyTorch Version** | 2.5.1+rocm6.1                    |

### WSL Environment Setup

```bash
# ROCm environment variables (required for GPU training/inference)
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export LD_PRELOAD=/opt/rocm/lib/libamdhip64.so

# Virtual environment
~/anpr_project/venv/bin/activate
```

---

## Oman License Plate Format

```
__________________________________________
|                 |      ب ب        |                           |
|   12345    |                    |    عُـمـان               |
|                 |      B B        |                           |
|_________________________________________|
```

### Plate Types:

| Type                     | Format                          | Example               |
| ------------------------ | ------------------------------- | --------------------- |
| **OGTRC** (Number Plate) | 1-5 digits + 1-2 suffix letters | 9466B, 7347BR, 3566DD |
| **Regular**              | 1-5 digits + 1-2 suffix letters | 81080W, 10080W        |

**Key Points:**
- Digits: English numbers only (1-5 digits)
- Suffix: Arabic letters (top row) + English letters (bottom row), max 2 characters
- "عمان" (Oman) country identifier on right side of plate
- Plate width: English digits on left, Arabic country name on right

---

## Project Structure

```
Test3OC/
├── anpr_project/
│   ├── src/
│   │   ├── train_improved_ocr.py          # OCR training v4
│   │   ├── train_improved_ocr_v5.py      # OCR training v5 (current)
│   │   ├── anpr_yolo_app.py             # ANPR class (YOLO + OCR)
│   │   ├── web_app.py                   # Flask web interface
│   │   ├── improved_detector.py          # Improved detector
│   │   └── test_images/                  # Test images
│   └── venv/                             # Python virtual environment
├── oman-licenceplates/
│   ├── dataset/                          # OCR training data (~9,900 images)
│   │   ├── train/metadata.jsonl
│   │   └── validation/metadata.jsonl
│   ├── yolo_dataset/                     # YOLO format dataset
│   │   ├── dataset.yaml
│   │   ├── images/train/
│   │   ├── images/valid/
│   │   └── labels/
│   └── Oman ANPR.v2i.coco/               # Original COCO dataset (1,686 images)
├── runs/detect/                          # YOLO trained models
│   ├── oman_plate_detector/              # YOLOv8n (nano) - BEST mAP50: 0.990
│   └── oman_plate_detector_yolov8s/     # YOLOv8s (small) - BEST mAP50: 0.991
├── best_improved_ocr_v4.pth              # OCR model v4 (79.23%)
├── best_improved_ocr_v5.pth              # OCR model v5 (79.78%) - CURRENT
├── best_improved_ocr_v5_augmented.pth    # OCR v5 with augmentation (79.27%)
├── best_improved_ocr_v5_checkpoint.pth  # Training checkpoint
└── PROJECT_README.md                    # This file
```

---

## Models

### OCR Model (EfficientNet-B0)

| Version     | Architecture    | Accuracy   | Epochs | Alphabet     | Notes                              |
| ----------- | --------------- | ---------- | ------ | ----------- | ---------------------------------- |
| Simple CNN  | Custom CNN      | 63.5%     | -      | 37 chars    | Initial model                      |
| ResNet      | ResNet18        | -          | -      | 38 chars    | Earlier attempt                    |
| v3          | EfficientNet-B0 | 77%        | 30     | 37 chars    | Improved                           |
| v4          | EfficientNet-B0 | 79.23%     | 50     | 37 chars    | Previous best                      |
| **v5**      | **EfficientNet-B0** | **79.78%** | **50** | **38 chars** | **Current best** (no augmentation) |
| v5 (aug)    | EfficientNet-B0 | 79.27%     | 50     | 38 chars    | With augmentation (better general.) |

#### v5 Training Parameters:

- **Epochs:** 50 (completed)
- **Batch Size:** 32
- **Label Smoothing:** 0.1
- **Data Augmentation (training):**
  - Rotation: ±15°
  - Perspective transform (distortion_scale=0.2)
  - Brightness/contrast: 0.3
  - Translation: 10%
- **Max Plate Length:** 7 characters
- **Alphabet:** `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ _` (38 chars)
- **Blank token:** `_` at index 37
- **Optimizer:** AdamW (lr=0.001, weight_decay=0.0001)
- **Scheduler:** CosineAnnealingLR (T_max=50)
- **GPU Training:** Auto-detect CUDA (was hardcoded to CPU - FIXED)

#### Post-Processing (v5):

- Decode stops at first blank token
- Removes spaces between digits and suffix (fixes "9466 B" → "9466B")
- Keeps max 2-letter suffix (Oman format)
- Handles OGTRC plates correctly (e.g., "9466BR" stays "9466BR")

### Detection Model (YOLO)

| Model       | Parameters | mAP50  | mAP50-95 | Precision | Recall | Best Epoch |
| ----------- | ---------- | ------ | -------- | --------- | ------ | ---------- |
| **YOLOv8n** | 3M         | 0.990  | 0.713    | 0.978    | 0.968  | 35         |
| **YOLOv8s** | 11M        | 0.991  | 0.706    | 0.981    | 0.964  | 34         |

**Note:** Both YOLO models achieved excellent detection (mAP50 > 0.99). YOLOv8n is smaller/faster, YOLOv8s is slightly more accurate.

#### YOLO Training Parameters:

- **Epochs:** 50 (stopped early at ~34-35)
- **Batch Size:** 8
- **Image Size:** 640x640
- **Optimizer:** AdamW (lr=0.002)
- **Patience:** 10
- **Early Stopping:** Triggered at epoch 34-35

#### YOLO Dataset:

| Split      | Images |
| ---------- | ------ |
| Train      | 1,180  |
| Validation | 337    |
| Test       | 169    |

---

## Training Commands

### OCR Training (v5) - Current Best

```bash
cd /mnt/c/Users/Masad/Documents/aiPorject/Test3OC/anpr_project/src
source ~/anpr_project/venv/bin/activate
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export LD_PRELOAD=/opt/rocm/lib/libamdhip64.so
python train_improved_ocr_v5.py
```

### Resume Training from Checkpoint

Training auto-resumes from last epoch if checkpoint exists.

### YOLO Training (v8n/v8s)

```bash
cd /home/night/anpr_project
source venv/bin/activate
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export LD_PRELOAD=/opt/rocm/lib/libamdhip64.so

python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # or 'yolov8s.pt'
model.train(
    data='/mnt/c/Users/Masad/Documents/aiPorject/Test3OC/oman-licenceplates/yolo_dataset/dataset.yaml',
    epochs=50,
    imgsz=640,
    batch=8,
    name='oman_plate_detector',
    project='/mnt/c/Users/Masad/Documents/aiPorject/Test3OC/runs/detect',
    device=0
)
"
```

---

## Web Interface

### Running the Web App

```bash
cd /mnt/c/Users/Masad/Documents/aiPorject/Test3OC/anpr_project/src
source ~/anpr_project/venv/bin/activate
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export LD_PRELOAD=/opt/rocm/lib/libamdhip64.so
python web_app.py
```

Then open **http://localhost:5000** in browser.

### Web App Files:

- `web_app.py` - Flask backend (imports from `anpr_yolo_app.py`)
- `anpr_yolo_app.py` - ANPR class with YOLO + OCR (includes CLAHE preprocessing)
- `templates/index.html` - HTML frontend
- `static/uploads/` - Uploaded images

---

## Testing

### Test Images Location

```
anpr_project/test_images/
├── IMG20260128065157.jpg        # Full car (81080 W)
├── IMG20260128065200.jpg        # Full car (81080 W)
├── IMG20260128065245.jpg        # Full car (81080 W)
├── IMG-20260304-WA0007001.jpg  # Full car (81080 W)
├── OGTRC-1.png                  # OGTRC plate (9466B)
├── OGTRC-2.png                  # OGTRC plate (7347BR)
├── OGTRC-3.png                  # OGTRC plate (3566DD)
├── crop-81080-W.png             # Cropped plate (81080W)
└── crop-*.png                  # Other cropped plates
```

### Test Results (OCR v5 with CLAHE preprocessing)

| Image                       | Detection | OCR Result |
| --------------------------- | --------- | ---------- |
| IMG20260128065157.jpg      | ✅        | "81080W"  |
| IMG20260128065200.jpg      | ✅        | "81080W"  |
| IMG20260128065245.jpg      | ✅        | "81080W"  |
| IMG-20260304-WA0007001.jpg | ✅        | "81080W"  |
| OGTRC-1.png                | ✅        | "9466BR"  |
| OGTRC-2.png                | ✅        | "7347BR"  |
| OGTRC-3.png                | ✅        | "3566DD"  |
| crop-81080-W.png            | N/A       | "81080W"  |
| crop-2026-03-07 081102.png | N/A       | "81080"   |
| crop-2026-03-07 081122.png | N/A       | "81080W"  |
| crop-2026-03-07 081131.png | N/A       | "81080W"  |
| crop-2026-03-07 081403.png | N/A       | "81080W"  |

---

## Key Fixes & Solutions

### Fix 1: GPU Training Not Working

- **Problem:** Training script had `device = 'cpu'` hardcoded
- **Solution:** Changed to `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
- **File:** `train_improved_ocr_v5.py`

### Fix 2: Checkpoint/Resume Training

- **Problem:** Training interrupted, couldn't resume
- **Solution:** Added checkpoint saving/loading with epoch, best_acc, optimizer, scheduler
- **File:** `train_improved_ocr_v5.py`

### Fix 3: Variable Length Plates (v5)

- **Problem:** OCR always outputs 7 chars (e.g., "9466" → "94660000")
- **Solution:** Added blank token `_` to alphabet, stop at first blank during inference
- **Files:** `train_improved_ocr_v5.py`, `anpr_yolo_app.py`

### Fix 4: Trailing Character Issue

- **Problem:** OGTRC plates showed extra chars (e.g., "9466 B" instead of "9466B")
- **Solution:** Post-processing to remove spaces, keep max 2-letter suffix
- **File:** `anpr_yolo_app.py`

### Fix 5: CLAHE Preprocessing

- **Problem:** Low contrast images reduce OCR accuracy
- **Solution:** Added CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing
- **File:** `anpr_yolo_app.py`

---

## Pre-trained Models Summary

| Model Type | File | Accuracy/mAP | Status |
|------------|------|--------------|--------|
| OCR v5 (best) | `best_improved_ocr_v5.pth` | 79.78% val acc | ✅ Ready |
| OCR v5 (aug) | `best_improved_ocr_v5_augmented.pth` | 79.27% val acc | ✅ Ready |
| YOLOv8n | `runs/detect/oman_plate_detector/weights/best.pt` | mAP50: 0.990 | ✅ Ready |
| YOLOv8s | `runs/detect/oman_plate_detector_yolov8s/weights/best.pt` | mAP50: 0.991 | ✅ Ready |

---

## Files Created/Modified

| File                                                     | Description                          |
| -------------------------------------------------------- | ------------------------------------ |
| `anpr_project/src/train_improved_ocr.py`                 | OCR training v4                     |
| `anpr_project/src/train_improved_ocr_v5.py`              | OCR training v5 (variable length)   |
| `anpr_project/src/anpr_yolo_app.py`                      | ANPR class with YOLO + OCR + CLAHE  |
| `anpr_project/src/web_app.py`                           | Flask web interface                  |
| `convert_coco_to_yolo.py`                               | COCO to YOLO converter               |
| `test_checkpoint.py`                                    | Check training checkpoint status     |
| `runs/detect/oman_plate_detector/weights/best.pt`       | Trained YOLOv8n                     |
| `runs/detect/oman_plate_detector_yolov8s/weights/best.pt` | Trained YOLOv8s                     |
| `best_improved_ocr_v5.pth`                              | Trained OCR v5 (79.78%)             |
| `best_improved_ocr_v5_augmented.pth`                    | OCR v5 with augmentation (79.27%)   |
| `PROJECT_README.md`                                     | This documentation file              |

---

## Next Steps for Next Agent

### Completed Tasks ✅
1. ✅ Data augmentation added to training
2. ✅ CLAHE preprocessing added to inference
3. ✅ OCR v5 trained (79.78%)
4. ✅ OCR v5 with augmentation trained (79.27%)
5. ✅ YOLOv8n and YOLOv8s trained

### Possible Improvements (If Needed)

1. **Collect more training data** - Current dataset ~10k images, more diverse data could help
2. **Try larger OCR model** - EfficientNet-B2 (but risk overfitting with ~10k images)
3. **Add more augmentation** - Blur, noise, color jitter during training
4. **Implement auto-deskew** - For heavily tilted plates
5. **Fine-tune on specific failure cases** - Identify and address specific error patterns

---

## ROCm/WSL Notes

- Use `HSA_OVERRIDE_GFX_VERSION=11.0.0` for RX 7900 XT (RDNA3 architecture)
- Use `LD_PRELOAD=/opt/rocm/lib/libamdhip64.so` for GPU access
- Install in WSL2 Ubuntu 22.04
- GPU stability: Add `TdrDelay 8` registry setting in Windows (prevents timeout)

### Testing GPU in WSL

```bash
# Check ROCm installation
rocminfo

# Check PyTorch GPU access
python -c "import torch; print(torch.cuda.is_available())"
```

---

## References

- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- PyTorch ROCm: https://pytorch.org/
- ROCm 6.1.3: https://rocm.docs.amd.com/
- EfficientNet: https://arxiv.org/abs/1905.11946

---

## Quick Status Check Commands

```bash
# Check OCR v5 checkpoint
python test_checkpoint.py

# Check YOLO models
ls -la runs/detect/oman_plate_detector/weights/
ls -la runs/detect/oman_plate_detector_yolov8s/weights/

# Check OCR models
ls -la best_improved_ocr*.pth

# Test OCR model
cd anpr_project/src
python -c "
import sys
sys.path.insert(0, '.')
from anpr_yolo_app import ANPR
anpr = ANPR()
print(anpr.predict('../test_images/OGTRC-1.png'))
"
```
