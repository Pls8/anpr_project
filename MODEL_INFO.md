# Model Information

This document describes the trained models included in this release.

## Included Files

| File | Description |
|------|-------------|
| `best_improved_ocr_v5.pth` | **Main OCR Model** - Best accuracy (79.78%) on standard images |
| `best_improved_ocr_v5_augmented.pth` | **Alternate OCR Model** - Better generalization on varied conditions |
| `oman_plate_detector_yolov8s/weights/best.pt` | **YOLO Plate Detector** - Detects license plates in images |

---

## Which Model to Use

### OCR Models

| Situation | Recommended Model |
|-----------|------------------|
| Standard/clear images | `best_improved_ocr_v5.pth` |
| Varied conditions (dark, blurry, crooked) | `best_improved_ocr_v5_augmented.pth` |

The augmented model was trained with rotation, brightness, and perspective transforms - making it more robust for real-world scenarios.

### Switching OCR Model

Edit `anpr_project/src/anpr_yolo_app.py`, line 22:

```python
# For standard images:
OCR_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'best_improved_ocr_v5.pth')

# For varied conditions:
OCR_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'best_improved_ocr_v5_augmented.pth')
```

---

## YOLO Detector

The YOLOv8s detector (`best.pt`) finds license plates in images before OCR runs.

- **mAP50**: 0.991 (99.1% detection rate)
- Detects plates in full images
- Crops plate region for OCR

---

## Performance

| Component | Metric | Value |
|-----------|--------|-------|
| YOLOv8s Detection | mAP50 | 0.991 |
| OCR v5 (main) | Val Accuracy | 79.78% |
| OCR v5 (augmented) | Val Accuracy | 79.27% |

---

## Requirements

- Python 3.10+
- PyTorch (CPU or GPU)
- OpenCV
- Ultralytics

Install: `pip install -r requirements.txt`

---

## Usage

```python
from anpr_project.src.anpr_yolo_app import ANPR

anpr = ANPR()
result = anpr.predict('path/to/image.jpg')
print(result)  # e.g., "81080W"
```

Or run the web app:

```bash
cd anpr_project/src
python web_app.py
# Open http://localhost:5000
```
