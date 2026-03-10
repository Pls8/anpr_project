# Omani ANPR Project

Automatic Number Plate Recognition for Omani License Plates

## Dataset

- **Source**: [Kaggle - Oman Licence Plates](https://www.kaggle.com/datasets/koti4878m/oman-licenceplates)
- **Train**: 9,906 labeled images
- **Validation**: 2,466 labeled images
- **Format**: Cropped license plate images with metadata.jsonl containing labels
- **Plate Format**: 1-8 characters (numbers + letters, e.g., "3598TH", "15833R", "LA12 AAA")

## Project Structure

```
anpr_project/
├── best_simple_model.pth     # Trained OCR model (63.5% accuracy)
├── config.json               # Project configuration
├── README.md                # This file
└── src/
    ├── anpr_app.py          # Main ANPR library
    ├── complete_anpr.py      # Full ANPR with YOLO detection
    ├── simple_detector.py   # Simple contour-based detection
    ├── video_anpr.py        # Video processing
    ├── train_simple.py      # Train CNN character classifier
    ├── test_simple.py        # Test trained model
    ├── inference.py          # EasyOCR inference
    ├── data_explorer.py      # Explore dataset
    ├── model.py              # CRNN model (legacy)
    ├── dataset.py            # Dataset class
    └── train.py              # CRNN training (legacy)
```

## Quick Start

### Option 1: Use Pretrained Model (Already Trained)

```bash
# Set ROCm environment
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export LD_PRELOAD=/opt/rocm/lib/libamdhip64.so

# Run on a single plate image (dataset images are already cropped)
python src/anpr_app.py path/to/plate_image.png

# Or use the simple detector (for full images)
python src/simple_detector.py path/to/image.jpg
```

### Option 2: Process Video

```bash
python src/video_anpr.py input_video.mp4 output_video.mp4
```

### Option 3: Full ANPR with Detection (Requires YOLO)

```bash
# Install YOLO
pip install ultralytics

# Run complete ANPR (detects vehicles + reads plates)
python src/complete_anpr.py input.jpg -o output.jpg
```

## Usage Examples

### Python API

```python
from anpr_app import ANPR

anpr = ANPR()
plate = anpr.predict("plate_image.png")
print(plate)  # e.g., "9050YM"
```

### Batch Processing

```python
from anpr_app import ANPR
import os

anpr = ANPR()
results = []
for f in os.listdir("plates/"):
    if f.endswith(".png"):
        plate = anpr.predict(f"plates/{f}")
        results.append({"file": f, "plate": plate})
        print(f"{f}: {plate}")
```

## Model Performance

- **Simple CNN Classifier**: 63.5% validation accuracy
- Trained for 50 epochs on Omani license plates
- Handles variable length plates (1-8 characters)

## Training (If You Want to Retrain)

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export LD_PRELOAD=/opt/rocm/lib/libamdhip64.so
python src/train_simple.py --epochs 100 --batch_size 64
```

## Hardware Requirements

- GPU with 8GB+ VRAM (tested on AMD RX 7900 XT 20GB with ROCm 6.1.3)
- 16GB+ RAM recommended
- WSL2 with Ubuntu 22.04

## ROCm Setup

```bash
# Add ROCm repository
wget https://repo.radeon.com/rocm/rocm.gpg.key
sudo apt-key add rocm.gpg.key
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/ubuntu/debian xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rocm-6.1.3

# Verify GPU
rocminfo
```

## License

MIT
