"""
ANPR Application using YOLO Detection + EfficientNet OCR
Updated to use trained YOLO detector and EfficientNet-B0 OCR
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
from PIL import Image
from ultralytics import YOLO

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
os.environ['LD_PRELOAD'] = '/opt/rocm/lib/libamdhip64.so'

MAX_PLATE_LEN = 7
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ _"  # Added '_' as blank token for variable length
NUM_CLASSES = len(ALPHABET)

OCR_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'best_improved_ocr_v5.pth')
YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'runs', 'detect', 'oman_plate_detector_yolov8s', 'weights', 'best.pt')


def preprocess_plate_image(image):
    """
    Preprocess plate image for better OCR results.
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast.
    """
    if len(image.shape) == 3:
        # Convert to grayscale for CLAHE
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Convert back to BGR for consistency
    if len(image.shape) == 3:
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    return enhanced


class EfficientNetOCR(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.classifiers = nn.ModuleList([
            nn.Linear(in_features, NUM_CLASSES) for _ in range(MAX_PLATE_LEN)
        ])
    
    def forward(self, x):
        features = self.backbone(x)
        outputs = [clf(features) for clf in self.classifiers]
        return torch.stack(outputs, dim=1)


class ANPR:
    def __init__(self, ocr_model_path=None, yolo_model_path=None, device=None):
        self.device = device or ('cpu')
        
        self.ocr_model_path = ocr_model_path or OCR_MODEL_PATH
        self.yolo_model_path = yolo_model_path or YOLO_MODEL_PATH
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print("Loading YOLO detector...")
        self.yolo = YOLO(self.yolo_model_path)
        print("Loading OCR model...")
        self.ocr = EfficientNetOCR()
        checkpoint = torch.load(self.ocr_model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.ocr.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.ocr.load_state_dict(checkpoint)
        self.ocr = self.ocr.to(self.device)
        self.ocr.eval()
        print("Models loaded successfully!")
    
    def detect_plate(self, image_path):
        """Detect plate using YOLO"""
        results = self.yolo(image_path, verbose=False)
        boxes = results[0].boxes
        
        if len(boxes) == 0:
            return None
        
        # Get first detection
        box = boxes[0]
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        conf = float(box.conf[0])
        
        return {
            'bbox': [x1, y1, x2, y2],
            'confidence': conf
        }
    
    def decode_predictions(self, preds):
        """Decode predictions, stopping at first blank token and removing trailing blanks"""
        result = ""
        blank_idx = ALPHABET.find('_')
        
        for idx in preds:
            idx = int(idx)
            if idx == blank_idx:
                break
            if idx < len(ALPHABET):
                result += ALPHABET[idx]
        
        result = result.strip()
        
        # Oman plate format:
        # OGTRC: 1-5 digits + 1-2 suffix letters (e.g., "9466B", "7347BR", "3566DD")
        # Regular: digits + suffix (letters at bottom)
        
        # Remove spaces between digits and suffix (model hallucination)
        result = result.replace(' ', '')
        
        # If pure digits (1-5), it's valid OGTRC - keep as-is
        if result.isdigit() and 1 <= len(result) <= 5:
            return result
        
        # If digits followed by letters - keep digits + max 2 letters suffix
        if len(result) >= 2:
            digit_count = 0
            for c in result:
                if c.isdigit():
                    digit_count += 1
                else:
                    break
            
            if 1 <= digit_count <= 5:
                suffix = result[digit_count:]
                # Keep max 2 letters suffix (Oman max suffix)
                if len(suffix) > 2:
                    suffix = suffix[:2]
                return result[:digit_count] + suffix
        
        return result
    
    def recognize_plate(self, image_path):
        """Full ANPR: Detect and recognize plate"""
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect plate
        detection = self.detect_plate(image_path)
        
        if detection is None:
            # Fallback: try OCR directly on the image (for already cropped plates)
            crop = img
        else:
            # Crop plate from detection with 15% padding to avoid cutting edges
            x1, y1, x2, y2 = detection['bbox']
            h, w = img.shape[:2]
            pad_x = int((x2 - x1) * 0.15)
            pad_y = int((y2 - y1) * 0.15)
            x1_p = max(0, x1 - pad_x)
            y1_p = max(0, y1 - pad_y)
            x2_p = min(w, x2 + pad_x)
            y2_p = min(h, y2 + pad_y)
            crop = img[y1_p:y2_p, x1_p:x2_p]
        
        if crop.size == 0:
            return {
                'plate_text': None,
                'detected': True,
                'message': 'Could not crop plate'
            }
        
        # OCR - apply preprocessing for better results
        plate_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        plate_enhanced = preprocess_plate_image(plate_rgb)
        
        # Convert enhanced image to PIL
        if len(plate_enhanced.shape) == 2:
            plate_pil = Image.fromarray(plate_enhanced, mode='L').convert('RGB')
        else:
            plate_pil = Image.fromarray(plate_enhanced)
        
        tensor = self.transform(plate_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            preds = torch.argmax(self.ocr(tensor), dim=2)
        
        text = self.decode_predictions(preds[0])
        
        return {
            'plate_text': text,
            'detected': True,
            'confidence': detection['confidence'] if detection else 1.0,
            'bbox': detection['bbox'] if detection else [0, 0, img.shape[1], img.shape[0]]
        }
    
    def predict(self, image_path):
        """Alias for recognize_plate - for compatibility with web_app.py"""
        result = self.recognize_plate(image_path)
        return result['plate_text'] if result['detected'] else "No plate detected"


def recognize_plate(image_path):
    anpr = ANPR()
    return anpr.recognize_plate(image_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python anpr_yolo_app.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print(f"Processing: {image_path}")
    anpr = ANPR()
    result = anpr.recognize_plate(image_path)
    print(f"Result: {result}")
