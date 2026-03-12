"""
OCR Training v6 - EfficientNet-B1 (larger than B0)
Variable Length Support with Blank Token
"""

import os
import sys
import json
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from tqdm import tqdm
import shutil

MAX_PLATE_LEN = 7
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ _"  # Added '_' as blank token
NUM_CLASSES = len(ALPHABET)
BLANK_IDX = ALPHABET.index('_')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
TEST3OC_DIR = os.path.dirname(PROJECT_DIR)
TRAIN_DIR = os.path.join(TEST3OC_DIR, "oman-licenceplates", "dataset", "train")
VAL_DIR = os.path.join(TEST3OC_DIR, "oman-licenceplates", "dataset", "validation")
CUSTOM_DIR = os.path.join(PROJECT_DIR, "test_images")
MODEL_SAVE_PATH = os.path.join(TEST3OC_DIR, "best_improved_ocr_v6_b1.pth")

print(f"=== OCR Training v6 - EfficientNet-B1 ===")
print(f"Alphabet: {ALPHABET}")
print(f"Number of classes: {NUM_CLASSES}")
print(f"Blank token index: {BLANK_IDX}")
print(f"Max plate length: {MAX_PLATE_LEN}")
print(f"Image size: 240x240 (B1)")


class CombinedDataset(Dataset):
    def __init__(self, data_dir, custom_dir=None, transform=None):
        self.data_dir = data_dir
        self.custom_dir = custom_dir
        self.transform = transform
        self.labels = {}
        
        metadata_path = os.path.join(data_dir, 'metadata.jsonl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    fname = data['file_name']
                    text = data['text']
                    number_match = re.search(r'<s_lp_number>\s*(.*?)\s*</s_lp_number>', text)
                    if number_match:
                        plate = number_match.group(1).strip().upper()
                        plate = ''.join([c for c in plate if c in ALPHABET.replace('_', '')])
                        if plate and '<unk>' not in plate and len(plate) <= MAX_PLATE_LEN:
                            self.labels[fname] = plate
        
        if custom_dir and os.path.exists(custom_dir):
            custom_labels = {}
            for fname in os.listdir(custom_dir):
                if fname.startswith('crop-') and fname.endswith('.png'):
                    plate = "81080 W"
                    custom_labels[fname] = plate
            
            self.custom_files = list(custom_labels.keys())
            self.custom_labels = custom_labels
            print(f"Added {len(self.custom_files)} custom images")
        else:
            self.custom_files = []
        
        self.image_files = list(self.labels.keys())
        print(f"Loaded {len(self.image_files)} dataset images")
    
    def __len__(self):
        return len(self.image_files) + len(self.custom_files)
    
    def __getitem__(self, idx):
        if idx < len(self.image_files):
            fname = self.image_files[idx]
            img_path = os.path.join(self.data_dir, fname)
            plate_text = self.labels[fname]
        else:
            cidx = idx - len(self.image_files)
            fname = self.custom_files[cidx]
            img_path = os.path.join(self.custom_dir, fname)
            plate_text = self.custom_labels[fname]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        target = torch.full((MAX_PLATE_LEN,), BLANK_IDX, dtype=torch.long)
        for i, char in enumerate(plate_text):
            if i < MAX_PLATE_LEN:
                target[i] = ALPHABET.index(char)
        
        return image, target, plate_text


class ImprovedOCR(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, num_chars=MAX_PLATE_LEN):
        super(ImprovedOCR, self).__init__()
        
        # EfficientNet-B1 (larger than B0)
        self.backbone = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
        
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        self.classifiers = nn.ModuleList([
            nn.Linear(in_features, num_classes) for _ in range(num_chars)
        ])
        
        for clf in self.classifiers:
            nn.init.xavier_uniform_(clf.weight)
            nn.init.zeros_(clf.bias)
    
    def forward(self, x):
        features = self.backbone(x)
        outputs = [clf(features) for clf in self.classifiers]
        return torch.stack(outputs, dim=1)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, targets, _ in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs.view(-1, NUM_CLASSES), targets.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(outputs, dim=2)
        correct += (preds == targets).all(dim=1).sum().item()
        total += targets.size(0)
    
    return total_loss / len(dataloader), correct / total * 100


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, targets, _ in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs.view(-1, NUM_CLASSES), targets.view(-1))
            
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=2)
            correct += (preds == targets).all(dim=1).sum().item()
            total += targets.size(0)
    
    return total_loss / len(dataloader), correct / total * 100


def decode_prediction(preds):
    """Decode prediction, stopping at first blank token"""
    result = ""
    for idx in preds:
        idx = idx.item()
        if idx == BLANK_IDX:
            break
        if idx < len(ALPHABET):
            result += ALPHABET[idx]
    return result.strip()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Training transforms with augmentation - B1 uses 240x240
    train_transform = transforms.Compose([
        transforms.Resize((240, 240)),  # B1 uses 240x240
        transforms.RandomRotation(degrees=15),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((240, 240)),  # B1 uses 240x240
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print("\nLoading datasets...")
    train_dataset = CombinedDataset(TRAIN_DIR, CUSTOM_DIR, transform=train_transform)
    val_dataset = CombinedDataset(VAL_DIR, transform=val_transform)
    
    # B1 is larger, reduce batch size to avoid OOM
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False, num_workers=2)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    model = ImprovedOCR(num_classes=NUM_CLASSES, num_chars=MAX_PLATE_LEN)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    start_epoch = 0
    best_val_acc = 0
    
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"\nLoading existing model from {MODEL_SAVE_PATH}...")
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        checkpoint_path = MODEL_SAVE_PATH.replace('.pth', '_checkpoint.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            start_epoch = checkpoint.get('epoch', 0)
            best_val_acc = checkpoint.get('best_val_acc', 0)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"Resuming from epoch {start_epoch+1}, best_val_acc: {best_val_acc:.2f}%")
    
    print("\nStarting training...")
    for epoch in range(start_epoch, 50):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/50 - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> Saved best model (val acc: {val_acc:.2f}%)")
        
        checkpoint_path = MODEL_SAVE_PATH.replace('.pth', '_checkpoint.pth')
        torch.save({
            'epoch': epoch + 1,
            'best_val_acc': best_val_acc,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, checkpoint_path)
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
