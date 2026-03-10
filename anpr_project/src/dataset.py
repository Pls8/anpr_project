"""
Dataset class for License Plate OCR
"""

import os
import json
import re
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class LicensePlateDataset(Dataset):
    """Dataset for Omani License Plates"""
    
    def __init__(self, img_dir, metadata_path, alphabet, img_height=32, img_width=128, transform=None):
        """
        Args:
            img_dir: Directory with images
            metadata_path: Path to metadata.jsonl
            alphabet: Character set
            img_height: Target image height
            img_width: Target image width
            transform: Optional transforms
        """
        self.img_dir = img_dir
        self.img_height = img_height
        self.img_width = img_width
        self.alphabet = alphabet
        self.char_to_idx = {char: idx for idx, char in enumerate(alphabet)}
        
        # Load labels
        self.labels = self._load_labels(metadata_path)
        self.image_files = list(self.labels.keys())
        
        # Default transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_height, img_width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
    
    def _load_labels(self, metadata_path):
        """Load labels from metadata.jsonl"""
        labels = {}
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                file_name = data['file_name']
                text = data['text']
                
                # Extract plate number
                number_match = re.search(r'<s_lp_number>\s*(.*?)\s*</s_lp_number>', text)
                if number_match:
                    plate = number_match.group(1).strip().upper()
                    # Only keep valid plates (no unknown)
                    if '<unk>' not in plate.lower() and plate:
                        # Filter to only known characters
                        plate = self._filter_chars(plate)
                        if plate:
                            labels[file_name] = plate
        
        return labels
    
    def _filter_chars(self, text):
        """Filter text to only include known characters"""
        filtered = ""
        for char in text:
            if char in self.alphabet:
                filtered += char
        return filtered
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.labels[img_name]
        
        # Convert label to tensor
        label_tensor = self._text_to_tensor(label)
        
        return {
            'image': image,
            'label': label,
            'label_tensor': label_tensor,
            'length': len(label)
        }
    
    def _text_to_tensor(self, text):
        """Convert text to tensor of indices"""
        tensor = torch.zeros(len(text), dtype=torch.long)
        for i, char in enumerate(text):
            tensor[i] = self.char_to_idx.get(char, 0)
        return tensor

def get_alphabet():
    """Get alphabet for Omani plates"""
    # Omani plates: Numbers (0-9), Letters (A-Z), space
    return "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ "

def test_dataset():
    """Test the dataset class"""
    TRAIN_DIR = "/mnt/c/Users/Masad/Documents/aiPorject/Test3OC/oman-licenceplates/dataset/train"
    
    alphabet = get_alphabet()
    dataset = LicensePlateDataset(
        img_dir=TRAIN_DIR,
        metadata_path=os.path.join(TRAIN_DIR, "metadata.jsonl"),
        alphabet=alphabet
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"\nSample:")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Label: {sample['label']}")
    print(f"  Label tensor: {sample['label_tensor']}")
    print(f"  Length: {sample['length']}")

if __name__ == "__main__":
    test_dataset()
