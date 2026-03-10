import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import random

TRAIN_IMG_DIR = "/mnt/c/Users/Masad/Documents/aiPorject/Test3OC/oman-licenceplates/dataset/train"
VAL_IMG_DIR = "/mnt/c/Users/Masad/Documents/aiPorject/Test3OC/oman-licenceplates/dataset/validation"

def load_metadata(jsonl_path):
    """Load labels from metadata.jsonl file"""
    labels = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            file_name = data['file_name']
            text = data['text']
            
            # Extract plate number from text
            # Format: <s_lp_type> Type </s_lp_type><s_lp_number> NUMBER </s_lp_number>
            import re
            number_match = re.search(r'<s_lp_number>\s*(.*?)\s*</s_lp_number>', text)
            if number_match:
                plate_number = number_match.group(1).strip()
                # Skip unknown plates
                if '<unk>' not in plate_number.lower():
                    labels[file_name] = plate_number
    
    return labels

def prepare_dataset():
    """Prepare training and validation datasets"""
    train_labels = load_metadata(os.path.join(TRAIN_IMG_DIR, "metadata.jsonl"))
    val_labels = load_metadata(os.path.join(VAL_IMG_DIR, "metadata.jsonl"))
    
    print(f"Train samples (with labels): {len(train_labels)}")
    print(f"Validation samples (with labels): {len(val_labels)}")
    
    return train_labels, val_labels

def get_alphabet():
    """Get the character set for Omani plates"""
    # Omani plates contain: numbers, letters, and potentially Arabic
    # Based on data: 0-9, A-Z, space
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ "
    return alphabet

def text_to_tensor(text, alphabet, max_len=10):
    """Convert text to tensor"""
    tensor = torch.zeros(max_len, len(alphabet))
    for i, char in enumerate(text[:max_len]):
        if char in alphabet:
            tensor[i][alphabet.index(char)] = 1
    return tensor

def tensor_to_text(tensor, alphabet):
    """Convert tensor to text"""
    indices = torch.argmax(tensor, dim=1)
    text = ""
    for idx in indices:
        if idx < len(alphabet):
            text += alphabet[idx]
    return text.strip()

if __name__ == "__main__":
    print("=" * 50)
    print("Omani ANPR - Data Preparation")
    print("=" * 50)
    
    train_labels, val_labels = prepare_dataset()
    
    print("\nSample labels from training data:")
    for i, (fname, label) in enumerate(list(train_labels.items())[:5]):
        print(f"  {fname}: {label}")
    
    print("\nSample labels from validation data:")
    for i, (fname, label) in enumerate(list(val_labels.items())[:5]):
        print(f"  {fname}: {label}")
    
    alphabet = get_alphabet()
    print(f"\nAlphabet ({len(alphabet)} chars): {alphabet}")
