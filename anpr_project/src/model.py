"""
CRNN Model for License Plate Recognition
Convolutional Recurrent Neural Network for sequence labeling
"""

import torch
import torch.nn as nn

class CRNN(nn.Module):
    """
    CRNN model architecture:
    - CNN (ResNet-style) for feature extraction
    - LSTM for sequence modeling
    - Linear for character prediction
    """
    def __init__(self, img_height=32, img_width=128, num_classes=38):
        super(CRNN, self).__init__()
        
        self.img_height = img_height
        self.num_classes = num_classes  # 37 letters + blank
        
        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            # Conv1: 3 -> 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # H/2, W/2
            
            # Conv2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # H/4, W/4
            
            # Conv3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # H/8, W/4
            
            # Conv4: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # H/16, W/4
            
            # Conv5: 256 -> 512
            nn.Conv2d(256, 512, kernel_size=2, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(512, 256, bidirectional=True, batch_first=True, num_layers=2)
        
        # Output layer
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # CNN
        conv = self.cnn(x)  # [B, 512, H', W']
        
        # Reshape for LSTM: [B, W, C]
        b, c, h, w = conv.size()
        conv = conv.squeeze(2)  # Remove height dimension
        conv = conv.permute(0, 2, 1)  # [B, W, C]
        
        # LSTM
        lstm_out, _ = self.lstm(conv)  # [B, W, 512]
        
        # Output
        output = self.fc(lstm_out)  # [B, W, num_classes]
        
        return output

class CTCLoss(nn.Module):
    """CTC Loss wrapper"""
    def __init__(self):
        super(CTCLoss, self).__init__()
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    
    def forward(self, predictions, targets, input_lengths, target_lengths):
        # predictions should be [T, N, C] - already log_softmax applied
        loss = self.ctc_loss(predictions, targets, input_lengths, target_lengths)
        return loss

def test_model():
    """Test the model with dummy input"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = CRNN(img_height=32, img_width=128, num_classes=38)
    model = model.to(device)
    
    # Dummy input
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 128).to(device)
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: [{batch_size}, 38, 38] (approx)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

if __name__ == "__main__":
    test_model()
