import torch
import torch.nn as nn
from torchvision import models

class IMCFNArch(nn.Module):
    """
    IMCFN: Image-based Malware Classification using Fine-tuned CNN
    Architecture based on VGG16 with custom top layers.

    Paper: "IMCFN: Image-based malware classification using fine-tuned convolutional
            neural network architecture" (Vasan et al., 2020)
    """

    def __init__(self, num_classes=6, dropout_rate=0.5):
        super(IMCFNArch, self).__init__()

        # Load pre-trained VGG16 using modern PyTorch weights API
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # Extract features (convolutional layers)
        self.features = vgg16.features

        # Freeze blocks 1-4 (layers 0-23)
        # Unfreeze block 5 (layers 24-30) for fine-tuning
        for i, layer in enumerate(self.features):
            if i < 24:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True

        # =========================================================
        # THE FIX: We inherit the EXACT pre-trained VGG16 classifier 
        # This preserves the 4096-unit FC1 and FC2 layers and their weights!
        # =========================================================
        self.classifier = vgg16.classifier
        
        # We only replace the FINAL layer (index 6, the 1000-class output) 
        # with our custom Dropout and multi-class output layer
        in_features = self.classifier[6].in_features
        self.classifier[6] = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        
        self.flatten = nn.Flatten()

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (Tensor): Input image tensor of shape (batch, 3, 224, 224)
        """
        # VGG16 feature extraction
        x = self.features(x)  # Output: [batch, 512, 7, 7]

        # Flatten to 1D vector
        x = self.flatten(x)  # Output: [batch, 25088]

        # Pass through the pre-trained 4096-unit layers and our custom output
        x = self.classifier(x)
        
        return x