import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights
from torch.nn import functional as F
import numpy as np

class ResNetVAD(nn.Module):
    """
    ResNet50-based model for Visual Voice Activity Detection (VVAD).
    Fine-tuned from a pre-trained ImageNet model as described in the S-VVAD paper.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(ResNetVAD, self).__init__()
        
        # Load pre-trained ResNet50
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Remove the final fully connected layer
        modules = list(resnet.children())[:-2]
        
        # Blocks 1-2 (frozen as per paper)
        self.block1_2 = nn.Sequential(*modules[:5])
        for param in self.block1_2.parameters():
            param.requires_grad = False
            
        # Blocks 3-5 (fine-tuned as per paper)
        self.block3 = modules[5]
        self.block4 = modules[6]
        self.block5 = modules[7]
        
        # Global average pooling and classification layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        x = self.block1_2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        
        features = x  # Store features for CAM generation
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x, features
    
    def compute_grad_cam(self, x, class_idx=None):
        """
        Compute Grad-CAM for the given input and class index.
        If class_idx is None, use the predicted class.
        
        Args:
            x: Input tensor
            class_idx: Class index to compute Grad-CAM for
            
        Returns:
            cam: Grad-CAM heatmap
            logits: Model prediction logits
        """
        # Ensure input requires gradients
        if not x.requires_grad:
            x.requires_grad_(True)
        
        # Forward pass
        logits, features = self.forward(x)
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1)
        
        # One-hot encoding of the target class
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, class_idx.view(-1, 1), 1.0)
        
        # Zero gradients before backward pass
        self.zero_grad()
        
        # Compute gradients
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and feature maps
        gradients = self.get_activations_gradient()
        activations = features.detach()
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        
        # Weighted combination of feature maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        return cam, logits
    
    def get_activations_gradient(self):
        """
        Helper method to get gradients of the last convolutional layer.
        """
        return self.block5[-1].conv3.weight.grad
    
    def get_class_activation_maps(self, x):
        """
        Generate class activation maps for both speaking and not-speaking classes.
        
        Args:
            x: Input tensor
            
        Returns:
            speaking_cam: CAM for speaking class
            not_speaking_cam: CAM for not-speaking class
        """
        # Forward pass
        logits, features = self.forward(x)
        
        # Compute CAM for speaking class (class 1)
        speaking_cam, _ = self.compute_grad_cam(x, torch.tensor([1]).to(x.device))
        
        # Compute CAM for not-speaking class (class 0)
        not_speaking_cam, _ = self.compute_grad_cam(x, torch.tensor([0]).to(x.device))
        
        return speaking_cam, not_speaking_cam, logits 