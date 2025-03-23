import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class FCN(nn.Module):
    """
    Fully Convolutional Network for Visual Voice Activity Detection Segmentation.
    As described in the S-VVAD paper, this network segments the input image into 
    speaking, not-speaking, and background regions.
    """
    def __init__(self, num_classes=3):
        super(FCN, self).__init__()
        
        # Load pre-trained VGG16 as the backbone
        vgg16 = models.vgg16(pretrained=True)
        
        # Extract feature encoder layers from VGG16
        self.features = vgg16.features
        
        # FCN decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
        )
        
        # Final classification layer
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder (VGG16 features)
        x = self.features(x)
        
        # Decoder
        x = self.decoder(x)
        
        # Classifier
        x = self.classifier(x)
        
        return x
    
    def predict_mask(self, x):
        """
        Predict segmentation mask from the input image.
        
        Args:
            x: Input tensor
            
        Returns:
            mask: Segmentation mask with class indices
        """
        with torch.no_grad():
            out = self.forward(x)
            mask = torch.argmax(out, dim=1)
        return mask

def create_fcn_from_cams(dynamic_images, speaking_cams, not_speaking_cams, device):
    """
    Create training data for FCN using dynamic images and CAMs.
    
    Args:
        dynamic_images: List of dynamic images
        speaking_cams: List of speaking CAMs
        not_speaking_cams: List of not-speaking CAMs
        device: Device to use (cpu or cuda)
        
    Returns:
        X: Input tensor for FCN training
        y: Target labels for FCN training
    """
    X = []
    y = []
    
    for i in range(len(dynamic_images)):
        # Convert dynamic image to tensor
        di = torch.from_numpy(dynamic_images[i].transpose(2, 0, 1)).float() / 255.0
        di = di.to(device)
        
        # Get CAMs
        speaking_cam = speaking_cams[i].squeeze()
        not_speaking_cam = not_speaking_cams[i].squeeze()
        
        # Create mask: 0 = not speaking, 1 = speaking, 2 = background
        mask = torch.ones_like(speaking_cam) * 2  # Default: background
        
        # Set speaking and not-speaking regions
        speaking_threshold = 0.5
        not_speaking_threshold = 0.5
        
        # Speaking has priority over not-speaking when both are above threshold
        mask[not_speaking_cam > not_speaking_threshold] = 0  # Not speaking
        mask[speaking_cam > speaking_threshold] = 1  # Speaking
        
        X.append(di)
        y.append(mask)
    
    # Stack tensors
    X = torch.stack(X)
    y = torch.stack(y).long()
    
    return X, y 