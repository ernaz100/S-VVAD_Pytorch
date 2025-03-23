import os
import torch

class Config:
    # Paths
    DATA_ROOT = "data"
    OUTPUT_DIR = "output"
    
    # Dynamic Image parameters
    DI_FRAMES = 10  # Number of frames to create one dynamic image
    
    # Model parameters
    RESNET_INPUT_SIZE = 256
    USE_MULTI_SCALE = True  # Whether to use multi-scale inputs as described in the paper
    MULTI_SCALE_RANGE = (256, 320)  # Range for multi-scale inputs
    
    # Training parameters
    BATCH_SIZE = 128
    LEARNING_RATE = 5e-6
    NUM_EPOCHS = 50
    VALIDATION_SPLIT = 0.1
    
    # FCN (Segmentation) parameters
    FCN_INPUT_SIZE = 512
    
    # Class labels
    CLASSES = {
        'speaking': 1,
        'not_speaking': 0,
        'background': 2
    }
    
    # Device configuration
    # Determine the best available device
    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    DEVICE = get_device.__func__() 