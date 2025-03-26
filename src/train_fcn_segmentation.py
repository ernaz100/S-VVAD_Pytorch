import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import cv2
import torch.nn.functional as F

from src.config import Config
from src.models.resnet_vad import ResNetVAD
from src.models.fcn_segmentation import FCN, create_fcn_from_cams
from src.data.dataset import FCNDataset
from src.data.realvad_dataset import RealVADVideoDataset
from src.utils.dynamic_images import generate_dynamic_images_from_video

def generate_cams_with_resnet(dynamic_images, resnet_model, device):
    """
    Generate Class Activation Maps (CAMs) using the trained ResNet VAD model.
    
    Args:
        dynamic_images: Batch of dynamic images (tensor of shape B, C, H, W)
        resnet_model: Trained ResNet VAD model
        device: Device to use (cpu or cuda)
        
    Returns:
        speaking_cams: List of speaking CAMs
        not_speaking_cams: List of not-speaking CAMs
        predictions: List of predicted speaking/not-speaking labels
    """
    resnet_model.eval()
    speaking_cams = []
    not_speaking_cams = []
    predictions = []
    
    # First, get predictions without gradients
    with torch.no_grad():
        inputs = dynamic_images.to(device, non_blocking=True)
        logits, _ = resnet_model(inputs)
        preds = torch.argmax(logits, dim=1)
        predictions = preds.cpu().numpy().tolist()
    
    # Compute CAMs one at a time with gradients enabled
    for i in range(inputs.size(0)):
        # Get single image and ensure it requires gradients
        img = inputs[i:i+1].detach().requires_grad_(True)
        
        # Enable gradients for model parameters
        for param in resnet_model.parameters():
            param.requires_grad = True
        
        # Compute CAMs for speaking class (class 1)
        speaking_cam, _ = resnet_model.compute_grad_cam(img, torch.tensor([1]).to(device))
        speaking_cams.append(speaking_cam.squeeze().cpu().numpy())
        
        # Compute CAMs for not-speaking class (class 0)
        not_speaking_cam, _ = resnet_model.compute_grad_cam(img, torch.tensor([0]).to(device))
        not_speaking_cams.append(not_speaking_cam.squeeze().cpu().numpy())
        
        # Clear gradients after each image
        resnet_model.zero_grad()
    
    return speaking_cams, not_speaking_cams, predictions

def train_fcn_segmentation(train_loader, val_loader, model, optimizer, criterion, device, num_epochs, output_dir, resnet_model):
    """
    Train the FCN segmentation model with weak supervision from ResNet VAD.
    As described in the S-VVAD paper, we use the ResNet VAD predictions as weak labels
    for training the segmentation model.
    """
    # Create tensorboard writer
    writer = SummaryWriter(os.path.join(output_dir, 'logs_fcn'))
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Keep track of best validation loss
    best_val_loss = float('inf')
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Gradient accumulation steps (effective batch size = batch_size * gradient_accumulation_steps)
    gradient_accumulation_steps = 4
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        # Reset optimizer gradients
        optimizer.zero_grad()
        
        for batch_idx, (inputs, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)")):
            inputs = inputs.to(device, non_blocking=True)
            
            # Generate CAMs using ResNet VAD
            with torch.no_grad():
                speaking_cams, not_speaking_cams, _ = generate_cams_with_resnet(inputs, resnet_model, device)
            
            # Convert CAMs to weak labels (0: background, 1: speaking, 2: not-speaking)
            weak_labels = torch.zeros((inputs.size(0), inputs.size(2), inputs.size(3)), dtype=torch.long, device=device)
            for i in range(inputs.size(0)):
                # Convert numpy arrays to tensors
                speaking_cam = torch.from_numpy(speaking_cams[i]).to(device)
                not_speaking_cam = torch.from_numpy(not_speaking_cams[i]).to(device)
                
                # Normalize CAMs to [0, 1]
                speaking_cam = (speaking_cam - speaking_cam.min()) / (speaking_cam.max() - speaking_cam.min())
                not_speaking_cam = (not_speaking_cam - not_speaking_cam.min()) / (not_speaking_cam.max() - not_speaking_cam.min())
                
                # Create weak labels based on CAM thresholds
                weak_labels[i] = torch.where(speaking_cam > 0.5, torch.tensor(1, device=device),
                                           torch.where(not_speaking_cam > 0.5, torch.tensor(2, device=device),
                                                     torch.tensor(0, device=device)))
            
            # Mixed precision training
            with autocast():
                # Forward pass through FCN
                outputs = model(inputs)
                
                # Apply softmax to get probabilities
                probs = F.softmax(outputs, dim=1)
                
                # Get speaking and not-speaking probabilities
                speaking_probs = probs[:, 1]  # Class 1 is speaking
                not_speaking_probs = probs[:, 0]  # Class 0 is not-speaking
                
                # Compute segmentation loss using weak labels
                loss = criterion(outputs, weak_labels)
                
                # Add regularization to encourage smooth segmentation
                smoothness_loss = torch.mean(torch.abs(speaking_probs[:, :, :, 1:] - speaking_probs[:, :, :, :-1])) + \
                                torch.mean(torch.abs(speaking_probs[:, :, 1:, :] - speaking_probs[:, :, :-1, :])) + \
                                torch.mean(torch.abs(not_speaking_probs[:, :, :, 1:] - not_speaking_probs[:, :, :, :-1])) + \
                                torch.mean(torch.abs(not_speaking_probs[:, :, 1:, :] - not_speaking_probs[:, :, :-1, :]))
                
                # Add CAM consistency loss
                cam_consistency_loss = F.mse_loss(speaking_probs, speaking_cams) + F.mse_loss(not_speaking_probs, not_speaking_cams)
                
                # Combine losses
                loss = loss + 0.1 * smoothness_loss + 0.1 * cam_consistency_loss
                
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Update weights if we've accumulated enough gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0) * gradient_accumulation_steps
        
        # Calculate training metrics
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)"):
                inputs = inputs.to(device, non_blocking=True)
                
                # Generate CAMs for validation
                speaking_cams, not_speaking_cams, _ = generate_cams_with_resnet(inputs, resnet_model, device)
                
                # Convert CAMs to weak labels
                weak_labels = torch.zeros((inputs.size(0), inputs.size(2), inputs.size(3)), dtype=torch.long, device=device)
                for i in range(inputs.size(0)):
                    # Convert numpy arrays to tensors
                    speaking_cam = torch.from_numpy(speaking_cams[i]).to(device)
                    not_speaking_cam = torch.from_numpy(not_speaking_cams[i]).to(device)
                    
                    # Normalize CAMs to [0, 1]
                    speaking_cam = (speaking_cam - speaking_cam.min()) / (speaking_cam.max() - speaking_cam.min())
                    not_speaking_cam = (not_speaking_cam - not_speaking_cam.min()) / (not_speaking_cam.max() - not_speaking_cam.min())
                    
                    # Create weak labels based on CAM thresholds
                    weak_labels[i] = torch.where(speaking_cam > 0.5, torch.tensor(1, device=device),
                                               torch.where(not_speaking_cam > 0.5, torch.tensor(2, device=device),
                                                         torch.tensor(0, device=device)))
                
                # Forward pass with mixed precision
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, weak_labels)
                
                # Statistics
                val_loss += loss.item() * inputs.size(0)
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader.dataset)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        # Log metrics to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        # Save model if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_fcn_segmentation.pth'))
            print(f"Saved model with val_loss: {val_loss:.4f}")
        
        # Save latest model
        torch.save(model.state_dict(), os.path.join(output_dir, 'latest_fcn_segmentation.pth'))
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_fcn_segmentation.pth')))
    
    # Close tensorboard writer
    writer.close()
    
    return model

def load_realvad_dynamic_images(realvad_dir):
    """
    Load dynamic images from the RealVAD dataset.
    
    Args:
        realvad_dir: Directory containing the RealVAD dataset
        
    Returns:
        dynamic_images: List of loaded dynamic images
        labels: List of corresponding labels (0: not-speaking, 1: speaking)
    """
    # Load annotations
    annotations_file = os.path.join(realvad_dir, 'dynamic_images', 'annotations.txt')
    image_dir = os.path.join(realvad_dir, 'dynamic_images', 'img')
    
    annotations = []
    with open(annotations_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                image_name, speaking_status, _ = line.split(',')
                # Convert speaking status to int (0: not-speaking, 1: speaking)
                speaking_status = int(speaking_status)
                annotations.append((image_name, speaking_status))
    
    # Load images
    dynamic_images = []
    labels = []
    
    for image_name, label in tqdm(annotations, desc="Loading dynamic images"):
        image_path = os.path.join(image_dir, image_name)
        
        # Load image
        image = cv2.imread(image_path)
        if image is not None:
            # Convert from BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            dynamic_images.append(image)
            labels.append(label)
    
    return dynamic_images, labels

def main():
    # Set device
    if Config.DEVICE == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        # Enable cuDNN benchmarking for faster training
        torch.backends.cudnn.benchmark = True
        # Enable cuDNN deterministic mode for reproducibility
        torch.backends.cudnn.deterministic = True
    elif Config.DEVICE == 'mps' and hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Load the trained ResNet VAD model
    print("Loading trained ResNet VAD model...")
    resnet_model = ResNetVAD(num_classes=2, pretrained=False)
    try:
        # Load the state dict
        state_dict = torch.load(os.path.join(Config.OUTPUT_DIR, 'best_resnet_vad.pth'), map_location=device)
        
        # Remove '_orig_mod.' prefix from state dict keys if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                new_state_dict[k[10:]] = v  # Remove '_orig_mod.' prefix
            else:
                new_state_dict[k] = v
        
        # Load the modified state dict
        resnet_model.load_state_dict(new_state_dict)
        print("Successfully loaded ResNet VAD model.")
    except Exception as e:
        print(f"Failed to load ResNet VAD model: {str(e)}")
        print("Make sure to train it first using train_resnet_vad.py")
        return
    
    resnet_model = resnet_model.to(device)
    resnet_model.eval()  # Set to evaluation mode
    
    # Load RealVAD dataset
    print("Loading RealVAD dataset...")
    realvad_dir = os.path.join(Config.DATA_ROOT, 'videos', 'RealVAD')
    
    # Create dataset with multi-scale support
    dataset = RealVADVideoDataset(
        root_dir=realvad_dir,
        input_size=256,
        multi_scale=True,
        scale_range=(256, 320)
    )
    
    # Split into train and validation sets
    dataset_size = len(dataset)
    val_size = int(dataset_size * Config.VALIDATION_SPLIT)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders with optimized settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    print(f"Dataset prepared: {len(train_loader.dataset)} training samples, {len(val_loader.dataset)} validation samples")
    
    # Initialize FCN model
    print("Initializing FCN model...")
    model = FCN(num_classes=3)  # 3 classes: speaking, not-speaking, background
    model = model.to(device)
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, 'enable_checkpointing'):
        model.enable_checkpointing()
    
    # Compile model for PyTorch 2.0+ optimizations if available
    if hasattr(torch, 'compile'):
        print("Compiling model for PyTorch 2.0+ optimizations...")
        model = torch.compile(model)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Train model
    print("Starting FCN training...")
    trained_model = train_fcn_segmentation(
        train_loader,
        val_loader,
        model,
        optimizer,
        criterion,
        device,
        Config.NUM_EPOCHS,
        Config.OUTPUT_DIR,
        resnet_model  # Pass ResNet model for CAM generation
    )
    
    print("FCN Training completed!")

if __name__ == "__main__":
    main() 