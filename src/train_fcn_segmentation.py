import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import cv2

from src.config import Config
from src.models.resnet_vad import ResNetVAD
from src.models.fcn_segmentation import FCN, create_fcn_from_cams
from src.data.dataset import FCNDataset
from src.data.realvad_dataset import RealVADDataset
from src.utils.dynamic_images import generate_dynamic_images_from_video

def generate_cams_with_resnet(dynamic_images, resnet_model, device):
    """
    Generate Class Activation Maps (CAMs) using the trained ResNet VAD model.
    
    Args:
        dynamic_images: List of dynamic images
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
    
    with torch.no_grad():
        for di in tqdm(dynamic_images, desc="Generating CAMs"):
            # Convert to tensor and normalize
            img = torch.from_numpy(di.transpose(2, 0, 1)).float() / 255.0
            img = img.to(device)
            
            # Normalize with ImageNet mean and std
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
            img = (img - mean) / std
            
            # Add batch dimension
            img = img.unsqueeze(0)
            
            # Generate CAMs
            speaking_cam, not_speaking_cam, logits = resnet_model.get_class_activation_maps(img)
            
            # Get prediction
            pred = torch.argmax(logits, dim=1).item()
            
            speaking_cams.append(speaking_cam)
            not_speaking_cams.append(not_speaking_cam)
            predictions.append(pred)
    
    return speaking_cams, not_speaking_cams, predictions

def train_fcn_segmentation(train_loader, val_loader, model, optimizer, criterion, device, num_epochs, output_dir):
    """
    Train the FCN segmentation model.
    
    Args:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        model: FCN segmentation model
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use (cpu or cuda)
        num_epochs: Number of epochs to train for
        output_dir: Directory to save model checkpoints
        
    Returns:
        trained_model: Trained FCN segmentation model
    """
    # Create tensorboard writer
    writer = SummaryWriter(os.path.join(output_dir, 'logs_fcn'))
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Keep track of best validation loss
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)"):
            inputs, masks = inputs.to(device), masks.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
        
        # Calculate training metrics
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)"):
                inputs, masks = inputs.to(device), masks.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, masks)
                
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
        resnet_model.load_state_dict(torch.load(os.path.join(Config.OUTPUT_DIR, 'best_resnet_vad.pth'), map_location=device))
        print("Successfully loaded ResNet VAD model.")
    except:
        print("Failed to load ResNet VAD model. Make sure to train it first using train_resnet_vad.py")
        return
    
    resnet_model = resnet_model.to(device)
    
    # Load RealVAD dataset
    print("Loading RealVAD dynamic images...")
    realvad_dir = os.path.join(Config.DATA_ROOT, 'videos', 'RealVAD')
    
    # Load dynamic images and labels
    dynamic_images, labels = load_realvad_dynamic_images(realvad_dir)
    print(f"Loaded {len(dynamic_images)} dynamic images.")
    
    # Generate CAMs using the trained ResNet VAD model
    print("Generating CAMs for FCN training...")
    speaking_cams, not_speaking_cams, predictions = generate_cams_with_resnet(
        dynamic_images, resnet_model, device
    )
    
    # Create FCN training data from CAMs
    print("Creating FCN training data...")
    X, y = create_fcn_from_cams(dynamic_images, speaking_cams, not_speaking_cams, device)
    
    # Create FCN dataset
    dataset = FCNDataset(X, y)
    
    # Split into train and validation sets
    dataset_size = len(dataset)
    val_size = int(dataset_size * Config.VALIDATION_SPLIT)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    print(f"Dataset prepared: {len(train_loader.dataset)} training samples, {len(val_loader.dataset)} validation samples")
    
    # Initialize FCN model
    print("Initializing FCN model...")
    model = FCN(num_classes=3)  # 3 classes: speaking, not-speaking, background
    model = model.to(device)
    
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
        Config.OUTPUT_DIR
    )
    
    print("FCN Training completed!")

if __name__ == "__main__":
    main() 