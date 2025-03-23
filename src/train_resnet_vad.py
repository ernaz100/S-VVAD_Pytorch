import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.config import Config
from src.models.resnet_vad import ResNetVAD
from src.data.dataset import create_dataloaders
from src.data.realvad_dataset import create_realvad_video_dataloaders

def train_resnet_vad(train_loader, val_loader, model, optimizer, criterion, device, num_epochs, output_dir):
    """
    Train the ResNet VAD model.
    
    Args:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        model: ResNet VAD model
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use (cpu or cuda)
        num_epochs: Number of epochs to train for
        output_dir: Directory to save model checkpoints
        
    Returns:
        trained_model: Trained ResNet VAD model
    """
    # Create tensorboard writer
    writer = SummaryWriter(os.path.join(output_dir, 'logs'))
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Keep track of best validation accuracy
    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Calculate training metrics
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Log metrics to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        # Save model if validation accuracy improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_resnet_vad.pth'))
            print(f"Saved model with val_acc: {val_acc:.4f}")
        
        # Save latest model
        torch.save(model.state_dict(), os.path.join(output_dir, 'latest_resnet_vad.pth'))
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_resnet_vad.pth')))
    
    # Close tensorboard writer
    writer.close()
    
    return model

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
    
    # Load RealVAD dataset using the original video
    print("Loading RealVAD dataset from video...")
    realvad_dir = os.path.join(Config.DATA_ROOT, 'videos', 'RealVAD')
    
    # Create dataloaders using RealVAD video dataset
    train_loader, val_loader = create_realvad_video_dataloaders(
        root_dir=realvad_dir,
        batch_size=Config.BATCH_SIZE,
        input_size=Config.RESNET_INPUT_SIZE,
        multi_scale=Config.USE_MULTI_SCALE,
        scale_range=Config.MULTI_SCALE_RANGE,
        panelist_ids=None,  # Use all panelists
        validation_split=Config.VALIDATION_SPLIT
    )
    
    print(f"Dataset loaded: {len(train_loader.dataset)} training samples, {len(val_loader.dataset)} validation samples")
    
    # Initialize model
    model = ResNetVAD(num_classes=2, pretrained=True)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Train model
    print("Starting training...")
    trained_model = train_resnet_vad(
        train_loader,
        val_loader,
        model,
        optimizer,
        criterion,
        device,
        Config.NUM_EPOCHS,
        Config.OUTPUT_DIR
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main() 