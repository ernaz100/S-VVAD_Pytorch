import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from src.config import Config
from src.models.resnet_vad import ResNetVAD
from src.data.dataset import create_dataloaders
from src.data.realvad_dataset import create_realvad_video_dataloaders

def train_resnet_vad(train_loader, val_loader, model, optimizer, criterion, device, num_epochs, output_dir):
    """
    Train the ResNet VAD model with GPU optimizations.
    
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
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Gradient accumulation steps (effective batch size = batch_size * gradient_accumulation_steps)
    gradient_accumulation_steps = 4
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Reset optimizer gradients
        optimizer.zero_grad()
        
        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)")):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Mixed precision training
            with autocast():
                # Forward pass
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
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
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                # Forward pass with mixed precision
                with autocast():
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
    
    # Load RealVAD dataset using the original video
    print("Loading RealVAD dataset from video...")
    realvad_dir = os.path.join(Config.DATA_ROOT, 'videos', 'RealVAD')
    
    # Create dataloaders using RealVAD video dataset with optimized settings
    train_loader, val_loader = create_realvad_video_dataloaders(
        root_dir=realvad_dir,
        batch_size=Config.BATCH_SIZE,
        input_size=Config.RESNET_INPUT_SIZE,
        multi_scale=Config.USE_MULTI_SCALE,
        scale_range=Config.MULTI_SCALE_RANGE,
        panelist_ids=None,  # Use all panelists
        validation_split=Config.VALIDATION_SPLIT,
        num_workers=8,  # Increased number of workers
        pin_memory=True  # Enable pin_memory for faster GPU transfer
    )
    
    print(f"Dataset loaded: {len(train_loader.dataset)} training samples, {len(val_loader.dataset)} validation samples")
    
    # Initialize model
    model = ResNetVAD(num_classes=2, pretrained=True)
    
    # Check if latest checkpoint exists and load it
    latest_checkpoint_path = os.path.join(Config.OUTPUT_DIR, 'latest_resnet_vad.pth')
    if os.path.exists(latest_checkpoint_path):
        print(f"Loading checkpoint from {latest_checkpoint_path}")
        model.load_state_dict(torch.load(latest_checkpoint_path))
    
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