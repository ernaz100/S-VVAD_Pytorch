import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import argparse

from src.config import Config
from src.models.resnet_vad import ResNetVAD
from src.models.fcn_segmentation import FCN
from src.utils.dynamic_images import generate_dynamic_images_from_video, preprocess_dynamic_image
from src.data.realvad_dataset import load_realvad_bbox_data, load_realvad_vad_data

def process_video_for_vad(video_path, resnet_model, fcn_model, device, output_dir=None, visualize=True):
    """
    Process a video for Visual Voice Activity Detection using the trained models.
    
    Args:
        video_path: Path to the video file
        resnet_model: Trained ResNet VAD model
        fcn_model: Trained FCN segmentation model
        device: Device to use (cpu or cuda)
        output_dir: Directory to save output video (if None, no video will be saved)
        visualize: Whether to visualize the results
        
    Returns:
        predictions: List of predictions (0 for not-speaking, 1 for speaking) for each dynamic image
        segmentations: List of segmentation masks
    """
    # Generate dynamic images from video
    print("Generating dynamic images from video...")
    dynamic_images = generate_dynamic_images_from_video(video_path, Config.DI_FRAMES)
    print(f"Generated {len(dynamic_images)} dynamic images.")
    
    # Set models to evaluation mode
    resnet_model.eval()
    fcn_model.eval()
    
    # Lists to store results
    predictions = []
    segmentations = []
    
    # Process dynamic images
    print("Processing dynamic images...")
    for di in tqdm(dynamic_images, desc="Processing dynamic images"):
        # Preprocess dynamic image for ResNet
        resnet_input = preprocess_dynamic_image(di, Config.RESNET_INPUT_SIZE).unsqueeze(0).to(device)
        
        # Get prediction and CAMs from ResNet
        with torch.no_grad():
            speaking_cam, not_speaking_cam, logits = resnet_model.get_class_activation_maps(resnet_input)
            prediction = torch.argmax(logits, dim=1).item()
        
        # Preprocess dynamic image for FCN
        fcn_input = torch.from_numpy(di.transpose(2, 0, 1)).float() / 255.0
        fcn_input = fcn_input.unsqueeze(0).to(device)
        
        # Get segmentation from FCN
        with torch.no_grad():
            segmentation = fcn_model.predict_mask(fcn_input)
            segmentation = segmentation.cpu().numpy()[0]
        
        predictions.append(prediction)
        segmentations.append(segmentation)
        
        # Visualize results
        if visualize:
            plt.figure(figsize=(15, 5))
            
            # Display dynamic image
            plt.subplot(131)
            plt.imshow(di)
            plt.title(f"Speaking: {prediction == 1}")
            plt.axis('off')
            
            # Display CAMs
            plt.subplot(132)
            plt.imshow(di)
            speaking_cam_np = speaking_cam.cpu().numpy()[0, 0]
            plt.imshow(speaking_cam_np, cmap='jet', alpha=0.5)
            plt.title("Speaking CAM")
            plt.axis('off')
            
            # Display segmentation
            plt.subplot(133)
            seg_display = np.zeros_like(di)
            seg_display[segmentation == 1] = [0, 255, 0]  # Green for speaking
            seg_display[segmentation == 0] = [255, 0, 0]  # Red for not-speaking
            plt.imshow(di)
            plt.imshow(seg_display, alpha=0.5)
            plt.title("Segmentation")
            plt.axis('off')
            
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)
            plt.close()
    
    # Save video if output_dir is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output video
        output_path = os.path.join(output_dir, os.path.basename(video_path).split('.')[0] + '_vad.mp4')
        create_output_video(video_path, output_path, predictions, segmentations, dynamic_images)
    
    return predictions, segmentations

def process_realvad_video(realvad_dir, resnet_model, fcn_model, device, panelist_id=1, output_dir=None, visualize=True):
    """
    Process the RealVAD video for a specific panelist.
    
    Args:
        realvad_dir: Directory containing the RealVAD dataset
        resnet_model: Trained ResNet VAD model
        fcn_model: Trained FCN segmentation model
        device: Device to use (cpu or cuda)
        panelist_id: ID of the panelist to process (1-9)
        output_dir: Directory to save output video (if None, no video will be saved)
        visualize: Whether to visualize the results
        
    Returns:
        predictions: List of predictions (0 for not-speaking, 1 for speaking)
        ground_truth: List of ground truth labels
        accuracy: Accuracy of the predictions
    """
    # Load VAD ground truth for the panelist
    print(f"Loading VAD ground truth for Panelist {panelist_id}...")
    vad_data = load_realvad_vad_data(realvad_dir, panelist_id)
    
    # Load bounding box data for the panelist
    print(f"Loading bounding box data for Panelist {panelist_id}...")
    bbox_data = load_realvad_bbox_data(realvad_dir, panelist_id)
    
    # Load the video
    video_path = os.path.join(realvad_dir, 'video.mp4')
    cap = cv2.VideoCapture(video_path)
    
    # Set models to evaluation mode
    resnet_model.eval()
    fcn_model.eval()
    
    # Lists to store results
    predictions = []
    ground_truth = []
    frames = []
    
    # Process frames
    print("Processing video frames...")
    frame_buffer = []
    frame_counter = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only process frames with VAD ground truth
        if frame_counter in vad_data and frame_counter in bbox_data:
            # Get ground truth and bounding box
            speaking_status = vad_data[frame_counter]
            bbox = bbox_data[frame_counter]
            
            # Extract person region using bounding box
            x, y, width, height = bbox
            person_frame = frame[y:y+height, x:x+width]
            
            # Skip if bounding box is invalid
            if person_frame.size == 0:
                frame_counter += 1
                continue
            
            # Resize to a standard size
            person_frame = cv2.resize(person_frame, (224, 224))
            
            # Convert from BGR to RGB
            person_frame = cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB)
            
            # Add frame to buffer
            frame_buffer.append(person_frame)
            
            # Once we have enough frames, create a dynamic image
            if len(frame_buffer) == Config.DI_FRAMES:
                # Create dynamic image
                from src.utils.dynamic_images import compute_dynamic_image
                dynamic_image = compute_dynamic_image(frame_buffer)
                
                # Preprocess for ResNet
                resnet_input = preprocess_dynamic_image(
                    dynamic_image, Config.RESNET_INPUT_SIZE
                ).unsqueeze(0).to(device)
                
                # Get prediction
                with torch.no_grad():
                    outputs, _ = resnet_model(resnet_input)
                    prediction = torch.argmax(outputs, dim=1).item()
                
                # Store results
                predictions.append(prediction)
                ground_truth.append(speaking_status)
                frames.append(dynamic_image)
                
                # Reset buffer
                frame_buffer = []
            
        frame_counter += 1
    
    cap.release()
    
    # Calculate accuracy
    correct = sum([1 for p, gt in zip(predictions, ground_truth) if p == gt])
    accuracy = correct / len(predictions) if predictions else 0
    
    print(f"Processed {len(predictions)} dynamic images.")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Visualize results if requested
    if visualize and frames:
        plt.figure(figsize=(12, 6))
        
        for i in range(min(5, len(frames))):
            plt.subplot(1, 5, i+1)
            plt.imshow(frames[i])
            title = f"Pred: {'Speaking' if predictions[i] == 1 else 'Not Speaking'}\nGT: {'Speaking' if ground_truth[i] == 1 else 'Not Speaking'}"
            plt.title(title)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return predictions, ground_truth, accuracy

def create_output_video(input_path, output_path, predictions, segmentations, dynamic_images):
    """
    Create an output video with VAD results overlaid.
    
    Args:
        input_path: Path to the input video
        output_path: Path to save the output video
        predictions: List of predictions for each dynamic image
        segmentations: List of segmentation masks
        dynamic_images: List of dynamic images
    """
    # Open input video
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames
    frame_idx = 0
    di_idx = 0
    frames_per_di = Config.DI_FRAMES
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Determine which dynamic image this frame belongs to
        if di_idx < len(dynamic_images) and frame_idx // frames_per_di == di_idx:
            # Get current prediction and segmentation
            prediction = predictions[di_idx]
            segmentation = segmentations[di_idx]
            
            # Resize segmentation to match frame size if needed
            if segmentation.shape[:2] != (height, width):
                segmentation = cv2.resize(segmentation.astype(np.uint8), (width, height), 
                                         interpolation=cv2.INTER_NEAREST)
            
            # Create overlay
            overlay = np.zeros_like(frame)
            overlay[segmentation == 1] = [0, 255, 0]  # Green for speaking
            overlay[segmentation == 0] = [255, 0, 0]  # Red for not-speaking
            
            # Add overlay to frame
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            # Add text indicating speaking status
            text = "Speaking" if prediction == 1 else "Not Speaking"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                       (255, 255, 255), 2, cv2.LINE_AA)
        
        # Write frame to output video
        out.write(frame)
        
        # Increment frame index
        frame_idx += 1
        if frame_idx % frames_per_di == 0:
            di_idx += 1
    
    # Release resources
    cap.release()
    out.release()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='S-VVAD Inference')
    parser.add_argument('--realvad', action='store_true', help='Use RealVAD dataset')
    parser.add_argument('--panelist', type=int, default=1, help='Panelist ID for RealVAD (1-9)')
    parser.add_argument('--video', type=str, help='Path to video file (if not using RealVAD)')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda, mps, or cpu)')
    args = parser.parse_args()
    
    # Set device
    if args.device:
        # Use device specified in command line argument
        if args.device == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
        elif args.device == 'mps' and hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        # Use device from config
        if Config.DEVICE == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
        elif Config.DEVICE == 'mps' and hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    else:
        args.output = Config.OUTPUT_DIR
    
    # Load trained models
    print("Loading trained models...")
    
    # Load ResNet VAD model
    resnet_model = ResNetVAD(num_classes=2, pretrained=False)
    try:
        resnet_model.load_state_dict(torch.load(os.path.join(Config.OUTPUT_DIR, 'best_resnet_vad.pth'), map_location=device))
        print("Successfully loaded ResNet VAD model.")
    except:
        print("Failed to load ResNet VAD model. Make sure to train it first using train_resnet_vad.py")
        return
    
    resnet_model = resnet_model.to(device)
    
    # Load FCN segmentation model
    fcn_model = FCN(num_classes=3)
    try:
        fcn_model.load_state_dict(torch.load(os.path.join(Config.OUTPUT_DIR, 'best_fcn_segmentation.pth'), map_location=device))
        print("Successfully loaded FCN segmentation model.")
    except:
        print("Failed to load FCN segmentation model. Make sure to train it first using train_fcn_segmentation.py")
        return
    
    fcn_model = fcn_model.to(device)
    
    # Process video
    start_time = time.time()
    
    if args.realvad:
        # Process RealVAD dataset
        realvad_dir = os.path.join(Config.DATA_ROOT, 'videos', 'RealVAD')
        predictions, ground_truth, accuracy = process_realvad_video(
            realvad_dir, resnet_model, fcn_model, device,
            panelist_id=args.panelist, output_dir=args.output, visualize=args.visualize
        )
        print(f"Accuracy: {accuracy:.4f}")
    elif args.video:
        # Process custom video
        predictions, segmentations = process_video_for_vad(
            args.video, resnet_model, fcn_model, device, args.output, visualize=args.visualize
        )
    else:
        # If no video is specified, prompt for one
        video_path = input("Enter path to video file: ")
        predictions, segmentations = process_video_for_vad(
            video_path, resnet_model, fcn_model, device, args.output, visualize=args.visualize
        )
    
    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main() 