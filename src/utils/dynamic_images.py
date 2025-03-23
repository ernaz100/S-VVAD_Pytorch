import numpy as np
import cv2
import torch

def compute_dynamic_image(frames):
    """
    Compute dynamic image from a sequence of frames using the method from [Bilen et al., 2016]
    as described in the S-VVAD paper.
    
    Args:
        frames: List of frames (numpy arrays of shape [H, W, C])
        
    Returns:
        dynamic_image: Computed dynamic image (numpy array of shape [H, W, C])
    """
    num_frames = len(frames)
    
    if num_frames <= 1:
        return frames[0]
    
    # Compute the rank coefficients based on the number of frames
    # using the formula from the original dynamic image paper
    rank_coeff = np.arange(1, num_frames + 1)
    rank_coeff = 2 * rank_coeff - num_frames - 1
    
    # Compute weighted sum of frames
    dynamic_image = np.zeros_like(frames[0], dtype=np.float32)
    for i, frame in enumerate(frames):
        dynamic_image += rank_coeff[i] * frame.astype(np.float32)
    
    # Normalize the dynamic image
    dynamic_image = np.clip(dynamic_image, 0, 255).astype(np.uint8)
    
    return dynamic_image

def generate_dynamic_images_from_video(video_path, sequence_length):
    """
    Generate dynamic images from a video file.
    
    Args:
        video_path: Path to the video file
        sequence_length: Number of frames to create one dynamic image
        
    Returns:
        dynamic_images: List of generated dynamic images
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    dynamic_images = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        
        if len(frames) == sequence_length:
            dynamic_image = compute_dynamic_image(frames)
            dynamic_images.append(dynamic_image)
            frames = []  # Reset frames for the next dynamic image
    
    # Process any remaining frames
    if len(frames) > 1:
        dynamic_image = compute_dynamic_image(frames)
        dynamic_images.append(dynamic_image)
    
    cap.release()
    return dynamic_images

def preprocess_dynamic_image(dynamic_image, input_size, multi_scale=False, scale_range=(256, 320)):
    """
    Preprocess a dynamic image for input to the ResNet model.
    Implements both single-scale and multi-scale preprocessing as described in the paper.
    
    Args:
        dynamic_image: The dynamic image to preprocess
        input_size: Final input size for the model
        multi_scale: Whether to use multi-scale preprocessing
        scale_range: Range for multi-scale resizing
        
    Returns:
        processed_image: Preprocessed image tensor
    """
    if multi_scale:
        # Multi-scale preprocessing
        k = np.random.randint(scale_range[0], scale_range[1] + 1)
        resized_image = cv2.resize(dynamic_image, (k, k))
        
        # Random crop of input_size x input_size
        if k > input_size:
            start_h = np.random.randint(0, k - input_size + 1)
            start_w = np.random.randint(0, k - input_size + 1)
            resized_image = resized_image[start_h:start_h+input_size, start_w:start_w+input_size]
    else:
        # Single-scale preprocessing
        resized_image = cv2.resize(dynamic_image, (input_size, input_size))
    
    # Convert to PyTorch tensor and normalize
    image_tensor = torch.from_numpy(resized_image.transpose(2, 0, 1)).float() / 255.0
    
    # Normalize with ImageNet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    return image_tensor 