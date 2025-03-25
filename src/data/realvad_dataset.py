import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from src.utils.dynamic_images import compute_dynamic_image, preprocess_dynamic_image

class RealVADVideoDataset(Dataset):
    """
    Dataset class for the RealVAD dataset that uses the original video and annotations.
    Creates dynamic images on-the-fly.
    """
    def __init__(self, root_dir, panelist_ids=None, input_size=256, 
                 multi_scale=False, scale_range=(256, 320), transform=None):
        """
        Initialize the RealVAD dataset using the original video and annotations.
        
        Args:
            root_dir: Root directory of the RealVAD dataset
            panelist_ids: List of panelist IDs to include (1-9). If None, use all panelists.
            input_size: Size of the input image for the model
            multi_scale: Whether to use multi-scale preprocessing
            scale_range: Range for multi-scale resizing
            transform: Additional transforms to apply
        """
        self.root_dir = root_dir
        self.input_size = input_size
        self.multi_scale = multi_scale
        self.scale_range = scale_range
        self.transform = transform
        self.frames_per_di = 10  # Number of frames to create one dynamic image
        
        if panelist_ids is None:
            panelist_ids = list(range(1, 10))  # Default: all panelists (1-9)
        self.panelist_ids = panelist_ids
        
        # Load video metadata
        self.video_path = os.path.join(root_dir, 'video.mp4')
        cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        
        # Load annotations for all panelists
        self.panelist_data = []
        for panelist_id in self.panelist_ids:
            # Load VAD data
            vad_data = self._load_vad_data(panelist_id)
            # Load bounding box data
            bbox_data = self._load_bbox_data(panelist_id)
            
            # Find frame ranges where we have both VAD and bbox data
            valid_frame_ranges = self._get_valid_frame_ranges(vad_data, bbox_data)
            
            for start_frame, end_frame in valid_frame_ranges:
                # Only include range if it's big enough for a dynamic image
                if end_frame - start_frame + 1 >= self.frames_per_di:
                    # Get all possible dynamic image segments in this range
                    for di_start in range(start_frame, end_frame - self.frames_per_di + 2, self.frames_per_di):
                        di_end = di_start + self.frames_per_di - 1
                        if di_end > end_frame:
                            break
                            
                        # Determine if this segment is speaking
                        speaking_frames = sum(vad_data.get(frame, 0) for frame in range(di_start, di_end + 1))
                        is_speaking = speaking_frames > self.frames_per_di / 2  # Majority rule
                        
                        self.panelist_data.append({
                            'panelist_id': panelist_id,
                            'start_frame': di_start,
                            'end_frame': di_end,
                            'is_speaking': 1 if is_speaking else 0,
                            'bbox_data': {frame: bbox_data[frame] for frame in range(di_start, di_end + 1) if frame in bbox_data}
                        })
        
        print(f"Loaded {len(self.panelist_data)} dynamic image segments from {len(self.panelist_ids)} panelists")
    
    def _load_vad_data(self, panelist_id):
        """Load VAD data for a specific panelist."""
        vad_file = os.path.join(self.root_dir, 'annotations', f'Panelist{panelist_id}_VAD.txt')
        vad_data = {}
        
        with open(vad_file, 'r') as f:
            for line in f:
                data = line.strip().split()
                if len(data) == 2:
                    frame_num = int(data[0])
                    speaking_status = int(data[1])
                    vad_data[frame_num] = speaking_status
        
        return vad_data
    
    def _load_bbox_data(self, panelist_id):
        """Load bounding box data for a specific panelist."""
        bbox_file = os.path.join(self.root_dir, 'annotations', f'Panelist{panelist_id}_bbox.txt')
        bbox_data = {}
        
        with open(bbox_file, 'r') as f:
            for line in f:
                data = line.strip().split()
                if len(data) == 5:
                    frame_num = int(data[0])
                    x = int(data[1])
                    y = int(data[2])
                    width = int(data[3])
                    height = int(data[4])
                    bbox_data[frame_num] = (x, y, width, height)
        
        return bbox_data
    
    def _get_valid_frame_ranges(self, vad_data, bbox_data):
        """
        Find continuous ranges of frames where both VAD and bbox data are available.
        Returns a list of (start_frame, end_frame) tuples.
        """
        # Find frames that have both VAD and bbox data
        valid_frames = sorted(set(vad_data.keys()).intersection(set(bbox_data.keys())))
        
        if not valid_frames:
            return []
            
        ranges = []
        start = valid_frames[0]
        prev = start
        
        for frame in valid_frames[1:]:
            if frame > prev + 1:  # Gap detected
                ranges.append((start, prev))
                start = frame
            prev = frame
            
        # Add the last range
        ranges.append((start, prev))
        
        return ranges
    
    def _extract_frames(self, start_frame, end_frame):
        """Extract a range of frames from the video."""
        frames = []
        cap = cv2.VideoCapture(self.video_path)
        
        # Set the starting position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for _ in range(end_frame - start_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
        cap.release()
        return frames
    
    def __len__(self):
        return len(self.panelist_data)
    
    def __getitem__(self, idx):
        segment = self.panelist_data[idx]
        panelist_id = segment['panelist_id']
        start_frame = segment['start_frame']
        end_frame = segment['end_frame']
        label = segment['is_speaking']
        bbox_data = segment['bbox_data']
        
        # Extract frames
        frames = self._extract_frames(start_frame, end_frame)
        
        # Apply bounding boxes to crop person from each frame
        person_frames = []
        for i, frame in enumerate(frames):
            frame_idx = start_frame + i
            if frame_idx in bbox_data:
                x, y, width, height = bbox_data[frame_idx]
                # Ensure bbox is within frame boundaries
                x = max(0, x)
                y = max(0, y)
                width = min(width, frame.shape[1] - x)
                height = min(height, frame.shape[0] - y)
                
                # Crop person from frame
                if width > 0 and height > 0:
                    person_frame = frame[y:y+height, x:x+width]
                    # Resize to a standard size
                    person_frame = cv2.resize(person_frame, (224, 224))
                    person_frames.append(person_frame)
        
        # If we couldn't extract person frames, return the original frames
        if not person_frames and frames:
            person_frames = [cv2.resize(frame, (224, 224)) for frame in frames]
        
        # Generate dynamic image
        if person_frames:
            dynamic_image = compute_dynamic_image(person_frames)
            
            # Preprocess dynamic image
            image_tensor = preprocess_dynamic_image(
                dynamic_image, 
                self.input_size, 
                self.multi_scale, 
                self.scale_range
            )
            
            # Apply additional transforms if provided
            if self.transform is not None:
                image_tensor = self.transform(image_tensor)
            
            return image_tensor, torch.tensor(label, dtype=torch.long)
        else:
            # Handle the case where no frames were extracted
            # Return a blank image and the label
            blank_image = torch.zeros(3, self.input_size, self.input_size)
            return blank_image, torch.tensor(label, dtype=torch.long)

def create_realvad_video_dataloaders(root_dir, batch_size=32, 
                                   input_size=256, multi_scale=False, scale_range=(256, 320),
                                   panelist_ids=None, validation_split=0.1, shuffle=True,
                                   num_workers=0, pin_memory=False):
    """
    Create train and validation dataloaders for the RealVAD dataset using the original video.
    
    Args:
        root_dir: Root directory of the RealVAD dataset
        batch_size: Batch size
        input_size: Input size for the model
        multi_scale: Whether to use multi-scale preprocessing
        scale_range: Range for multi-scale resizing
        panelist_ids: List of panelist IDs to include (1-9). If None, use all panelists.
        validation_split: Fraction of data to use for validation
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
    """
    # Create full dataset
    dataset = RealVADVideoDataset(
        root_dir=root_dir,
        panelist_ids=panelist_ids,
        input_size=input_size,
        multi_scale=multi_scale,
        scale_range=scale_range
    )
    
    # Determine the split index
    dataset_size = len(dataset)
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - val_size
    
    # Create train and validation datasets
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, 
                            num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, val_loader 