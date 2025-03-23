# S-VVAD: Visual Voice Activity Detection by Motion Segmentation

This is a PyTorch implementation of the paper [S-VVAD: Visual Voice Activity Detection by Motion Segmentation](https://openaccess.thecvf.com/content/WACV2021/papers/Shahid_S-VVAD_Visual_Voice_Activity_Detection_by_Motion_Segmentation_WACV_2021_paper.pdf) by Muhammad Shahid, Cigdem Beyan, and Vittorio Murino (WACV 2021).

## Overview

S-VVAD is a novel visual Voice Activity Detection (VAD) method that operates directly on entire video frames, without explicitly detecting a person or body parts. It learns body motion cues associated with speech activity within a weakly supervised segmentation framework, so it not only detects speakers but also localizes their positions in the image.

The implementation follows a two-stage pipeline:
1. A ResNet50-based model is fine-tuned to classify dynamic images as speaking or not-speaking
2. Class Activation Maps (CAMs) from the first stage are used to train a Fully Convolutional Network (FCN) for segmentation

## Key Features

- End-to-end visual voice activity detection
- Person-independent (works on new people without retraining)
- No need for face/lip detection
- Simultaneous detection and localization of speakers
- Works with multiple people in the same frame
- Integration with the RealVAD dataset for training and evaluation

## Requirements

See `requirements.txt` for the full list of dependencies. The main requirements are:

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.20.0
opencv-python>=4.5.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
pillow>=8.2.0
tqdm>=4.60.0
tensorboard>=2.5.0
```

To install the dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── src
│   ├── config.py              # Configuration parameters
│   ├── inference.py           # Inference script for processing videos
│   ├── train_resnet_vad.py    # Training script for ResNet VAD model
│   ├── train_fcn_segmentation.py  # Training script for FCN segmentation
│   ├── data
│   │   └── dataset.py         # Dataset handling utilities
│   ├── models
│   │   ├── resnet_vad.py      # ResNet50 model for VAD
│   │   └── fcn_segmentation.py    # FCN model for segmentation
│   └── utils
│       └── dynamic_images.py  # Utilities for dynamic image generation
├── data                       # Data directory (to be populated)
│   └── videos                 # Video files for training and testing
├── output                     # Output directory for models and results
└── requirements.txt           # Project dependencies
```

## Data Preparation

This implementation includes support for the RealVAD dataset, which is structured as follows:

```
data/videos/RealVAD/
├── video.mp4                      # Main video file
├── dynamic_images/
│   ├── annotations.txt            # Speaking/not-speaking labels for dynamic images
│   └── img/                       # Pre-computed dynamic images
└── annotations/
    ├── PanelistX_bbox.txt         # Bounding box coordinates for each panelist
    └── PanelistX_VAD.txt          # Speaking/not-speaking labels for each frame
```

The RealVAD dataset structure is already set up in this implementation. The annotations include:
- Speaking/not-speaking labels for each dynamic image
- Bounding box coordinates for each panelist in each frame
- Frame-level voice activity labels for each panelist

If you want to use your own dataset, place your video files in the `data/videos` directory and adapt the data loading code in `src/data/dataset.py`.

## Usage

### 1. Training the ResNet VAD Model

```bash
python -m src.train_resnet_vad
```

This will train the ResNet50-based model on the RealVAD dataset to classify dynamic images as speaking or not-speaking.

### 2. Training the FCN Segmentation Model

```bash
python -m src.train_fcn_segmentation
```

This will use the trained ResNet VAD model to generate CAMs and train the FCN model for segmentation.

### 3. Running Inference

For general inference on a video file:
```bash
python -m src.inference --video /path/to/video.mp4 --visualize
```

For inference using the RealVAD dataset:
```bash
python -m src.inference --realvad --panelist 1 --visualize
```

Command line options:
- `--realvad`: Use the RealVAD dataset
- `--panelist N`: Specify the panelist ID (1-9) to analyze
- `--video PATH`: Path to a custom video file
- `--output DIR`: Output directory for results
- `--visualize`: Show visualization of results

## Customization

The configuration parameters can be modified in `src/config.py`, including:

- Dynamic image parameters (number of frames per dynamic image)
- Model parameters (input size, multi-scale settings)
- Training parameters (batch size, learning rate, epochs)
- Device configuration (automatically selects the best available device: CUDA, MPS, or CPU)

### Device Selection

The code automatically selects the best available device:
- CUDA if you have an NVIDIA GPU
- MPS (Metal Performance Shaders) if you have an Apple Silicon Mac (M1/M2/M3)
- CPU as fallback

You can also manually specify the device during inference:
```bash
python -m src.inference --realvad --panelist 1 --device mps
```

Supported device options:
- `cuda`: NVIDIA GPU acceleration
- `mps`: Apple Silicon GPU acceleration
- `cpu`: CPU execution (slower but always available)

## Citation

If you use this implementation in your research, please cite the original paper:

```
@InProceedings{Shahid_2021_WACV,
    author    = {Shahid, Muhammad and Beyan, Cigdem and Murino, Vittorio},
    title     = {S-VVAD: Visual Voice Activity Detection by Motion Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2021},
    pages     = {2332-2341}
}
``` 