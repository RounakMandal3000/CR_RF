# CR_RF: Cloud Removal using Rectified Flow

A deep learning framework for removing clouds from satellite imagery using Rectified Flow models combined with Variational Autoencoders (VAE) and cloud probability conditioning.

## Overview

CR_RF implements a sophisticated cloud removal pipeline for satellite imagery by leveraging:
- **Rectified Flow (Flux) Models**: A transformer-based flow matching architecture for high-quality cloud-free image generation
- **Variational Autoencoders (VAE)**: For encoding and decoding satellite imagery into latent representations
- **Cloud Probability Conditioning**: Integrates cloud probability maps to guide the reconstruction process
- **Dual-Stream Architecture**: Combines image and text conditioning through double and single stream transformer blocks

## Architecture

### Flux architecture based Model (`Cloud_removal_Flux.py`)
The core model uses a dual-stream transformer architecture with:
- **Double Stream Blocks**: Process both image and text conditioning simultaneously
- **Single Stream Blocks**: Refine features in a unified representation
- **Cloud Probability CNN**: Extracts spatial features from cloud probability maps
- **Positional Embeddings**: N-dimensional embeddings for spatial awareness
- **Timestep Conditioning**: For the flow matching denoising process

Key parameters:
- Configurable hidden dimensions, attention heads, and depth
- MLP ratio for feed-forward layers
- Guidance embedding for classifier-free guidance

### Autoencoder Models

#### Standard Autoencoder (`autoencoder.py`)
- Encoder-decoder architecture with ResNet blocks
- Attention mechanisms for capturing long-range dependencies
- Diagonal Gaussian regularization for latent space

#### Cloud-less VAE (`vae_cloudless.py`)
Specialized VAE trained on cloud-free imagery:
- Multi-resolution downsampling/upsampling
- Channel multiplication for progressive feature extraction
- Configurable resolution and latent channels

## Repository Structure

```
CR_RF/
├── Cloud_removal_Flux.py          # Main Flux transformer model
├── autoencoder.py                 # Standard autoencoder implementation
├── vae_cloudless.py              # VAE trained on cloudless images
├── layers.py                      # Custom transformer layers and blocks
├── math.py                        # Mathematical utilities
├── utils.py                       # General utility functions
├── pipeline.py                    # Inference pipeline
├── cloud_probability_extractor.py # Cloud probability computation
├── rf_training.py                # Rectified Flow training script
├── train_flow.py                 # Flow model training
├── vae_cloudless_training.py     # VAE training for cloudless images
├── vae_sar_training.py           # VAE training for SAR imagery
├── run_cloudprob_nohup.sh        # Shell script for cloud probability extraction
├��─ autoencoder.png               # Autoencoder architecture diagram
└── double_stream.png             # Double stream architecture diagram
```

## Key Components

### 1. Cloud Probability Extraction (`cloud_probability_extractor.py`)
Extracts cloud probability maps from satellite imagery using pre-trained models.

### 2. Training Scripts
- **`rf_training.py`**: Trains the rectified flow model
- **`train_flow.py`**: Flow-based model training pipeline
- **`vae_cloudless_training.py`**: Trains VAE on cloud-free images
- **`vae_sar_training.py`**: Trains VAE on SAR (Synthetic Aperture Radar) data

### 3. Custom Layers (`layers.py`)
Implements specialized components:
- `DoubleStreamBlock`: Dual-pathway processing for image and text
- `SingleStreamBlock`: Unified feature processing
- `EmbedND`: Multi-dimensional positional embeddings
- `MLPEmbedder`: MLP-based feature embedding
- `LastLayer`: Final output layer with adaptive layer normalization

### 4. Inference Pipeline (`pipeline.py`)
End-to-end inference pipeline for cloud removal:
- Model loading utilities
- Noise generation for diffusion process
- Image preprocessing and conditioning
- Text and CLIP embeddings integration


## Model Architecture 

### Parameters
```python
FluxParams(
    in_channels=64,           # Input latent channels
    out_channels=64,          # Output latent channels
    vec_in_dim=768,          # Vector conditioning dimension
    context_in_dim=4096,     # Text context dimension
    hidden_size=3072,        # Transformer hidden size
    mlp_ratio=4.0,          # MLP expansion ratio
    num_heads=24,            # Attention heads
    depth=19,                # Double stream blocks
    depth_single_blocks=38,  # Single stream blocks
    axes_dim=[16, 56, 56],   # Positional embedding dimensions
    theta=10000,             # RoPE theta parameter
    qkv_bias=True,           # Bias in attention
    guidance_embed=True      # Classifier-free guidance
)
```

## Features
**Transformer-based Architecture**: Leverages self-attention for global context understanding  
**Cloud Probability Conditioning**: Uses cloud masks to guide reconstruction  
**Flow Matching**: Rectified flow for efficient generation  
**Multi-Modal Conditioning**: Supports both image and text conditioning  
**Flexible Training**: Modular training scripts for different components  
**Efficient Inference**: Optimized pipeline for production use


This project builds upon:
- Rectified Flow models
- Flux architecture principles
- VAE-based image generation techniques
- Satellite imagery processing methods


**Note**: This project is under active development.
