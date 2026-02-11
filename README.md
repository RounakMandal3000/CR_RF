# Cloud Removal Flux

Transformer-based architecture for spatiotemporal cloud removal in remote-sensing imagery, implemented with PyTorch.

## Overview

- Implements the `CR` model defined in `Cloud_removal_Flux.py` for flow-matching between cloudy and cloud-free observations.
- Extends multi-stream attention blocks (`DoubleStreamBlock`, `SingleStreamBlock`) and sinusoidal positional embeddings (`EmbedND`) from the local `layers` module.
- Supports guidance conditioning and vector features fused through a lightweight CNN encoder.

## Repository Structure

- `Cloud_removal_Flux.py` – Main model definition with input embeddings, attention blocks, and final projection head.
- `layers.py` (expected) – Required building blocks such as attention layers, positional embeddings, timestep embeddings, and final head utilities.

## Requirements

- Python 3.10+
- PyTorch 2.1+

## SLURM Jobs

- `cloud_vae.sh`: Submits a GPU job for training the cloudless VAE (`vae_cloudless_training.py`). Activates the `cloud` Conda environment and logs outputs to `cloud_vae_v3.out`/`.err`.
- `cloud_sar_vae.sh`: Launches the SAR-conditioned VAE training script (`vae_sar_training.py`) with identical resource requests (1 GPU, 8 CPU cores, 32 GB RAM, 70 h wall time).
- `rf_training.sh`: Schedules the random-feature or transformer training pipeline (`rf_training.py`) using the same environment and resource profile.
This code takes the pretrained VAEs and the pre-built datasets to train the entire RF model from scratch

