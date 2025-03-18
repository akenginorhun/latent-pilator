# Latent-pilator: Manipulating Facial Images Through Autoencoder Latent Space

This project was developed as a final project for CS131 Computer Vision at Stanford University.

## Overview
Latent-pilator is a deep learning project that enables real-time manipulation of facial features through an autoencoder's latent space. The project uses a VAE (Variational Autoencoder) architecture that learns to encode and decode facial images, and allows users to interactively manipulate the facial attributes through the learned latent representations.

## Features
- VAE-based face manipulation with cross-validation for optimal latent dimension
- Interactive GUI for real-time face manipulation
- Comprehensive training pipeline with visualization options
- Cross-validation support for model optimization
- Quantitative evaluation metrics (PSNR, SSIM, MSE)
- GPU-accelerated training with multi-GPU support

## Prerequisites
- Conda (Miniconda or Anaconda)
- Python 3.9
- CUDA-capable GPU (recommended for training)
- CelebA dataset

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/latent-pilator.git
cd latent-pilator
```

2. Create and activate the conda environment:
```bash
# Create the environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate latent-pilator
```

3. Download and Setup CelebA Dataset:
   - Download from: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
   - Required files:
     * `img_align_celeba.zip` (Images)
     * `list_attr_celeba.txt` (Attribute annotations)
   - Extract to `data/celeba/` directory

## Running the Code

The project supports two main modes: training and GUI. Command-line arguments:

- `--mode`: Choose between 'train' or 'gui' (default: 'gui')
- `--config`: Path to configuration file (default: 'configs/config.yaml')
- `--skip-cv`: Skip cross-validation and use predefined latent dimension
- `--visualize`: Show live reconstruction visualization during training

### Training Mode Examples

```bash
# Train with cross-validation to find optimal latent dimension
python main.py --mode train

# Train with custom configuration file
python main.py --mode train --config configs/custom_config.yaml

# Train with predefined latent dimension (skip cross-validation)
python main.py --mode train --skip-cv

# Train with live visualization of reconstructions
python main.py --mode train --visualize

# Combine multiple flags
python main.py --mode train --config configs/custom_config.yaml --visualize
```

### GUI Mode
```bash
python main.py --mode gui
```

## Hyperparameters
The project's hyperparameters can be configured in `configs/config.yaml`. Key parameters include:

### Model Parameters
- `latent_dim`: Dimension of the latent space (default: 64)
- `input_channels`: Number of input channels (default: 3 for RGB)

### Training Parameters
- `num_epochs`: Number of training epochs
- `batch_size`: Batch size for training
- `learning_rate`: Learning rate for optimization
- `kl_weight`: Weight for KL divergence loss
- `weight_decay`: L2 regularization factor

### Cross-validation Parameters
- `sample_size`: Number of samples for cross-validation
- `n_folds`: Number of folds for k-fold CV
- `dimensions`: List of latent dimensions to test

### Resource Parameters
- `num_workers`: Number of data loading workers
- `multi_gpu`: Enable multi-GPU training
- `pin_memory`: Enable pinned memory for faster data transfer

## Project Structure
```
latent-pilator/
├── analysis/              # Analysis and evaluation tools
│   └── cross_validation.py
├── configs/              # Configuration files
│   └── config.yaml
├── data/                 # Data handling and preprocessing
│   └── dataset.py
├── gui/                  # GUI implementation
│   └── interface.py
├── models/              # Neural network models
│   └── autoencoder.py
├── training/           # Training scripts
│   └── trainer.py
├── utils/             # Utility functions
│   ├── metrics.py
│   └── visualization.py
├── trained_models/    # Saved model checkpoints
├── environment.yml    # Conda environment specification
└── main.py           # Main entry point
```

## File Descriptions

- `main.py`: Entry point for both training and GUI modes
- `models/autoencoder.py`: VAE model architecture implementation
- `training/trainer.py`: Training loop and optimization logic
- `analysis/cross_validation.py`: Cross-validation implementation
- `gui/interface.py`: PyQt5-based GUI implementation
- `data/dataset.py`: CelebA dataset loader and preprocessing
- `utils/metrics.py`: Evaluation metrics (PSNR, SSIM, MSE)
- `utils/visualization.py`: Training visualization tools
- `configs/config.yaml`: Configuration file for all hyperparameters

## Acknowledgments
- CelebA dataset (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- VAE architecture reference: [A Basic Variational Autoencoder in PyTorch Trained on the CelebA Dataset](https://medium.com/the-generator/a-basic-variational-autoencoder-in-pytorch-trained-on-the-celeba-dataset-f29c75316b26)
- Latent space manipulation approach: [VAE for the CelebA dataset](https://goodboychan.github.io/python/coursera/tensorflow_probability/icl/2021/09/14/03-Variational-AutoEncoder-Celeb-A.html)