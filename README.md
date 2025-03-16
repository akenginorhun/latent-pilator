# Latent-pilator: Manipulating Facial Images Through Autoencoder Latent Space

## Overview
Latent-pilator is a deep learning project that enables real-time manipulation of facial features through an autoencoder's latent space. Users can modify attributes like age, expressions, smile, and lighting through an interactive GUI by manipulating the learned latent representations.

## Features
- Train autoencoder models on facial images (CelebA dataset)
- Interactive GUI for real-time face manipulation
- Support for various facial attribute modifications
- Latent space visualization and analysis tools
- Quantitative evaluation metrics (PSNR, SSIM, MSE)

## Prerequisites

- Conda (Miniconda or Anaconda)
- Python 3.9 or later
- CelebA dataset (manual download required)

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

The CelebA dataset needs to be downloaded manually from the official source:

a. Download the dataset files from:
   - Main dataset page: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
   - Required files:
     * `img_align_celeba.zip` (Images)
     * `list_attr_celeba.txt` (Attribute annotations)

b. Extract and organize the dataset:
```bash
# Create the data directory
mkdir -p data/celeba

# Extract the image dataset
unzip img_align_celeba.zip -d data/celeba/

# Move the attribute file
mv list_attr_celeba.txt data/celeba/
```

The final directory structure should look like:
```
data/celeba/
├── img_align_celeba/     # Contains all images
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
└── list_attr_celeba.txt  # Attribute annotations
```

## Project Structure
```
latent-pilator/
├── data/                    # Data handling and preprocessing
│   ├── __init__.py
│   └── dataset.py
├── models/                  # Neural network models
│   ├── __init__.py
│   └── autoencoder.py
├── training/               # Training scripts and configs
│   ├── __init__.py
│   └── trainer.py
├── gui/                    # GUI implementation
│   ├── __init__.py
│   └── interface.py
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── metrics.py
│   └── visualization.py
├── configs/               # Configuration files
│   └── config.yaml
├── requirements.txt       # Project dependencies
└── main.py               # Main entry point
```

## Usage

The project can be run in two modes:

1. GUI Mode (default):
```bash
python main.py
```

2. Training Mode:
```bash
python main.py --mode train --config configs/config.yaml
```

## Model Architecture
The project uses a deep convolutional autoencoder architecture with the following key components:
- Encoder: Converts input images into latent representations
- Latent Space: Compressed representation of facial features
- Decoder: Reconstructs images from latent representations

## Evaluation Metrics
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- MSE (Mean Squared Error)
- Latent space disentanglement metrics

## Environment Details

The conda environment includes the following main dependencies:
- Python 3.9
- PyTorch and torchvision
- PyQt5 for GUI
- OpenCV for image processing
- NumPy for numerical operations
- Matplotlib for visualization
- Other utility packages (Pillow, tqdm, etc.)

## Development

To deactivate the conda environment when you're done:
```bash
conda deactivate
```

To update the environment if changes are made to `environment.yml`:
```bash
conda env update -f environment.yml --prune
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- CelebA dataset (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- PyTorch team
- Referenced works in facial feature manipulation