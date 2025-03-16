import torch
import torch.nn.functional as F
import numpy as np
from scipy import signal
from PIL import Image

def compute_mse(img1, img2):
    """
    Compute Mean Squared Error between two images
    
    Args:
        img1, img2: torch.Tensor of shape (C, H, W) in range [0, 1]
    """
    return F.mse_loss(img1, img2).item()

def compute_psnr(img1, img2):
    """
    Compute Peak Signal to Noise Ratio
    
    Args:
        img1, img2: torch.Tensor of shape (C, H, W) in range [0, 1]
    """
    mse = compute_mse(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

def gaussian_kernel(size=11, sigma=1.5):
    """
    Create a Gaussian kernel for SSIM computation
    """
    x = np.linspace(-size//2, size//2, size)
    x, y = np.meshgrid(x, x)
    g = np.exp(-(x**2 + y**2)/(2*sigma**2))
    return g/g.sum()

def compute_ssim(img1, img2, window_size=11):
    """
    Compute Structural Similarity Index (SSIM)
    
    Args:
        img1, img2: torch.Tensor of shape (C, H, W) in range [0, 1]
        window_size: Size of the sliding window
    """
    # Constants for stability
    C1 = (0.01 * 1)**2
    C2 = (0.03 * 1)**2
    
    # Create Gaussian kernel
    kernel = gaussian_kernel(window_size)
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = kernel.expand(img1.size(0), 1, window_size, window_size)
    
    # Compute means
    mu1 = F.conv2d(img1.unsqueeze(0), kernel, padding=window_size//2)
    mu2 = F.conv2d(img2.unsqueeze(0), kernel, padding=window_size//2)
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    # Compute variances and covariance
    sigma1_sq = F.conv2d(img1.unsqueeze(0)**2, kernel, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(img2.unsqueeze(0)**2, kernel, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(img1.unsqueeze(0) * img2.unsqueeze(0), kernel, padding=window_size//2) - mu1_mu2
    
    # Compute SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()

def compute_metrics(original, generated):
    """
    Compute all metrics between original and generated images
    
    Args:
        original: torch.Tensor of shape (C, H, W) in range [0, 1]
        generated: torch.Tensor of shape (C, H, W) in range [0, 1]
    
    Returns:
        dict: Dictionary containing MSE, PSNR, and SSIM values
    """
    metrics = {
        'mse': compute_mse(original, generated),
        'psnr': compute_psnr(original, generated),
        'ssim': compute_ssim(original, generated)
    }
    return metrics 