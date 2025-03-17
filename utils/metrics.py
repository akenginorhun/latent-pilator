import torch
import torch.nn.functional as F
import numpy as np
from scipy import signal
from PIL import Image
from sklearn.metrics import silhouette_score
from scipy.stats import entropy

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

def compute_kl_divergence(mu, log_var):
    """
    Compute KL divergence between the encoder distribution and standard normal
    
    Args:
        mu: Mean of the encoder distribution
        log_var: Log variance of the encoder distribution
    """
    return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp()).item()

def compute_latent_space_metrics(latent_vectors, labels=None):
    """
    Compute metrics about the latent space organization
    
    Args:
        latent_vectors: torch.Tensor of shape (N, latent_dim)
        labels: Optional labels for computing supervised metrics
    
    Returns:
        dict: Dictionary containing latent space metrics
    """
    metrics = {}
    
    # Convert to numpy for sklearn compatibility
    latent_np = latent_vectors.detach().cpu().numpy()
    
    # Compute average pairwise distances
    pairwise_distances = torch.cdist(latent_vectors, latent_vectors)
    metrics['avg_pairwise_distance'] = pairwise_distances.mean().item()
    
    # Compute latent space coverage (variance explained)
    latent_cov = np.cov(latent_np.T)
    eigenvalues = np.linalg.eigvals(latent_cov)
    metrics['latent_variance_explained'] = np.sum(np.real(eigenvalues))
    
    # If labels are provided, compute supervised metrics
    if labels is not None:
        # Silhouette score for cluster separation
        try:
            metrics['silhouette_score'] = silhouette_score(latent_np, labels.cpu().numpy())
        except:
            metrics['silhouette_score'] = 0.0
    
    return metrics

def compute_interpolation_smoothness(model, start_points, end_points, n_steps=10):
    """
    Compute smoothness of latent space interpolation
    
    Args:
        model: VAE model
        start_points: Starting points in latent space
        end_points: Ending points in latent space
        n_steps: Number of interpolation steps
    
    Returns:
        float: Average reconstruction difference between consecutive interpolation points
    """
    with torch.no_grad():
        alphas = torch.linspace(0, 1, n_steps).to(start_points.device)
        
        # Interpolate in latent space
        interpolations = torch.stack([
            (1 - alpha) * start_points + alpha * end_points
            for alpha in alphas
        ])
        
        # Generate reconstructions
        reconstructions = model.decode(interpolations.view(-1, start_points.size(-1)))
        
        # Compute average difference between consecutive reconstructions
        diff = torch.mean(torch.abs(reconstructions[1:] - reconstructions[:-1]))
        
        return diff.item()

def compute_reconstruction_metrics(original_batch, reconstructed_batch):
    """
    Compute all reconstruction metrics for a batch of images
    
    Args:
        original_batch: torch.Tensor of shape (B, C, H, W)
        reconstructed_batch: torch.Tensor of shape (B, C, H, W)
    
    Returns:
        dict: Dictionary containing reconstruction quality metrics
    """
    metrics = {}
    
    # Compute average metrics across batch
    mse_values = []
    psnr_values = []
    ssim_values = []
    
    for orig, recon in zip(original_batch, reconstructed_batch):
        mse_values.append(compute_mse(orig, recon))
        psnr_values.append(compute_psnr(orig, recon))
        ssim_values.append(compute_ssim(orig, recon))
    
    metrics['mse'] = np.mean(mse_values)
    metrics['psnr'] = np.mean(psnr_values)
    metrics['ssim'] = np.mean(ssim_values)
    
    return metrics

def evaluate_vae_performance(model, data_loader, device, n_interpolation_samples=10):
    """
    Comprehensive evaluation of VAE performance
    
    Args:
        model: VAE model
        data_loader: DataLoader containing validation/test data
        device: Device to run evaluation on
        n_interpolation_samples: Number of samples for interpolation evaluation
    
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    model.eval()
    metrics = {
        'reconstruction': {'mse': 0.0, 'psnr': 0.0, 'ssim': 0.0},
        'latent': {'kl_divergence': 0.0},
        'interpolation': {'smoothness': 0.0}
    }
    
    n_batches = 0
    all_latent_vectors = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            batch_size = images.size(0)
            
            # Forward pass
            recon_batch, mu, log_var = model(images)
            
            # Compute reconstruction metrics
            batch_metrics = compute_reconstruction_metrics(images, recon_batch)
            for key in batch_metrics:
                metrics['reconstruction'][key] += batch_metrics[key] * batch_size
            
            # Compute KL divergence
            metrics['latent']['kl_divergence'] += compute_kl_divergence(mu, log_var) * batch_size
            
            # Store latent vectors and labels for later analysis
            all_latent_vectors.append(mu)
            all_labels.append(labels)
            
            n_batches += 1
            
            # Compute interpolation metrics on a subset of samples
            if batch_idx == 0:
                idx1 = torch.randperm(batch_size)[:n_interpolation_samples]
                idx2 = torch.randperm(batch_size)[:n_interpolation_samples]
                metrics['interpolation']['smoothness'] = compute_interpolation_smoothness(
                    model, mu[idx1], mu[idx2]
                )
    
    # Normalize metrics by total number of samples
    total_samples = n_batches * data_loader.batch_size
    for key in metrics['reconstruction']:
        metrics['reconstruction'][key] /= total_samples
    metrics['latent']['kl_divergence'] /= total_samples
    
    # Compute latent space metrics
    all_latent_vectors = torch.cat(all_latent_vectors, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics['latent'].update(compute_latent_space_metrics(all_latent_vectors, all_labels))
    
    return metrics

def compute_attribute_vectors(model, dataset, device='cuda', max_samples=100, attribute_names=None):
    """
    Compute attribute vectors for all attributes in the dataset.
    
    Args:
        model: VAE model
        dataset: CelebADataset instance
        device: Device to run computation on
        max_samples: Maximum number of samples to use per attribute category
    
    Returns:
        dict: Dictionary mapping attribute names to their corresponding vectors
              Format: {attr_name: vector_tensor}
    """
    model.eval()
    attribute_vectors = {}
    
    # Get all attribute names from dataset
    if attribute_names is None:
        attribute_names = dataset.attributes
    
    for attr_name in attribute_names:
        # Get images for this attribute using the dataset method
        print(f"Getting images for {attr_name}")
        attr_images = dataset.get_attribute_images(attr_name, max_samples=max_samples)
        
        # Extract attribute vector using the model
        print(f"Extracting attribute vector for {attr_name}")
        attr_vector = model.extract_attribute_vector(
            attr_images['with'],
            attr_images['without']
        )
        attribute_vectors[attr_name] = attr_vector
    return attribute_vectors