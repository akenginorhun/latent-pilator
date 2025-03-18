import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import silhouette_score
from skimage.metrics import structural_similarity as ssim
from data.dataset import CelebADataset

def compute_mse(img1, img2):
    """
    Compute Mean Squared Error between two images
    
    Args:
        img1, img2: torch.Tensor of shape (C, H, W) in range [0, 1]
    """
    return F.mse_loss(img1, img2).item()


def compute_kl_divergence(mu, log_var):
    """
    Compute KL divergence between the encoder distribution and standard normal
    
    Args:
        mu: Mean of the encoder distribution
        log_var: Log variance of the encoder distribution
    """
    return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp()).item()


def compute_vector_consistency(model, dataset, attribute_names, device='cuda', max_samples=100):
    """Measures consistency of attribute vectors computed from different samples."""
    model.eval()
    consistency_scores = {}

    for attr in attribute_names:
        # Compute attribute vectors from two random subsets
        img1, img2 = dataset.get_attribute_images(attr, max_samples//2), dataset.get_attribute_images(attr, max_samples//2)
        v1 = model.extract_attribute_vector(img1['with'].to(device), img1['without'].to(device))
        v2 = model.extract_attribute_vector(img2['with'].to(device), img2['without'].to(device))

        # Compute cosine similarity between the two vectors
        consistency_scores[attr] = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()

    return consistency_scores

def compute_latent_separability(model, dataset, attribute_vectors, device='cuda', max_samples=100):
    """Measures alignment between computed attribute vectors and actual latent differences."""
    model.eval()
    separability_scores = {}

    for attr, attr_vector in attribute_vectors.items():
        # Get images and latent encodings
        img_data = dataset.get_attribute_images(attr, max_samples=max_samples)
        z_with, z_without = model.encode(img_data['with'].to(device)), model.encode(img_data['without'].to(device))

        # Compute mean latent difference and cosine similarity
        latent_diff = (z_with - z_without).mean(dim=0)
        separability_scores[attr] = abs(F.cosine_similarity(latent_diff.unsqueeze(0), attr_vector.unsqueeze(0)).item())

    return separability_scores


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
        attribute_names = dataset.get_attribute_names()
    
    for attr_name in attribute_names:
        # Get images for this attribute using the dataset method
        attr_images = dataset.get_attribute_images(attr_name, max_samples=max_samples)
        
        if attr_images['with'] is None or attr_images['without'] is None:
            print(f"Warning: Could not get images for attribute {attr_name}")
            continue
            
        # Move images to the correct device
        images_with = attr_images['with'].to(device)
        images_without = attr_images['without'].to(device)
        
        # Extract attribute vector using the model
        attr_vector = model.extract_attribute_vector(images_with, images_without)
        attribute_vectors[attr_name] = attr_vector
    return attribute_vectors

def compute_structural_similarity(img1, img2):
    """
    Compute Structural Similarity Index (SSIM) between two images using skimage
    
    Args:
        img1, img2: torch.Tensor of shape (B, C, H, W) or (C, H, W) in range [0, 1]
    """
    # Handle batched tensors by averaging over the batch
    if len(img1.shape) == 4:  # If batched (B, C, H, W)
        ssim_scores = []
        for i in range(img1.size(0)):
            # Convert to numpy arrays and transpose to (H, W, C)
            img1_np = img1[i].cpu().numpy().transpose(1, 2, 0)
            img2_np = img2[i].cpu().numpy().transpose(1, 2, 0)
            
            # Compute SSIM for each channel and average
            channel_scores = []
            for j in range(img1_np.shape[2]):
                channel_scores.append(ssim(img1_np[:,:,j], img2_np[:,:,j], data_range=1.0))
            ssim_scores.append(np.mean(channel_scores))
        return np.mean(ssim_scores)
    else:  # If single image (C, H, W)
        # Convert to numpy arrays and transpose to (H, W, C)
        img1_np = img1.cpu().numpy().transpose(1, 2, 0)
        img2_np = img2.cpu().numpy().transpose(1, 2, 0)
        
        # Compute SSIM for each channel and average
        channel_scores = []
        for i in range(img1_np.shape[2]):
            channel_scores.append(ssim(img1_np[:,:,i], img2_np[:,:,i], data_range=1.0))
        return np.mean(channel_scores)

def compute_latent_clustering(z, labels):
    """
    Compute clustering quality in latent space using silhouette score
    
    Args:
        z: torch.Tensor of shape (N, latent_dim)
        labels: numpy array of shape (N,) containing cluster labels
    """
    z_np = z.cpu().numpy()
    return silhouette_score(z_np, labels)

def evaluate_vae_performance(model, device='mps', max_samples=1000):
    """
    Comprehensive evaluation pipeline for VAE performance
    
    Args:
        model: VAE model instance
        device: Device to run computation on ('mps' or 'cpu')
        max_samples: Maximum number of samples to evaluate
        
    Returns:
        dict: Dictionary containing reconstruction and latent space metrics and scores
              Format:
              {
                  'reconstruction': {
                      'mse': float,
                      'ssim': float,
                      'kl_div': float,
                      'score': float
                  },
                  'latent': {
                      'clustering': float,
                      'separability': float,
                      'consistency': float,
                      'score': float
                  }
              }
    """
    model.eval()
    results = {
        'reconstruction': {},
        'latent': {}
    }
    
    # Create a fresh dataset instance for evaluation using config from the model
    eval_dataset = CelebADataset(
        root_dir=model.config['data']['root_dir'],
        attr_path=model.config['data']['attr_path'],
        target_size=model.config['data']['image_size']
    )
    
    # Sample data from the evaluation dataset
    indices = torch.randperm(len(eval_dataset))[:max_samples]
    images = torch.stack([eval_dataset[i][0] for i in indices]).to(device)
    
    with torch.no_grad():
        # Reconstruction metrics
        recon, mu, log_var = model(images)
        
        # MSE (already normalized by number of pixels)
        results['reconstruction']['mse'] = compute_mse(recon, images)
        
        # SSIM
        results['reconstruction']['ssim'] = compute_structural_similarity(recon, images)
        
        # KL Divergence
        results['reconstruction']['kl_div'] = compute_kl_divergence(mu, log_var)
        
        # Latent space metrics
        z = model.encode(images)
        
        # Latent clustering (using random labels for demonstration)
        # In practice, you might want to use actual attribute labels
        random_labels = np.random.randint(0, 2, size=len(indices))
        results['latent']['clustering'] = compute_latent_clustering(z, random_labels)
        
        # Latent separability (using attribute vectors)
        attribute_vectors = compute_attribute_vectors(model, eval_dataset, device, max_samples=100)
        separability_scores = compute_latent_separability(model, eval_dataset, attribute_vectors, device, max_samples=100)
        results['latent']['separability'] = np.mean(list(separability_scores.values()))
        
        # Vector consistency
        consistency_scores = compute_vector_consistency(model, eval_dataset, eval_dataset.attributes, device, max_samples=100)
        results['latent']['consistency'] = np.mean(list(consistency_scores.values()))
    
    # Compute final scores (normalized between 0 and 1)
    # Reconstruction score (higher is better)
    # Normalize MSE to [0, 1] using exponential decay
    mse_score = np.exp(-results['reconstruction']['mse'])
    # Normalize SSIM from [-1, 1] to [0, 1]
    ssim_score = (results['reconstruction']['ssim'] + 1) / 2
    # Normalize KL divergence to [0, 1] using exponential decay
    kl_score = np.exp(-min(results['reconstruction']['kl_div'], 5))
    
    results['reconstruction']['score'] = (
        0.34 * mse_score +      # MSE (normalized to [0, 1])
        0.33 * ssim_score +     # SSIM (normalized to [0, 1])
        0.33 * kl_score         # KL (normalized to [0, 1])
    )
    
    # Latent space score (higher is better)
    # Normalize clustering from [-1, 1] to [0, 1]
    clustering_score = (results['latent']['clustering'] + 1) / 2
    # Separability is already in [0, 1]
    separability_score = results['latent']['separability']
    # Normalize consistency from [-1, 1] to [0, 1]
    consistency_score = (results['latent']['consistency'] + 1) / 2
    
    results['latent']['score'] = (
        0.34 * clustering_score +      # Clustering (normalized to [0, 1])
        0.33 * separability_score +    # Separability (already in [0, 1])
        0.33 * consistency_score       # Consistency (normalized to [0, 1])
    )
    
    return results