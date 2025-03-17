import os
import sys
import platform
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
import torch.nn.functional as F
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.autoencoder import VAE
from data.dataset import get_celeba_dataloader
from utils.analysis import evaluate_vae_performance

class Trainer:
    def __init__(self, config, visualize_training=False):
        self.config = config
        self.visualize_training = visualize_training
        self.reference_image = None  # Will store our reference image for visualization
        
        # Create models directory if it doesn't exist
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trained_models')
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize visualization if enabled
        if self.visualize_training:
            plt.ion()  # Enable interactive mode
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
            self.fig.suptitle('Training Progress: Original vs Reconstruction')
            self.ax1.set_title('Original')
            self.ax2.set_title('Reconstruction')
            plt.show(block=False)
        
        # Detect and configure device
        if torch.backends.mps.is_available() and platform.processor() == 'arm':
            self.device = torch.device('mps')
            print("Using Apple Silicon GPU (MPS) for training")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
                self.multi_gpu = True
            else:
                self.multi_gpu = False
        else:
            self.device = torch.device('cpu')
            print("No GPU found, using CPU for training")
        
        # Initialize dataloader with recommended arguments
        self.train_loader, self.val_loader = get_celeba_dataloader(
            root_dir=config['data']['root_dir'],
            attr_path=config['data']['attr_path'],
            batch_size=config['training']['batch_size'],
            num_workers=min(os.cpu_count(), config['training']['num_workers']),
            pin_memory=True,
            prefetch_factor=config['training']['prefetch_factor'],
            target_size=config['data']['image_size']
        )
        
        # Get a reference image for visualization before initializing analyzer
        if self.visualize_training:
            # Get first batch
            images, _ = next(iter(self.train_loader))
            idx = random.randint(0, images.size(0) - 1)
            self.reference_image = images[idx:idx+1].clone()
            self.reference_image_display = images[idx].cpu().permute(1, 2, 0).numpy()
            self.reference_image_display = np.clip(self.reference_image_display, 0, 1)
        
        # Initialize model
        self.model = VAE(
            latent_dim=config['model']['latent_dim'],
            input_channels=config['model']['input_channels'],
            image_size=config['data']['image_size'],
            config=config  # Pass the configuration
        )
        
        # Multi-GPU if available (DataParallel). For better speed, consider DistributedDataParallel.
        if hasattr(self, 'multi_gpu') and self.multi_gpu:
            self.model = nn.DataParallel(self.model)
        
        self.model = self.model.to(self.device)

    def update_visualization(self, images=None):
        """
        Update the visualization plot with the reference image and its reconstruction
        """
        if not self.visualize_training:
            return
            
        with torch.no_grad():
            if self.reference_image is None and images is not None:
                idx = np.random.randint(0, images.size(0))
                self.reference_image = images[idx:idx+1].clone()
                self.reference_image_display = images[idx].cpu().permute(1, 2, 0).numpy()
                self.reference_image_display = np.clip(self.reference_image_display, 0, 1)
            
            recon, _, _ = self.model(self.reference_image.to(self.device))
            recon = recon[0].cpu().permute(1, 2, 0).numpy()
            recon = np.clip(recon, 0, 1)
            
            self.ax1.clear()
            self.ax2.clear()
            self.ax1.imshow(self.reference_image_display)
            self.ax2.imshow(recon)
            self.ax1.set_title('Original')
            self.ax2.set_title('Reconstruction')
            self.ax1.axis('off')
            self.ax2.axis('off')
            
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)  # Small pause to allow plot to update

    def compute_loss(self, recon_batch, images, mu, log_var, batch_size):
        """
        Compute the VAE loss function.
        """
        # Compute reconstruction loss (MSE) on flattened tensors
        recon_loss = F.mse_loss(recon_batch.view(batch_size, -1), images.view(batch_size, -1))
        
        # Compute KL divergence loss using mean instead of sum/batch_size
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Compute total loss with KL weight
        total_loss = recon_loss + self.config['training']['kl_weight'] * kl_loss
        
        return total_loss, recon_loss, kl_loss

    def evaluate_model(self, model, print_metrics=False):
        """
        Evaluate model performance using comprehensive metrics
        
        Returns:
            float: A composite score that considers both reconstruction and latent space metrics.
                  Higher scores indicate better performance.
        """
        # Evaluate using the comprehensive evaluation function
        evaluation_metrics = evaluate_vae_performance(
            model=model,
            device=self.device
        )
        
        # Print detailed metrics
        if print_metrics:
            print("\nEvaluation Metrics:")
            print("Reconstruction:")
            print(f"  MSE: {evaluation_metrics['reconstruction']['mse']:.4f}")
            print(f"  SSIM: {evaluation_metrics['reconstruction']['ssim']:.4f}")
            print(f"  KL Divergence: {evaluation_metrics['reconstruction']['kl_div']:.4f}")
            print("\nLatent Space:")
            print(f"  Clustering Score: {evaluation_metrics['latent']['clustering']:.4f}")
            print(f"  Separability: {evaluation_metrics['latent']['separability']:.4f}")
            print(f"  Consistency: {evaluation_metrics['latent']['consistency']:.4f}")
        
        # Compute weighted composite score
        # We'll use the pre-computed scores from evaluate_vae_performance
        # which are already normalized between 0 and 1
        composite_score = (
            0.5 * evaluation_metrics['reconstruction']['score'] +  # Reconstruction score (higher is better)
            0.5 * evaluation_metrics['latent']['score']           # Latent space score (higher is better)
        )
        
        return composite_score

    def save_final_model(self):
        """
        Save the final trained model with latent dimension in filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        latent_dim = self.config['model']['latent_dim']
        filename = f'vae_latent{latent_dim}_{timestamp}.pt'
        save_path = os.path.join(self.models_dir, filename)
        
        # If using DataParallel, save the internal module
        model_to_save = self.model.module if hasattr(self, 'multi_gpu') and self.multi_gpu else self.model
        torch.save(model_to_save.state_dict(), save_path)
        print(f"\nFinal model saved to: {save_path}")

    def train_model(self, train_loader=None, num_epochs=None):
        """
        Train the model with the given parameters.
        """
        if train_loader is None:
            train_loader = self.train_loader
        if num_epochs is None:
            num_epochs = self.config['training']['num_epochs']
        
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Initialize learning rate scheduler
        scheduler = None
        scheduler_config = self.config['training'].get('scheduler', {})
        if scheduler_config.get('name') == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                **scheduler_config.get('params', {})
            )
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            
            progress_bar = tqdm(
                train_loader,
                desc=f'Epoch {epoch+1}/{num_epochs}',
                mininterval=1.0
            )
            
            for batch_idx, (images, _) in enumerate(progress_bar):
                images = images.to(self.device, non_blocking=True)
                batch_size = images.size(0)
                
                if batch_idx == 0:
                    self.update_visualization(images)
                
                optimizer.zero_grad()
                
                recon_batch, mu, log_var = self.model(images)
                loss, recon_loss, kl_loss = self.compute_loss(
                    recon_batch, images, mu, log_var, batch_size
                )
                loss.backward()
                if 'max_grad_norm' in self.config['training']:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['max_grad_norm']
                    )
                optimizer.step()
                
                total_loss += loss.item()
                
                current_lr = optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'recon': f'{recon_loss.item():.4f}',
                    'kl': f'{kl_loss.item():.4f}',
                    'lr': f'{current_lr:.2e}'
                })
            
            avg_loss = total_loss / len(train_loader)
            
            # Update scheduler with training loss
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_loss)  # Use average training loss
            
            # Evaluate and print metrics every epoch
            val_score = self.evaluate_model(model=self.model, print_metrics=True)
            print(f'====> Epoch: {epoch+1} Average loss: {avg_loss:.4f} Validation score: {val_score:.4f}')

        # Save the final trained model
        self.save_final_model()
        return self.model

    def __del__(self):
        """
        Cleanup method to close plots when trainer is destroyed
        """
        if self.visualize_training:
            plt.close('all')

