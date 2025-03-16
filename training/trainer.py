import os
import sys
import platform
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.autoencoder import VAE
from data.dataset import get_celeba_dataloader
from utils.metrics import compute_metrics
from analysis.latent_analysis import LatentSpaceAnalyzer

class Trainer:
    def __init__(self, config, skip_cv=False, visualize_training=False):
        self.config = config
        self.skip_cv = skip_cv
        self.visualize_training = visualize_training
        self.reference_image = None  # Will store our reference image for visualization
        
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
        
        # Initialize analyzer (for cross-val) with training dataloader
        self.analyzer = LatentSpaceAnalyzer(
            config=config,
            dataloader=self.train_loader,
            skip_cv=skip_cv,
            visualize_training=visualize_training,
            reference_image=self.reference_image if self.visualize_training else None,
            reference_image_display=self.reference_image_display if self.visualize_training else None,
            visualization_fig=self.fig if self.visualize_training else None,
            visualization_axes=(self.ax1, self.ax2) if self.visualize_training else None
        )
        
        # Initialize model
        self.model = VAE(
            latent_dim=config['model']['latent_dim'],
            input_channels=config['data']['input_channels'],
            image_size=config['data']['image_size']
        )
        
        # Multi-GPU if available (DataParallel). For better speed, consider DistributedDataParallel.
        if hasattr(self, 'multi_gpu') and self.multi_gpu:
            self.model = nn.DataParallel(self.model)
        
        self.model = self.model.to(self.device)
        
        # We'll create MSELoss once here (rather than inside the loop)
        self.recon_criterion = nn.MSELoss()
        
        # TensorBoard writer
        self.writer = SummaryWriter(config['training']['log_dir'])
        
        # Make checkpoint directory
        os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    
    def find_optimal_latent_dim(self):
        """
        Perform cross-validation to find optimal latent dimension
        """
        print("Starting cross-validation for latent dimensions...")
        best_dim, cv_results = self.analyzer.cross_validate_latent_dim()
        print(f"Optimal latent dimension found: {best_dim}")
        return best_dim, cv_results
    
    def update_visualization(self, model, images=None):
        """
        Update the visualization plot with the reference image and its reconstruction
        """
        if not self.visualize_training:
            return
            
        with torch.no_grad():
            # Get reconstruction of reference image
            recon, _, _ = model(self.reference_image.to(self.device))
            recon = recon[0].cpu()
            
            # Convert reconstruction to numpy for plotting
            recon = recon.permute(1, 2, 0).numpy()
            recon = np.clip(recon, 0, 1)
            
            # Clear previous plots
            self.ax1.clear()
            self.ax2.clear()
            
            # Update plots
            self.ax1.imshow(self.reference_image_display)
            self.ax2.imshow(recon)
            self.ax1.set_title('Original')
            self.ax2.set_title('Reconstruction')
            self.ax1.axis('off')
            self.ax2.axis('off')
            
            # Refresh the plot
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)  # Small pause to allow plot to update

    def train_model(self, model):
        """
        Train the model for the full number of epochs with GPU acceleration
        """
        print("Starting full model training...")
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Initialize mixed precision scaler if on CUDA
        scaler = torch.cuda.amp.GradScaler() if (self.device.type == 'cuda' and self.config['training']['mixed_precision']) else None
        
        # Initialize learning rate scheduler
        scheduler_config = self.config['training'].get('scheduler', {})
        scheduler_name = scheduler_config.get('name', 'plateau')
        scheduler_params = scheduler_config.get('params', {})
        
        if scheduler_name == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_params.get('mode', 'min'),
                factor=scheduler_params.get('factor', 0.1),
                patience=scheduler_params.get('patience', 10),
                threshold=scheduler_params.get('threshold', 1e-4),
                threshold_mode='rel',
                cooldown=scheduler_params.get('cooldown', 0),
                min_lr=scheduler_params.get('min_lr', 1e-6)
            )
        elif scheduler_name == 'one_cycle':
            # Calculate total steps for the scheduler
            total_steps = self.config['training']['num_epochs'] * len(self.train_loader)
            
            # Get scheduler parameters with defaults
            max_lr = scheduler_params.get('max_lr', self.config['training']['learning_rate'])
            pct_start = scheduler_params.get('pct_start', 0.3)
            
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=total_steps,
                pct_start=pct_start,
                cycle_momentum=True,  # Enable momentum cycling
                base_momentum=0.85,   # Minimum momentum
                max_momentum=0.95,    # Maximum momentum
                div_factor=25.0,      # Initial learning rate = max_lr/div_factor
                final_div_factor=1e4  # Final learning rate = max_lr/final_div_factor
            )
        else:
            print(f"Warning: Unknown scheduler '{scheduler_name}', proceeding without scheduling")
            scheduler = None
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['training']['num_epochs']):
            model.train()
            total_loss = 0
            total_recon_loss = 0
            total_kl_loss = 0
            
            # TQDM progress bar
            progress_bar = tqdm(
                self.train_loader,
                desc=f'Epoch {epoch+1}/{self.config["training"]["num_epochs"]}',
                mininterval=1.0  # update bar at least every 1s
            )
            
            for batch_idx, (images, _) in enumerate(progress_bar):
                images = images.to(self.device, non_blocking=True)
                batch_size = images.size(0)
                
                # Update visualization at the start of each epoch
                if batch_idx == 0:
                    self.update_visualization(model, images)
                
                # Zero-grad first, outside the autocast
                optimizer.zero_grad()
                
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        recon_batch, mu, log_var = model(images)
                        recon_loss = self.recon_criterion(recon_batch, images)
                        # Normalize KL loss by batch size
                        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size
                        loss = recon_loss + self.config['training']['kl_weight'] * kl_loss
                    
                    # Scale the loss, then backward
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping if configured
                    if 'max_grad_norm' in self.config['training']:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            self.config['training']['max_grad_norm']
                        )
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Regular FP32 training (CPU or MPS)
                    recon_batch, mu, log_var = model(images)
                    recon_loss = self.recon_criterion(recon_batch, images)
                    # Normalize KL loss by batch size
                    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size
                    loss = recon_loss + self.config['training']['kl_weight'] * kl_loss
                    loss.backward()
                    
                    # Gradient clipping if configured
                    if 'max_grad_norm' in self.config['training']:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            self.config['training']['max_grad_norm']
                        )
                    
                    optimizer.step()
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                
                # Update progress bar's displayed info
                current_lr = optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'recon': f'{recon_loss.item():.4f}',
                    'kl': f'{kl_loss.item():.4f}',
                    'lr': f'{current_lr:.2e}'
                })
                
                # Log to TensorBoard occasionally
                global_step = epoch * len(self.train_loader) + batch_idx
                if batch_idx % self.config['training']['log_interval'] == 0:
                    self.writer.add_scalar('train/loss', loss.item(), global_step)
                    self.writer.add_scalar('train/recon_loss', recon_loss.item(), global_step)
                    self.writer.add_scalar('train/kl_loss', kl_loss.item(), global_step)
                    self.writer.add_scalar('train/lr', current_lr, global_step)
            
            # Average epoch losses
            avg_loss = total_loss / len(self.train_loader)
            avg_recon_loss = total_recon_loss / len(self.train_loader)
            avg_kl_loss = total_kl_loss / len(self.train_loader)
            print(f'====> Epoch: {epoch+1} Average loss: {avg_loss:.4f} (Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}) LR: {current_lr:.2e}')
            
            # Update learning rate scheduler
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_loss)  # Pass validation loss for ReduceLROnPlateau
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f'Learning rate adjusted to: {current_lr:.2e}')
                else:
                    scheduler.step()  # Step for other schedulers like OneCycleLR
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                
                # Save best model if configured
                if self.config['training'].get('save_best_only', False):
                    checkpoint_path = os.path.join(
                        self.config['training']['checkpoint_dir'],
                        'best_model.pt'
                    )
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                        'loss': best_loss,
                    }, checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= self.config['training'].get('early_stopping_patience', float('inf')):
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
                    break
            
            # Regular checkpoint saving
            if not self.config['training'].get('save_best_only', False) and \
               (epoch + 1) % self.config['training']['save_interval'] == 0:
                checkpoint_path = os.path.join(
                    self.config['training']['checkpoint_dir'],
                    f'checkpoint_epoch_{epoch+1}.pt'
                )
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                    'loss': avg_loss,
                }, checkpoint_path)
        
        return model
    
    def analyze_model(self, model):
        """
        Analyze the trained model to find attribute directions
        """
        print("Analyzing model and finding attribute directions...")
        attribute_directions = self.analyzer.find_attribute_directions(model)
        self.analyzer.visualize_latent_space(model)
        return attribute_directions
    
    def train(self):
        """
        Main training pipeline:
        1. If enabled, run cross-validation to find optimal latent dimension
        2. Train final model with optimal/configured dimension
        3. Analyze model
        4. Save final results
        """
        # 1. Cross-validation (if enabled)
        if self.config['cross_validation']['enabled'] and not self.skip_cv:
            print("Starting cross-validation to find optimal latent dimension...")
            best_dim = self.analyzer.run_cross_validation()
            print(f"Cross-validation complete. Optimal latent dimension: {best_dim}")
        else:
            print("Skipping cross-validation, using predefined latent dimension...")
            best_dim = self.config['model']['latent_dim']
        
        print(f"\nTraining final model with latent dimension: {best_dim}")
        
        # 2. Initialize model with best/configured dimension
        self.model = VAE(
            latent_dim=best_dim,
            input_channels=self.config['data']['input_channels'],
            image_size=self.config['data']['image_size']
        ).to(self.device)
        
        # Multi-GPU if available
        if hasattr(self, 'multi_gpu') and self.multi_gpu:
            self.model = nn.DataParallel(self.model)
        
        # 3. Train final model
        self.model = self.train_model(self.model)
        
        # 4. Analyze model
        print("\nAnalyzing trained model...")
        attribute_directions = self.analyzer.find_attribute_directions(self.model)
        self.analyzer.visualize_latent_space(self.model)
        
        # 5. Save final results
        print("\nSaving final model and results...")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'latent_dim': best_dim,
            'attribute_directions': attribute_directions
        }, os.path.join(self.config['training']['checkpoint_dir'], 'final_model.pt'))
        
        print("Training complete! Model and analysis results saved.")

    def __del__(self):
        """
        Cleanup method to close plots when trainer is destroyed
        """
        if self.visualize_training:
            plt.close('all')

def main():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Add argparse to handle command line arguments
    parser = argparse.ArgumentParser(description='Train VAE model')
    parser.add_argument('--skip-cv', action='store_true', help='Skip cross-validation and use predefined latent dimension')
    args = parser.parse_args()
    
    trainer = Trainer(config, skip_cv=args.skip_cv)
    trainer.train()

if __name__ == '__main__':
    main()
