import os
import sys
import torch
import torch.nn as nn
import numpy as np
import platform
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.utils import shuffle

# Add parent directory to path to allow imports from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import get_celeba_dataloader
from models.autoencoder import VAE

class LatentSpaceAnalyzer:
    def __init__(self, config, dataloader=None, skip_cv=False, visualize_training=False,
                 reference_image=None, reference_image_display=None, visualization_fig=None,
                 visualization_axes=None):
        """
        Initialize the analyzer with configuration
        
        Args:
            config: Dictionary containing configuration parameters
            dataloader: Optional pre-initialized dataloader
            skip_cv: Whether to skip cross-validation
            visualize_training: Whether to show live reconstruction visualization
            reference_image: Reference image tensor for visualization
            reference_image_display: Reference image numpy array for display
            visualization_fig: Matplotlib figure for visualization
            visualization_axes: Tuple of (ax1, ax2) for visualization
        """
        print("\n=== Initializing LatentSpaceAnalyzer ===")
        
        # Store visualization parameters
        self.visualize_training = visualize_training
        self.reference_image = reference_image
        self.reference_image_display = reference_image_display
        self.fig = visualization_fig
        self.ax1, self.ax2 = visualization_axes if visualization_axes else (None, None)
        
        # Configure device
        if torch.backends.mps.is_available() and platform.processor() == 'arm':
            self.device = torch.device('mps')
            print("Using Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("Using CUDA GPU")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")
            
        self.config = config
        self.skip_cv = skip_cv
        
        # Use provided dataloader or create new one
        if dataloader is not None:
            self.dataloader = dataloader
        else:
            # Load data with CV-specific batch size
            self.dataloader = get_celeba_dataloader(
                root_dir=config['data']['root_dir'],
                attr_path=config['data']['attr_path'],
                batch_size=config['cross_validation']['batch_size'],
                num_workers=min(os.cpu_count(), config['training']['num_workers']),
                pin_memory=True
            )[0]  # Only take the train loader
        
        # CV parameters from config
        self.cv_sample_size = config['cross_validation']['sample_size']
        self.cv_n_folds = config['cross_validation']['n_folds']
        self.cv_epochs = config['cross_validation']['epochs']
        self.latent_dims = config['cross_validation']['dimensions']
        
    def sample_dataset(self, n_samples):
        """
        Create a sampled subset of the dataset for faster cross-validation
        
        Args:
            n_samples: Number of samples to include in the subset
            
        Returns:
            Subset of the original dataset
        """
        dataset = self.dataloader.dataset
        total_size = len(dataset)
        
        if n_samples >= total_size:
            return dataset
            
        # Generate random indices for sampling
        indices = torch.randperm(total_size)[:n_samples].tolist()
        return Subset(dataset, indices)

    def update_visualization(self, model):
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
            plt.pause(0.01)

    def train_model(self, model, train_loader):
        """
        Quick training loop using a DataLoader for cross-validation
        Returns the average loss across all epochs
        """
        model.train()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config['cross_validation']['learning_rate'],
            weight_decay=self.config['cross_validation']['weight_decay']
        )
        
        epoch_losses = []
        for epoch in range(self.cv_epochs):
            total_loss = 0
            num_batches = len(train_loader)
            
            with tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{self.cv_epochs}", 
                     leave=False, dynamic_ncols=True, position=0) as progress_bar:
                
                # Update visualization at the start of each epoch
                if self.visualize_training:
                    self.update_visualization(model)
                
                for batch_idx, (images, _) in enumerate(progress_bar):
                    images = images.to(self.device)
                    
                    optimizer.zero_grad()
                    recon_batch, mu, log_var = model(images)
                    
                    recon_loss = nn.MSELoss()(recon_batch, images)
                    # Normalize KL loss by batch size
                    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / images.size(0)
                    
                    loss = recon_loss + self.config['cross_validation']['kl_weight'] * kl_loss
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            print('\033[K', end='')
            epoch_losses.append(total_loss / num_batches)
            
        if self.visualize_training:
            self.update_visualization(model)

        torch.cuda.empty_cache()
        return np.mean(epoch_losses)

    def run_cross_validation(self):
        """
        Run k-fold cross-validation to find optimal latent dimension.
        Returns the dimension with best average performance across folds.
        """
        print("\nStarting Cross-Validation")
        
        # Setup
        sampled_dataset = self.sample_dataset(self.cv_sample_size)
        indices = list(range(len(sampled_dataset)))
        kf = KFold(n_splits=self.cv_n_folds, shuffle=True, random_state=42)
        
        # Track progress
        total_steps = len(self.latent_dims) * self.cv_n_folds * self.cv_epochs
        current_step = 0
        
        # Results storage
        results = {}
        cv_summary = {}
        
        for dim in self.latent_dims:
            print(f"\nTesting Latent Dimension: {dim}")
            fold_scores = []
            cv_summary[dim] = {'recon_scores': [], 'attr_scores': [], 'combined_scores': []}
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
                current_step += self.cv_epochs
                print(f"Fold {fold + 1}/{self.cv_n_folds} (Progress: {current_step}/{total_steps} steps)")
                
                train_subset = Subset(sampled_dataset, train_idx)
                val_subset = Subset(sampled_dataset, val_idx)
                
                train_loader = DataLoader(
                    train_subset,
                    batch_size=self.config['cross_validation']['batch_size'],
                    shuffle=True,
                    num_workers=2
                )
                val_loader = DataLoader(
                    val_subset,
                    batch_size=self.config['cross_validation']['batch_size'],
                    shuffle=False,
                    num_workers=2
                )
                
                model = VAE(
                    latent_dim=dim,
                    input_channels=self.config['data']['input_channels'],
                    image_size=self.config['data']['image_size']
                ).to(self.device)
                
                avg_loss = self.train_model(model, train_loader)
                
                recon_score = self.evaluate_reconstruction(model, val_loader)  # Now smaller is better
                attr_score = self.evaluate_attribute_prediction(model, val_loader)  # Larger is better
                
                # Normalize reconstruction score to [0,1] range (1 is best) for combination
                max_mse = 1.0  # Assuming pixel values are in [0,1]
                normalized_recon_score = 1.0 - min(recon_score / max_mse, 1.0)
                
                # Combine scores with equal weight (both now larger is better)
                combined_score = 0.5 * normalized_recon_score + 0.5 * attr_score
                
                cv_summary[dim]['recon_scores'].append(recon_score)  # Store original MSE
                cv_summary[dim]['attr_scores'].append(attr_score)
                cv_summary[dim]['combined_scores'].append(combined_score)
                fold_scores.append(combined_score)
            
            results[dim] = np.mean(fold_scores)
        
        # Find best dimension
        best_dim = max(results.items(), key=lambda x: x[1])[0]
        
        # Print summary
        print("\n" + "="*50)
        print("Cross-Validation Summary")
        print("="*50)
        for dim in self.latent_dims:
            recon_mean = np.mean(cv_summary[dim]['recon_scores'])
            recon_std = np.std(cv_summary[dim]['recon_scores'])
            attr_mean = np.mean(cv_summary[dim]['attr_scores'])
            attr_std = np.std(cv_summary[dim]['attr_scores'])
            combined_mean = np.mean(cv_summary[dim]['combined_scores'])
            combined_std = np.std(cv_summary[dim]['combined_scores'])
            
            print(f"\nDimension {dim}:")
            print(f"  Reconstruction MSE: {recon_mean:.4f} ± {recon_std:.4f} (lower is better)")
            print(f"  Attribute Score: {attr_mean:.4f} ± {attr_std:.4f} (higher is better)")
            print(f"  Combined Score: {combined_mean:.4f} ± {combined_std:.4f} (higher is better)")
        
        print("\nBest latent dimension:", best_dim)
        print("="*50)
        
        # Save CV results
        cv_results = {
            'best_dim': best_dim,
            'results': results,
            'summary': cv_summary
        }
        torch.save(cv_results, 'analysis/cv_results.pt')
        
        return best_dim

    def evaluate_reconstruction(self, model, val_loader):
        """
        Evaluate reconstruction quality using MSE.
        Returns the average MSE per sample (smaller values indicate better reconstruction).
        """
        model.eval()
        total_mse = 0
        total_samples = 0

        with torch.no_grad():
            for images, _ in tqdm(val_loader, desc="Evaluating recon", leave=False):
                images = images.to(self.device)
                recon_batch, _, _ = model(images)
                
                # Compute MSE (smaller is better)
                mse = nn.MSELoss(reduction='sum')(recon_batch, images).item()
                total_mse += mse
                total_samples += images.size(0)
        
        # Return average MSE per sample
        return total_mse / total_samples if total_samples > 0 else float('inf')
    
    def evaluate_attribute_prediction(self, model, val_loader):
        """
        Evaluate how well the latent space preserves attribute information
        """
        model.eval()
        latent_vecs = []
        attr_list = []
        
        with torch.no_grad():
            for images, attributes in tqdm(val_loader, desc="Evaluating attrs", leave=False):
                images = images.to(self.device)
                mu, _ = model.encoder(images)
                latent_vecs.append(mu.cpu())
                attr_list.append(attributes)
        
        latent_vecs = torch.cat(latent_vecs, dim=0).numpy()
        attr_array = torch.cat(attr_list, dim=0).numpy()
        
        total_score = 0
        num_valid_attrs = 0

        for attr_idx in range(attr_array.shape[1]):
            column_values = attr_array[:, attr_idx].numpy() if hasattr(attr_array[:, attr_idx], 'numpy') \
                                                             else attr_array[:, attr_idx]
            unique_vals = np.unique(column_values)
            
            if len(unique_vals) < 2:
                continue

            clf = LinearSVC(max_iter=2000)
            try:
                clf.fit(latent_vecs, column_values)
                score = clf.score(latent_vecs, column_values)
                total_score += score
                num_valid_attrs += 1
            except Exception:
                continue

        return total_score / max(num_valid_attrs, 1)
    
    def find_attribute_directions(self, model, n_components=10):
        """
        Find meaningful attribute directions in latent space using PCA
        """
        print("Finding attribute directions...")
        model.eval()
        
        # Collect latent vectors
        latent_vecs = []
        all_attributes = []
        
        with torch.no_grad():
            for images, attrs in tqdm(self.dataloader, desc="Encoding images"):
                images = images.to(self.device)
                mu, _ = model.encoder(images)
                latent_vecs.append(mu.cpu())
                all_attributes.append(attrs)
        
        latent_vecs = torch.cat(latent_vecs, dim=0).numpy()
        all_attributes = torch.cat(all_attributes, dim=0).numpy()
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        pca.fit(latent_vecs)
        
        # Find correlation between PCA components and each attribute
        correlations = {}
        for i, component in enumerate(pca.components_):
            proj = latent_vecs @ component
            for j in range(all_attributes.shape[1]):
                corr = np.corrcoef(proj, all_attributes[:, j])[0, 1]
                correlations[(i, j)] = abs(corr)
        
        # Sort correlations by absolute value
        sorted_correlations = sorted(
            correlations.items(), key=lambda x: x[1], reverse=True
        )
        
        # Assign each PCA component to the attribute with which it has the highest correlation
        attribute_directions = {}
        used_components = set()
        used_attributes = set()
        
        for (comp_idx, attr_idx), corr in sorted_correlations:
            if comp_idx not in used_components and attr_idx not in used_attributes:
                direction = torch.from_numpy(pca.components_[comp_idx]).float()
                attribute_directions[attr_idx] = direction
                used_components.add(comp_idx)
                used_attributes.add(attr_idx)
                
                if len(attribute_directions) == n_components:
                    break
        
        return attribute_directions
    
    def visualize_latent_space(self, model, save_path='analysis/latent_viz.png'):
        """
        Visualize latent space using t-SNE
        """
        print("Visualizing latent space...")
        model.eval()
        
        # Collect latent vectors and attributes
        latent_vecs = []
        all_attributes = []
        
        with torch.no_grad():
            for images, attrs in tqdm(self.dataloader, desc="Collecting latent vectors"):
                images = images.to(self.device)
                mu, _ = model.encoder(images)
                latent_vecs.append(mu.cpu())
                all_attributes.append(attrs)
        
        latent_vecs = torch.cat(latent_vecs, dim=0).numpy()
        all_attributes = torch.cat(all_attributes, dim=0).numpy()
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        latent_2d = tsne.fit_transform(latent_vecs)
        
        # Plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1],
                              c=all_attributes[:, 0], alpha=0.5)
        plt.colorbar(scatter)
        plt.title('t-SNE visualization of latent space')
        plt.savefig(save_path)
        plt.close()

def main():
    print("\n=== Starting Latent Space Analysis ===")
    
    # Load config
    print("Loading configuration...")
    import yaml
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    print("\nLoading CelebA dataset...")
    from data.dataset import get_celeba_dataloader
    dataloader = get_celeba_dataloader(
        root_dir=config['data']['root_dir'],
        attr_path=config['data']['attr_path'],
        batch_size=config['training']['batch_size']
    )
    print(f"Dataset loaded with batch size: {config['training']['batch_size']}")
    
    # Initialize analyzer
    analyzer = LatentSpaceAnalyzer(config)
    
    # Find optimal latent dimension via cross-validation
    print("\nStarting cross-validation to find optimal latent dimension...")
    best_dim = analyzer.run_cross_validation()
    print(f"\nCross-validation complete!")
    print(f"Best latent dimension found: {best_dim}")
    
    # Train final model with best dimension on full dataset
    print(f"\nTraining final model with optimal dimension {best_dim} on full dataset...")
    from models.autoencoder import VAE
    final_model = VAE(
        latent_dim=best_dim,
        input_channels=config['data']['input_channels'],
        image_size=config['data']['image_size']
    ).to(analyzer.device)
    
    # Use the trainer class for final training to ensure all config parameters are used
    from training.trainer import Trainer
    trainer = Trainer(config, skip_cv=True)  # skip_cv since we already did it
    trainer.model = final_model  # Replace the model with our best_dim model
    final_model = trainer.train_model(final_model)  # Use trainer's train_model method
    
    # Find attribute directions
    print("\nAnalyzing attribute directions in latent space...")
    attribute_directions = analyzer.find_attribute_directions(final_model)
    
    # Visualize latent space
    print("\nGenerating t-SNE visualization of latent space...")
    analyzer.visualize_latent_space(final_model)
    
    # Save results
    print("\nSaving analysis results...")
    torch.save({
        'best_dim': best_dim,
        'attribute_directions': attribute_directions,
        'model_state': final_model.state_dict()  # Save the final trained model
    }, 'analysis/latent_analysis_results.pt')
    
    print("\n=== Analysis Complete! ===")
    print(f"Results saved to: analysis/latent_analysis_results.pt")
    print(f"Visualization saved to: analysis/latent_viz.png")

if __name__ == '__main__':
    main()
