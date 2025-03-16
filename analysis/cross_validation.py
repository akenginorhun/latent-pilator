import os
import sys
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import get_celeba_dataloader
from training.trainer import Trainer

class CrossValidator:
    def __init__(self, config, visualize_training=False):
        """
        Initialize the cross validator with configuration
        
        Args:
            config: Dictionary containing configuration parameters
            visualize_training: Whether to visualize training progress
        """
        print("\n=== Initializing CrossValidator ===")
        self.config = config
        self.visualize_training = visualize_training
        
        # Load data with CV-specific batch size
        self.dataloader = get_celeba_dataloader(
            root_dir=config['data']['root_dir'],
            attr_path=config['data']['attr_path'],
            batch_size=config['training']['batch_size'],
            num_workers=min(os.cpu_count(), config['training']['num_workers']),
            pin_memory=True,
            target_size=config['data']['image_size']  # Pass the image size from config
        )[0]  # Only take the train loader

    def sample_dataset(self, n_samples):
        """
        Create a sampled subset of the dataset for faster cross-validation
        """
        dataset = self.dataloader.dataset
        total_size = len(dataset)
        
        if n_samples >= total_size:
            return dataset
            
        indices = torch.randperm(total_size)[:n_samples].tolist()
        return Subset(dataset, indices)

    def run_cross_validation(self, sample_size, n_folds, dimensions):
        """
        Run k-fold cross-validation to find optimal latent dimension.
        Returns the dimension with best average performance across folds.
        """
        print("\nStarting Cross-Validation")
        
        # Setup
        sampled_dataset = self.sample_dataset(sample_size)
        indices = list(range(len(sampled_dataset)))
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Results storage
        results = {}
        
        for dim in dimensions:
            print(f"\nTesting Latent Dimension: {dim}")
            fold_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
                print(f"Fold {fold + 1}/{n_folds}")
                
                # Create data loaders for this fold
                train_subset = Subset(sampled_dataset, train_idx)
                val_subset = Subset(sampled_dataset, val_idx)
                
                train_loader = DataLoader(
                    train_subset,
                    batch_size=self.config['training']['batch_size'],
                    shuffle=True,
                    num_workers=2
                )
                val_loader = DataLoader(
                    val_subset,
                    batch_size=self.config['training']['batch_size'],
                    shuffle=False,
                    num_workers=2
                )
                
                # Create a new model with current dimension
                cv_config = self.config.copy()
                cv_config['model'] = self.config['model'].copy()  # Deep copy the model config
                cv_config['model']['latent_dim'] = dim
                cv_config['model']['input_channels'] = self.config['model'].get('input_channels', 3)
                cv_config['model']['image_size'] = self.config['data']['image_size']  # Ensure image size is passed
                
                # Create trainer for this fold with visualization parameter
                trainer = Trainer(cv_config, visualize_training=self.visualize_training)
                
                # Train and evaluate
                model = trainer.train_model(train_loader=train_loader, val_loader=val_loader)
                val_score = trainer.evaluate_model(model, val_loader)
                fold_scores.append(val_score)
            
            results[dim] = sum(fold_scores) / len(fold_scores)
        
        # Find best dimension
        best_dim = min(results.items(), key=lambda x: x[1])[0]
        
        # Print summary
        print("\n" + "="*50)
        print("Cross-Validation Summary")
        print("="*50)
        for dim, score in results.items():
            print(f"Dimension {dim}: {score:.4f}")
        
        print("\nBest latent dimension:", best_dim)
        print("="*50)
        
        return best_dim
