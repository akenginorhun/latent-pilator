import argparse
import os
import yaml
from training.trainer import Trainer
from analysis.cross_validation import CrossValidator
from gui.interface import LatentPilatorGUI
from PyQt5.QtWidgets import QApplication
import sys
import torch

def run_cross_validation(config, visualize_training=False):
    """
    Run cross-validation to find the optimal latent dimension
    
    Args:
        config: Configuration dictionary
        visualize_training: Whether to visualize training progress
    
    Returns:
        optimal_latent_dim: The best performing latent dimension
    """
    cross_validator = CrossValidator(config, visualize_training=visualize_training)
    
    # Get cross-validation parameters from config
    cv_params = config.get('training', {}).get('cross_validation', {})
    sample_size = cv_params.get('sample_size', 1000)
    n_folds = cv_params.get('n_folds', 3)
    dimensions = cv_params.get('dimensions', [32, 64, 128, 256])
    
    optimal_latent_dim = cross_validator.run_cross_validation(
        sample_size=sample_size,
        n_folds=n_folds,
        dimensions=dimensions
    )
    
    return optimal_latent_dim

def train(config_path, skip_cv=False, visualize_training=False):
    """
    Train the VAE model using the specified configuration
    
    Args:
        config_path: Path to the config file
        skip_cv: If True, skip the cross-validation phase
        visualize_training: If True, show live reconstruction visualization
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Run cross-validation if not skipped
    if not skip_cv:
        optimal_latent_dim = run_cross_validation(config, visualize_training=visualize_training)
        print(f"\nUsing optimal latent dimension from cross-validation: {optimal_latent_dim}")
        
        # Update config with optimal latent dimension
        config['model']['latent_dim'] = optimal_latent_dim
    
    # Initialize and start training
    trainer = Trainer(config, visualize_training=visualize_training)
    trainer.train_model()

def launch_gui():
    """
    Launch the GUI interface
    """
    app = QApplication(sys.argv)
    window = LatentPilatorGUI()
    window.show()
    sys.exit(app.exec_())

def main():
    parser = argparse.ArgumentParser(description='Latent-pilator: Face Manipulation through Latent Space')
    parser.add_argument('--mode', type=str, choices=['train', 'gui'], default='gui',
                      help='Mode to run the application in (train or gui)')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                      help='Path to configuration file for training')
    parser.add_argument('--skip-cv', action='store_true',
                      help='Skip cross-validation and use predefined latent dimension')
    parser.add_argument('--visualize', action='store_true',
                      help='Show live reconstruction visualization during training')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        if not os.path.exists(args.config):
            print(f"Error: Config file {args.config} not found!")
            return
        train(args.config, skip_cv=args.skip_cv, visualize_training=args.visualize)
    else:
        launch_gui()

if __name__ == '__main__':
    main() 