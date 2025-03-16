import argparse
import os
import yaml
from training.trainer import Trainer
from gui.interface import LatentPilatorGUI
from PyQt5.QtWidgets import QApplication
import sys

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
    
    # Create necessary directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('analysis', exist_ok=True)
    
    # Initialize and start training
    trainer = Trainer(config, skip_cv=skip_cv, visualize_training=visualize_training)
    trainer.train()

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