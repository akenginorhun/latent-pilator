import sys
import os
import torch
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QHBoxLayout, QPushButton, QLabel, QFileDialog,
                           QSlider, QGroupBox, QScrollArea)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
import torchvision.transforms as transforms
from PIL import Image
import random
import yaml

# Add parent directory to path to import from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.autoencoder import VAE
from data.dataset import CelebADataset, get_transforms

class LatentPilatorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Latent-pilator")
        self.setGeometry(100, 100, 800, 600)  # Made window smaller
        
        # Load config
        with open('configs/config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize image processing transforms
        self.input_transform, self.output_transform = get_transforms(
            image_size=self.config['data']['image_size']
        )
        
        # Load model and analysis results
        self.load_model_and_directions()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load dataset
        self.dataset = CelebADataset(
            root_dir='data/celeba/img_align_celeba',
            attr_path='data/celeba/list_attr_celeba.txt',
            transform=self.input_transform
        )
        
        # Initialize UI
        self.init_ui()
        
        self.current_image = None
        self.latent_vector = None
        
    def load_model_and_directions(self):
        # Load the trained model
        checkpoint = torch.load('checkpoints/best_model.pt')
        latent_dim = checkpoint['model_state_dict']['encoder.fc_mu.weight'].size(0)  # Get latent dim from model weights
        
        # Initialize model with correct latent dimension
        self.model = VAE(latent_dim=latent_dim)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Initialize attribute directions as identity matrix for now
        # Each row represents a different attribute direction
        self.attribute_directions = torch.eye(latent_dim)  # Default to identity matrix
        
        # Load attribute names
        with open('data/celeba/list_attr_celeba.txt', 'r') as f:
            next(f)  # Skip number of images
            self.attribute_names = next(f).strip().split()
    
    def init_ui(self):
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        layout.setSpacing(20)  # Add some spacing between panels
        
        # Left panel - Image display and buttons
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)  # Reduce spacing between elements
        
        # Original image
        self.original_label = QLabel("Original Image")
        self.original_image = QLabel()
        self.original_image.setFixedSize(self.config['data']['image_size'], self.config['data']['image_size'])
        self.original_image.setStyleSheet("border: 1px solid black")
        
        # Generated image
        self.generated_label = QLabel("Generated Image")
        self.generated_image = QLabel()
        self.generated_image.setFixedSize(self.config['data']['image_size'], self.config['data']['image_size'])
        self.generated_image.setStyleSheet("border: 1px solid black")
        
        # Buttons container
        buttons_container = QWidget()
        buttons_layout = QVBoxLayout(buttons_container)
        buttons_layout.setSpacing(5)  # Minimal spacing between buttons
        
        # Random image button
        self.random_btn = QPushButton("Fetch Random Image")
        self.random_btn.clicked.connect(self.fetch_random_image)
        buttons_layout.addWidget(self.random_btn)
        
        # Reset button
        self.reset_btn = QPushButton("Reset Sliders")
        self.reset_btn.clicked.connect(self.reset_sliders)
        buttons_layout.addWidget(self.reset_btn)
        
        left_layout.addWidget(self.original_label)
        left_layout.addWidget(self.original_image)
        left_layout.addWidget(self.generated_label)
        left_layout.addWidget(self.generated_image)
        left_layout.addWidget(buttons_container)
        left_layout.addStretch()
        
        # Right panel - Controls
        right_panel = QWidget()
        right_scroll = QScrollArea()  # Add scroll area for sliders
        right_scroll.setWidget(right_panel)
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(5)  # Minimal spacing between sliders
        
        # Create attribute sliders
        self.create_attribute_sliders(right_layout)
        
        # Add panels to main layout
        layout.addWidget(left_panel, 1)  # Give less space to left panel
        layout.addWidget(right_scroll, 1)  # Give more space to right panel
    
    def create_attribute_sliders(self, layout):
        self.sliders = {}
        
        # Get number of latent dimensions from the model
        num_latent_dims = self.model.latent_dim
        
        # Create a slider for each latent dimension
        for i in range(num_latent_dims):
            # Create a horizontal container for each slider group
            container = QWidget()
            container_layout = QHBoxLayout(container)
            container_layout.setContentsMargins(5, 5, 5, 5)  # Minimal margins
            
            # Label for dimension
            label = QLabel(f"Dim {i+1}")
            label.setFixedWidth(50)  # Fixed width for labels
            container_layout.addWidget(label)
            
            # Slider with previous style
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(-100)
            slider.setMaximum(100)
            slider.setValue(0)
            slider.setTickPosition(QSlider.TicksBelow)  # Revert to previous style
            slider.setTickInterval(10)  # Revert to previous style
            slider.valueChanged.connect(self.update_image)
            container_layout.addWidget(slider)
            
            layout.addWidget(container)
            self.sliders[i] = slider
            
        # Add stretch at the end to keep sliders at the top
        layout.addStretch()
    
    def fetch_random_image(self):
        # Get a random index
        random_idx = random.randint(0, len(self.dataset) - 1)
        
        # Get the image from dataset
        img_tensor, _ = self.dataset[random_idx]
        
        # Convert tensor to PIL Image for display
        img_np = ((img_tensor * 0.5 + 0.5) * 255).byte().permute(1, 2, 0).numpy()
        image = Image.fromarray(img_np)
        
        # Display and process the image
        self.display_image(image, self.original_image)
        self.current_image = image
        
        # Get latent vector
        with torch.no_grad():
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            self.latent_vector = self.model.encode(img_tensor)
            
        # Generate initial reconstruction
        self.update_image()
    
    def update_image(self):
        if self.latent_vector is None:
            return
            
        # Create manipulation vector based on slider values
        manipulation = torch.zeros_like(self.latent_vector)
        
        for dim, slider in self.sliders.items():
            value = slider.value() / 100.0  # Normalize to [-1, 1]
            # Each row in attribute_directions represents a basis vector
            direction = self.attribute_directions[dim].to(self.device)
            manipulation += value * direction
        
        # Apply manipulation
        with torch.no_grad():
            new_latent = self.latent_vector + manipulation
            generated = self.model.decode(new_latent)
            
            # Convert to image
            generated = generated.cpu().squeeze(0)
            generated = generated * 0.5 + 0.5  # Denormalize
            generated = generated.clamp(0, 1)
            generated = (generated * 255).byte().permute(1, 2, 0).numpy()
            
            # Display generated image
            generated_pil = Image.fromarray(generated)
            self.display_image(generated_pil, self.generated_image)
    
    def display_image(self, pil_image, label):
        # Resize image to fit label while maintaining aspect ratio
        pil_image = pil_image.resize((self.config['data']['image_size'], self.config['data']['image_size']), Image.Resampling.LANCZOS)
        
        # Convert PIL image to QPixmap
        img = pil_image.convert('RGB')
        data = img.tobytes('raw', 'RGB')
        qim = QImage(data, img.size[0], img.size[1], img.size[0] * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qim)
        
        # Display image
        label.setPixmap(pixmap)
    
    def reset_sliders(self):
        for slider in self.sliders.values():
            slider.setValue(0)

def main():
    app = QApplication(sys.argv)
    window = LatentPilatorGUI()
    window.show()
    sys.exit(app.exec_()) 