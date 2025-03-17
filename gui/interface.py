import sys
import os
import torch
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QHBoxLayout, QPushButton, QLabel, QFileDialog,
                           QSlider, QGroupBox, QScrollArea, QMessageBox)
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
from utils.analysis import compute_attribute_vectors

class LatentPilatorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Latent-pilator")
        self.setGeometry(100, 100, 800, 600)
        
        # Load config
        with open('configs/config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize image processing transforms
        self.input_transform, self.output_transform = get_transforms(
            image_size=self.config['data']['image_size']
        )
        
        # Initialize variables
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load dataset
        self.dataset = CelebADataset(
            root_dir=self.config['data']['root_dir'],
            attr_path=self.config['data']['attr_path'],
            transform=self.input_transform
        )
        
        # Initialize UI
        self.init_ui()
        
        self.current_image = None
        self.latent_vector = None
        self.attribute_directions = None
        
    def init_ui(self):
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        layout.setSpacing(20)
        
        # Left panel - Image display and buttons
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)
        
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
        buttons_layout.setSpacing(5)
        
        # Upload model button
        self.upload_btn = QPushButton("Upload Model")
        self.upload_btn.clicked.connect(self.upload_model)
        buttons_layout.addWidget(self.upload_btn)
        
        # Random image button (disabled until model is loaded)
        self.random_btn = QPushButton("Fetch Random Image")
        self.random_btn.clicked.connect(self.fetch_random_image)
        self.random_btn.setEnabled(False)
        buttons_layout.addWidget(self.random_btn)
        
        # Reset button (disabled until model is loaded)
        self.reset_btn = QPushButton("Reset Sliders")
        self.reset_btn.clicked.connect(self.reset_sliders)
        self.reset_btn.setEnabled(False)
        buttons_layout.addWidget(self.reset_btn)
        
        left_layout.addWidget(self.original_label)
        left_layout.addWidget(self.original_image)
        left_layout.addWidget(self.generated_label)
        left_layout.addWidget(self.generated_image)
        left_layout.addWidget(buttons_container)
        left_layout.addStretch()
        
        # Right panel - Controls
        right_panel = QWidget()
        right_scroll = QScrollArea()
        right_scroll.setWidget(right_panel)
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.right_layout = QVBoxLayout(right_panel)
        self.right_layout.setSpacing(5)
        
        # Add panels to main layout
        layout.addWidget(left_panel, 1)
        layout.addWidget(right_scroll, 1)
        
        # Add status label
        self.status_label = QLabel("Please upload a model to begin")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.right_layout.addWidget(self.status_label)
    
    def upload_model(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(self, "Select Model Checkpoint", "", "Checkpoint Files (*.pt);;All Files (*)")
            if not file_name:
                return
                
            # Load the checkpoint
            try:
                checkpoint = torch.load(file_name, map_location=self.device)
                if 'model_state_dict' not in checkpoint:
                    raise ValueError("Invalid checkpoint format: missing model_state_dict")
                    
                latent_dim = checkpoint['model_state_dict']['encoder.fc_mu.weight'].size(0)
                
                # Initialize model with correct dimensions
                self.model = VAE(
                    latent_dim=latent_dim,
                    input_channels=self.config['model']['input_channels'],
                    image_size=self.config['data']['image_size'],
                    config=self.config
                )
                
                # Load state dict with strict=False to handle potential mismatches
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                self.model = self.model.to(self.device)
                self.model.eval()
                
                # Update status
                self.status_label.setText("Extracting attribute vectors...")
                QApplication.processEvents()
                
                # Compute attribute vectors using the metrics function
                self.attribute_directions = compute_attribute_vectors(
                    model=self.model,
                    dataset=self.dataset,
                    device=self.device,
                    max_samples=100
                )
                
                # Create sliders for attributes
                self.create_attribute_sliders(self.right_layout)
                
                # Enable buttons
                self.random_btn.setEnabled(True)
                self.reset_btn.setEnabled(True)
                
                # Update status
                self.status_label.setText("Model loaded successfully")
                
            except (KeyError, RuntimeError) as e:
                raise ValueError(f"Error loading model checkpoint: {str(e)}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            self.status_label.setText("Error loading model")
            self.model = None
            self.attribute_directions = None
    
    def create_attribute_sliders(self, layout):
        # Clear existing sliders
        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().setParent(None)
        
        self.sliders = {}
        
        # Create a slider for each attribute
        for attr_name in self.attribute_directions.keys():
            container = QWidget()
            container_layout = QHBoxLayout(container)
            container_layout.setContentsMargins(5, 5, 5, 5)
            
            # Label for attribute
            label = QLabel(attr_name.replace('_', ' ').title())
            label.setFixedWidth(150)
            container_layout.addWidget(label)
            
            # Slider
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(-100)
            slider.setMaximum(100)
            slider.setValue(0)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(10)
            slider.valueChanged.connect(self.update_image)
            container_layout.addWidget(slider)
            
            layout.addWidget(container)
            self.sliders[attr_name] = slider
        
        # Add stretch at the end to keep sliders at the top
        layout.addStretch()
    
    def fetch_random_image(self):
        if self.model is None:
            QMessageBox.warning(self, "Warning", "Please load a model first")
            return
            
        # Get a random index
        random_idx = random.randint(0, len(self.dataset) - 1)
        
        # Get the image from dataset
        img_tensor, _ = self.dataset[random_idx]
        
        # Convert tensor to PIL Image for display
        # Note: Dataset outputs values in [0, 1] range
        img_np = (img_tensor * 255).byte().permute(1, 2, 0).numpy()
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
        if self.model is None or self.latent_vector is None:
            return
            
        # Create manipulation vector based on slider values
        manipulation = torch.zeros_like(self.latent_vector)
        
        for attr_name, slider in self.sliders.items():
            value = slider.value() / 100.0  # Normalize to [-1, 1]
            direction = self.attribute_directions[attr_name].to(self.device)
            manipulation += value * direction
        
        # Apply manipulation
        with torch.no_grad():
            new_latent = self.latent_vector + manipulation
            generated = self.model.decode(new_latent)
            
            # Convert to image
            generated = generated.cpu().squeeze(0)
            # Note: Model output is in [0, 1] range due to sigmoid activation
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
        if self.model is None:
            return
        for slider in self.sliders.values():
            slider.setValue(0)

def main():
    app = QApplication(sys.argv)
    window = LatentPilatorGUI()
    window.show()
    sys.exit(app.exec_()) 