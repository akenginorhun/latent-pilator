import torch
import torch.nn as nn
from data.dataset import get_transforms

class Encoder(nn.Module):
    def __init__(self, latent_dim=32, input_channels=3, input_size=64):
        super(Encoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.input_size = input_size
        
        # Explicit encoder layers
        self.encoder = nn.Sequential(
            # Layer 1: 3 -> 32
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            # Layer 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            # Layer 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            
            # Layer 4: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            
            # Layer 5: 256 -> 512
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )
        
        # Calculate the flattened size dynamically
        test_input = torch.rand(1, input_channels, input_size, input_size)
        test_output = self.encoder(test_input)
        self.flattened_size = test_output.shape[1] * test_output.shape[2] * test_output.shape[3]
        
        # Projection to latent space
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_var = nn.Linear(self.flattened_size, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

class Decoder(nn.Module):
    def __init__(self, latent_dim=32, output_channels=3, output_size=64):
        super(Decoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_size = output_size
        self.final_dim = 512
        
        # Calculate initial spatial dimensions
        # The encoder has 5 layers with stride 2, so we divide by 2^5
        self.initial_size = output_size // 32  # equivalent to output_size / (2^5)
        
        # Initial projection from latent space
        self.decoder_input = nn.Linear(latent_dim, self.final_dim * self.initial_size * self.initial_size)
        
        # Explicit decoder layers
        self.decoder = nn.Sequential(
            # Layer 1: 512 -> 256
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            
            # Layer 2: 256 -> 128
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            
            # Layer 3: 128 -> 64
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            # Layer 4: 64 -> 32
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )
        
        # Final layer to reconstruct image
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, out_channels=output_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self.final_dim, self.initial_size, self.initial_size)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

class VAE(nn.Module):
    def __init__(self, latent_dim=32, input_channels=3, image_size=64, config=None):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim, input_channels, image_size)
        self.decoder = Decoder(latent_dim, input_channels, image_size)
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.config = config  # Store the configuration
        
        # Get transforms
        self.input_transform, self.output_transform = get_transforms(image_size)

    def forward(self, x):
        # Apply input transform if x is not already a tensor
        if not isinstance(x, torch.Tensor):
            x = self.input_transform(x)
        
        mu, log_var = self.encoder(x)
        z = self.encoder.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def encode(self, x):
        # Apply input transform if x is not already a tensor
        if not isinstance(x, torch.Tensor):
            x = self.input_transform(x)
            
        mu, log_var = self.encoder(x)
        return self.encoder.reparameterize(mu, log_var)

    def decode(self, z):
        result = self.decoder(z)
        # Apply output transform if available
        if self.output_transform is not None:
            result = self.output_transform(result)
        return result

    def extract_attribute_vector(self, images_with_attr, images_without_attr):
        """
        Extract the attribute vector by computing the difference between the mean latent
        representations of images with and without a specific attribute.
        
        Args:
            images_with_attr (torch.Tensor): Batch of images with the attribute
            images_without_attr (torch.Tensor): Batch of images without the attribute
            
        Returns:
            torch.Tensor: The attribute vector in latent space
        """
        with torch.no_grad():
            # Encode both sets of images
            z_with_attr = self.encode(images_with_attr)
            z_without_attr = self.encode(images_without_attr)
            
            # Compute mean latent vectors
            mean_with_attr = torch.mean(z_with_attr, dim=0)
            mean_without_attr = torch.mean(z_without_attr, dim=0)
            
            # Compute attribute vector as the difference
            attribute_vector = mean_with_attr - mean_without_attr
            
            return attribute_vector

    def manipulate_latent(self, x, direction, strength):
        """
        Manipulate the latent representation along a specific direction
        
        Args:
            x (torch.Tensor): Input image
            direction (torch.Tensor): Direction in latent space
            strength (float): Strength of manipulation
        """
        # Apply input transform if x is not already a tensor
        if not isinstance(x, torch.Tensor):
            x = self.input_transform(x)
            
        z = self.encode(x.unsqueeze(0))
        z_new = z + direction * strength
        return self.decode(z_new) 