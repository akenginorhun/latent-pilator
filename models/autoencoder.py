import torch
import torch.nn as nn
from data.dataset import get_transforms

class Encoder(nn.Module):
    def __init__(self, latent_dim=32, input_channels=3, input_size=64):
        super(Encoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.input_size = input_size
        
        # Progressive channel scaling
        hidden_dims = [32, 64, 128, 256, 512]
        in_channels = input_channels
        modules = []
        
        # Build encoder layers dynamically
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                             kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
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
        
        # Progressive channel scaling (reversed from encoder)
        hidden_dims = [512, 256, 128, 64, 32]
        self.final_dim = hidden_dims[0]
        
        # Calculate initial spatial dimensions
        test_input = torch.rand(1, output_channels, output_size, output_size)
        test_encoder = Encoder(latent_dim, output_channels, output_size)
        with torch.no_grad():
            test_output = test_encoder.encoder(test_input)
        self.initial_size = test_output.shape[2]  # Assuming square feature maps
        
        # Initial projection from latent space
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0] * self.initial_size * self.initial_size)
        
        # Build decoder layers
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                     hidden_dims[i + 1],
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
        
        self.decoder = nn.Sequential(*modules)
        
        # Final layer to reconstruct image
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                              hidden_dims[-1],
                              kernel_size=3,
                              stride=2,
                              padding=1,
                              output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=output_channels,
                     kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self.final_dim, self.initial_size, self.initial_size)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

class VAE(nn.Module):
    def __init__(self, latent_dim=32, input_channels=3, image_size=64):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim, input_channels, image_size)
        self.decoder = Decoder(latent_dim, input_channels, image_size)
        self.latent_dim = latent_dim
        self.image_size = image_size
        
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
        # Apply output transform
        if self.output_transform is not None:
            result = self.output_transform(result)
        return result

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