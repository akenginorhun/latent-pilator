import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim=32, input_channels=3, input_size=64):
        super(Encoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.input_size = input_size
        
        # Calculate number of downsampling steps needed
        n_downsample = max(2, min(5, (input_size - 1).bit_length() - 3))  # Automatically determine number of layers
        
        # Calculate dimensions for each layer
        dims = []
        current_dim = input_size
        base_channels = 32  # Base number of channels
        channels = [input_channels] + [min(base_channels * (2**i), 512) for i in range(n_downsample)]
        
        for i in range(len(channels)-1):
            # Calculate output dimension after conv layer
            # Formula: output_size = (input_size + 2*padding - kernel_size) / stride + 1
            current_dim = (current_dim + 2*1 - 4) // 2 + 1
            dims.append(current_dim)
        
        # Build encoder layers dynamically
        layers = []
        for i in range(len(channels)-1):
            layers.extend([
                nn.Conv2d(channels[i], channels[i+1], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(channels[i+1]),
                nn.LeakyReLU(0.2)
            ])
        
        self.encoder = nn.Sequential(*layers)
        
        # Calculate the flattened size based on the final feature map size
        self.flattened_size = channels[-1] * dims[-1] * dims[-1]
        
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
        
        # Calculate number of upsampling steps needed
        n_upsample = max(2, min(5, (output_size - 1).bit_length() - 3))  # Match encoder's downsample count
        
        # Calculate dimensions for each layer
        dims = []
        current_dim = output_size
        for _ in range(n_upsample):  # Working backwards from output size
            current_dim = (current_dim + 2*1 - 4) // 2 + 1
        
        # Initial dimension after reshape from latent space
        self.initial_dim = current_dim
        base_channels = 32  # Base number of channels
        channels = [min(base_channels * (2**(n_upsample-1)), 512)] + \
                  [min(base_channels * (2**i), 512) for i in range(n_upsample-2, -1, -1)] + \
                  [output_channels]
        
        # Calculate initial feature map size
        self.initial_size = (channels[0], self.initial_dim, self.initial_dim)
        
        self.fc = nn.Linear(latent_dim, self.initial_size[0] * self.initial_size[1] * self.initial_size[2])
        
        # Build decoder layers dynamically
        layers = []
        for i in range(len(channels)-1):
            layers.extend([
                nn.ConvTranspose2d(channels[i], channels[i+1], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(channels[i+1]) if i < len(channels)-2 else nn.Identity(),  # No BatchNorm in last layer
                nn.ReLU() if i < len(channels)-2 else nn.Tanh()  # Tanh in last layer
            ])
        
        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), *self.initial_size)
        x = self.decoder(x)
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim=32, input_channels=3, image_size=64):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim, input_channels, image_size)
        self.decoder = Decoder(latent_dim, input_channels, image_size)
        self.latent_dim = latent_dim

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.encoder.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    def encode(self, x):
        mu, log_var = self.encoder(x)
        return self.encoder.reparameterize(mu, log_var)

    def decode(self, z):
        return self.decoder(z)

    def interpolate(self, x1, x2, steps=10):
        z1 = self.encode(x1.unsqueeze(0))
        z2 = self.encode(x2.unsqueeze(0))
        alpha = torch.linspace(0, 1, steps)
        z = torch.zeros(steps, self.latent_dim)
        
        for i, a in enumerate(alpha):
            z[i] = a * z2 + (1-a) * z1
            
        return self.decode(z)

    def manipulate_latent(self, x, direction, strength):
        """
        Manipulate the latent representation along a specific direction
        
        Args:
            x (torch.Tensor): Input image
            direction (torch.Tensor): Direction in latent space
            strength (float): Strength of manipulation
        """
        z = self.encode(x.unsqueeze(0))
        z_new = z + direction * strength
        return self.decode(z_new) 