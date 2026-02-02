# models/vae_pytorch.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=20, hidden_dim=512, activation=nn.ReLU):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        # Encoder: maps input to latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, latent_dim * 2)  # Outputs mean and log variance
        )
        # Decoder: reconstructs input from latent space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Use Sigmoid to ensure output is between 0 and 1
        )

    def encode(self, x):
        params = self.encoder(x)
        z_mean, z_log_var = params[:, :self.latent_dim], params[:, self.latent_dim:]
        return z_mean, z_log_var

    def reparameterize(self, z_mean, z_log_var):
        # Reparameterization trick to sample from normal distribution
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        return self.decode(z), z_mean, z_log_var


class VAEDataAugmentationPyTorch:
    def __init__(self, input_dim, latent_dim=20, hidden_dim=512, learning_rate=1e-3, kl_weight=1.0, recon_weight=1.0,
                 l1_reg=0.0, l2_reg=0.0, activation=nn.ReLU, early_stopping=True, patience=10,
                 num_interpolation_points=5, device='cpu'):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight
        self.recon_weight = recon_weight
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.activation = activation  # Keep as class
        self.early_stopping = early_stopping
        self.patience = patience
        self.num_interpolation_points = num_interpolation_points
        self.device = device
        self.model = VAE(input_dim, latent_dim, hidden_dim, activation).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def loss_function(self, recon_x, x, z_mean, z_log_var):
        # Reconstruction loss (binary cross-entropy) and KL divergence
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        return self.recon_weight * BCE + self.kl_weight * KLD

    def train_model(self, dataloader, epochs=50):
        self.model.train()
        for epoch in range(epochs):
            train_loss = 0
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(self.device)
                self.optimizer.zero_grad()
                recon_batch, z_mean, z_log_var = self.model(batch_x)
                loss = self.loss_function(recon_batch, batch_x, z_mean, z_log_var)
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()
            print(f'Epoch {epoch + 1}, Loss: {train_loss / len(dataloader.dataset):.4f}')

    def encode(self, x):
        self.model.eval()
        with torch.no_grad():
            z_mean, z_log_var = self.model.encode(x.to(self.device))
            return z_mean.cpu().numpy()

    def decode(self, z):
        self.model.eval()
        with torch.no_grad():
            z = torch.tensor(z, dtype=torch.float32).to(self.device)
            recon_x = self.model.decode(z)
            return recon_x.cpu().numpy()

    def augment_data(self, X, y):
        # Data augmentation based on class-wise interpolation
        unique_classes = np.unique(y)
        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(X)

        augmented_data = []
        augmented_labels = []

        for cls in unique_classes:
            x_class = x_scaled[y == cls]
            dataset = CustomDataset(x_class)
            dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
            self.train_model(dataloader, epochs=50)

            z_mean = self.encode(dataset.data)
            recon_x = self.decode(z_mean)

            # Interpolate between original and reconstructed points
            for original, decoded in zip(x_class, recon_x):
                interpolated_points = linear_interpolate_points(original, decoded, self.num_interpolation_points)
                augmented_data.extend(interpolated_points[1:-1])  # Avoid using the original and decoded points
                augmented_labels.extend([cls] * self.num_interpolation_points)

        augmented_data = np.array(augmented_data)
        augmented_labels = np.array(augmented_labels)

        # Reverse scaling to original data range
        augmented_data_inverse = scaler.inverse_transform(augmented_data)
        return augmented_data_inverse, augmented_labels


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], 0  # Dummy label

def linear_interpolate_points(point_a, point_b, num_points=5):
    # Linearly interpolate between two points
    return np.linspace(point_a, point_b, num=num_points + 2)  # num_points+2 to include both endpoints
