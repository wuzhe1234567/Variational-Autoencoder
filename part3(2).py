import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import matplotlib.pyplot as plt
import umap
import torchvision.utils as vutils

# Data folder path
data_folder = r"C:\Users\wuzhe\Desktop\STUDY\COM3710\Lab2\data\keras_png_slices_seg_train"

# Function to load and preprocess images from a folder
def load_images_from_folder(folder, img_size=(128, 128)):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if img_path.endswith('.png'):
            img = Image.open(img_path).convert('L')  # Convert image to grayscale
            img = img.resize(img_size)  # Resize image to ensure consistency
            img = np.array(img)
            images.append(img)
    return np.array(images)

# Load images from the specified folder
images = load_images_from_folder(data_folder)

# Print information about loaded images
print(f"Number of images loaded: {len(images)}")
if len(images) > 0:
    print(f"Shape of a sample image: {images[0].shape}")
else:
    print("No images were loaded.")

# Handle case when no images are found
if images.size == 0:
    raise ValueError("No images found in the specified folder.")
else:
    # Normalize images to the range [0, 1]
    images = (images - np.min(images)) / (np.max(images) - np.min(images))

# Convert numpy array to PyTorch tensor and add channel dimension
images_tensor = torch.tensor(images[:, np.newaxis, :, :], dtype=torch.float32)

# Create a PyTorch dataset and data loader
dataset = TensorDataset(images_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define a simple Variational Autoencoder (VAE) model
class SimpleVAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(SimpleVAE, self).__init__()
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Flatten(),  # Flatten the input image
            nn.Linear(128 * 128, 512),  # Input layer
            nn.ReLU(),
            nn.Linear(512, latent_dim * 2)  # Output layer with 2*latent_dim to get mu and logvar
        )
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128 * 128),
            nn.Sigmoid()  # Ensure output is in the range [0, 1]
        )

    def encode(self, x):
        x = self.encoder(x)
        mu, logvar = x.chunk(2, dim=-1)  # Split into mean (mu) and log variance (logvar)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # Calculate standard deviation
        eps = torch.randn_like(std)  # Generate random noise
        return mu + eps * std  # Reparameterization trick

    def decode(self, z):
        return self.decoder(z).view(-1, 1, 128, 128)  # Reshape output to image dimensions

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Define the loss function for VAE
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')  # Reconstruction loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL Divergence loss
    return BCE + KLD

# Initialize the VAE model and optimizer
vae = SimpleVAE().cuda()
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    vae.train()
    total_loss = 0
    for batch in dataloader:
        batch = batch[0].cuda()  # Move data to GPU
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(batch)  # Forward pass
        loss = loss_function(recon_batch, batch, mu, logvar)  # Compute loss
        loss.backward()  # Backward pass
        total_loss += loss.item()
        optimizer.step()  # Update model parameters

    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader.dataset)}')

# Visualize the latent space using UMAP
vae.eval()
latent_vectors = []
for batch in dataloader:
    batch = batch[0].cuda()
    mu, logvar = vae.encode(batch)
    z = vae.reparameterize(mu, logvar)
    latent_vectors.append(z.cpu().detach().numpy())

latent_vectors = np.concatenate(latent_vectors)
umap_embedding = umap.UMAP(n_neighbors=15, min_dist=0.1).fit_transform(latent_vectors)

# Scatter plot of UMAP embedding
plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=umap_embedding[:, 0], cmap='Spectral')
plt.colorbar()  # Add color bar to show the value scale
plt.title('UMAP Visualization of VAE Latent Space')
plt.show()

# Sample from the latent space and generate images
z = torch.randn(64, 20).cuda()  # Generate random latent vectors
with torch.no_grad():
    sampled_images = vae.decode(z).cpu()  # Generate images from latent vectors

# Display generated images
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Generated MRI Brain Images")
plt.imshow(np.transpose(vutils.make_grid(sampled_images, padding=2, normalize=True), (1, 2, 0)))
plt.show()
