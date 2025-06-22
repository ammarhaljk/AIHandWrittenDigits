"""
MNIST Handwritten Digit Generation using Variational Autoencoder (VAE)
Training Script for Google Colab with T4 GPU

This script trains a VAE from scratch on the MNIST dataset to generate handwritten digits.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 5
LATENT_DIM = 20
INPUT_DIM = 28 * 28

class VAE(nn.Module):
    """
    Variational Autoencoder for MNIST digit generation
    """
    def __init__(self, input_dim=784, latent_dim=20):
        super(VAE, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Latent space - mean and log variance
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()  # Output values between 0 and 1
        )
        
    def encode(self, x):
        """Encode input to latent space parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for backpropagation through sampling"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode from latent space to output"""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through the entire VAE"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss function combining reconstruction loss and KL divergence
    """
    # Reconstruction loss (Binary Cross Entropy)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss

def train_vae():
    """Main training function"""
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to 784 dimensions
    ])
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model
    model = VAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    model.train()
    train_losses = []
    
    print("Starting VAE training...")
    for epoch in range(EPOCHS):
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        
        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            
            # Calculate loss
            loss = vae_loss(recon_batch, data, mu, logvar)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'Loss': loss.item() / len(data)})
        
        avg_loss = epoch_loss / len(train_dataset)
        train_losses.append(avg_loss)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
        
        # Save model checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'vae_checkpoint_epoch_{epoch+1}.pth')
    
    # Save final model
    torch.save(model.state_dict(), 'mnist_vae_final.pth')
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('VAE Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    return model

def generate_digits(model, digit_class, num_samples=5):
    """
    Generate handwritten digits using the trained VAE
    Note: This is a basic generation approach. For conditional generation,
    you would need a conditional VAE (CVAE) which includes class labels.
    """
    model.eval()
    with torch.no_grad():
        # Sample from latent space
        z = torch.randn(num_samples, LATENT_DIM).to(device)
        
        # Generate images
        generated = model.decode(z)
        generated = generated.view(num_samples, 28, 28)
        
        return generated.cpu().numpy()

def visualize_generations(model, num_samples=10):
    """Visualize generated samples"""
    generated_images = generate_digits(model, 0, num_samples)  # Generate any digit
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < len(generated_images):
            ax.imshow(generated_images[i], cmap='gray')
            ax.set_title(f'Generated {i+1}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Training execution
if __name__ == "__main__":
    # Train the model
    trained_model = train_vae()
    
    # Generate and visualize some samples
    print("Generating sample digits...")
    visualize_generations(trained_model)
    
    print("Training completed! Model saved as 'mnist_vae_final.pth'")
    print("You can now use this model in your web application.")

"""
Notes for improvement:
1. This is a basic VAE that generates random digits. For digit-specific generation,
   you would need to implement a Conditional VAE (CVAE) that takes class labels as input.
2. The model architecture can be enhanced with convolutional layers for better image quality.
3. Training time on T4 GPU should be around 15-20 minutes for 50 epochs.
4. Adjust hyperparameters (learning rate, latent dimension, architecture) based on results.
"""