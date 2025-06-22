# app.py

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define VAE model class (same as in training)
class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, input_dim), nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Load model
model = VAE()
model.load_state_dict(torch.load("mnist_vae_final.pth", map_location=device))
model.to(device)
model.eval()

# Streamlit UI
st.title("MNIST Digit Generator using VAE")
num_samples = st.slider("Number of Digits to Generate", 1, 10, 5)

if st.button("Generate"):
    with torch.no_grad():
        z = torch.randn(num_samples, 20).to(device)
        generated = model.decode(z).cpu().view(-1, 28, 28).numpy()

        fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))
        if num_samples == 1:
            axes = [axes]
        for i in range(num_samples):
            axes[i].imshow(generated[i], cmap="gray")
            axes[i].axis("off")
        st.pyplot(fig)
