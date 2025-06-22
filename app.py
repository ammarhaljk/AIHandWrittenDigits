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
@st.cache_resource
def load_model():
    model = VAE()
    try:
        model.load_state_dict(torch.load("mnist_vae_final.pth", map_location=device))
        model.to(device)
        model.eval()
        return model, True
    except FileNotFoundError:
        st.error("Model file 'mnist_vae_final.pth' not found. Please ensure the model file is in the same directory.")
        return None, False

model, model_loaded = load_model()

# Streamlit UI
st.title("MNIST Digit Generator using VAE")

# Add warning about unconditional VAE
st.warning("""
⚠️ **Important Note**: This VAE is unconditional, meaning it generates random digits from the learned distribution. 
It cannot generate specific digits on command. For digit-specific generation, you would need a Conditional VAE (CVAE) 
trained with digit labels.
""")

if model_loaded:
    # User input for digit selection
    selected_digit = st.selectbox(
        "Choose a digit to attempt to generate (Note: Results are random due to unconditional VAE)",
        options=list(range(10)),
        index=0
    )
    
    # Fixed number of samples
    num_samples = 5
    st.write(f"Generating {num_samples} samples (results will be random, not necessarily digit {selected_digit})")
    
    # Add seed option for reproducibility
    use_seed = st.checkbox("Use seed for reproducible results")
    if use_seed:
        seed_value = st.number_input("Seed value", value=42, min_value=0, max_value=9999)
    
    if st.button("Generate Samples"):
        with torch.no_grad():
            # Set seed if requested
            if use_seed:
                torch.manual_seed(seed_value)
                np.random.seed(seed_value)
            
            # Generate random latent vectors
            z = torch.randn(num_samples, 20).to(device)
            generated = model.decode(z).cpu().view(-1, 28, 28).numpy()
            
            # Create plot
            fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2.5, 3))
            if num_samples == 1:
                axes = [axes]
            
            for i in range(num_samples):
                axes[i].imshow(generated[i], cmap="gray")
                axes[i].set_title(f"Sample {i+1}", fontsize=12)
                axes[i].axis("off")
            
            plt.suptitle(f"Generated Samples (Requested digit: {selected_digit})", fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Clear the plot to prevent memory issues
            plt.close(fig)
    
    # Additional controls
    st.markdown("---")
    st.subheader("Multiple Generation Attempts")
    
    if st.button("Generate 3 Sets of Samples"):
        with torch.no_grad():
            fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 2, 6))
            
            for set_idx in range(3):
                z = torch.randn(num_samples, 20).to(device)
                generated = model.decode(z).cpu().view(-1, 28, 28).numpy()
                
                for i in range(num_samples):
                    axes[set_idx, i].imshow(generated[i], cmap="gray")
                    axes[set_idx, i].set_title(f"Set {set_idx+1}, Sample {i+1}", fontsize=10)
                    axes[set_idx, i].axis("off")
            
            plt.suptitle(f"Multiple Generation Attempts (Requested digit: {selected_digit})", fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    
    # Information section
    st.markdown("---")
    st.subheader("About this VAE")
    st.info("""
    **Current Model**: Unconditional Variational Autoencoder
    - **Latent Dimension**: 20
    - **Architecture**: Encoder (784→512→256→128→20) + Decoder (20→128→256→512→784)
    - **Limitation**: Cannot generate specific digits on command
    
    **For Conditional Generation**: You would need a Conditional VAE (CVAE) that includes digit labels during training.
    """)
    
else:
    st.error("Cannot load the VAE model. Please check if 'mnist_vae_final.pth' exists in the current directory.")
    st.info("If you don't have a trained model, you need to train a VAE on MNIST first.")
