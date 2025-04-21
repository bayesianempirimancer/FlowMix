from flax import linen as nn
import jax
import jax.numpy as jnp
import optax
import numpy as np
import yaml
from src.utils.data_utils import load_data
from models import VAE

class FlowMix(nn.Module):
    encoder: nn.Module
    decoder: nn.Module

    def setup(self):
        self.encoder_model = self.encoder()
        self.decoder_model = self.decoder()

    def __call__(self, y, h):
        # Encode the input data
        z_mean, z_log_var, x_emb = self.encoder_model(y, h)
        # Sample from the latent space
        z = self.reparameterize(z_mean, z_log_var)
        # Decode the latent representation
        reconstructed_points = self.decoder_model(z)
        return reconstructed_points, z_mean, z_log_var

    def reparameterize(self, mean, log_var):
        eps = jax.random.normal(jax.random.PRNGKey(0), mean.shape)
        return mean + jnp.exp(0.5 * log_var) * eps

    def compute_loss(self, reconstructed, original, z_mean, z_log_var):
        reconstruction_loss = jnp.mean((reconstructed - original) ** 2)
        kl_divergence = -0.5 * jnp.mean(1 + z_log_var - jnp.square(z_mean) - jnp.exp(z_log_var))
        return reconstruction_loss + kl_divergence

def train_vae(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load data
    train_data, val_data = load_data(config['data'])

    # Initialize model
    encoder = ...  # Import and initialize your encoder model
    decoder = ...  # Import and initialize your decoder model
    vae = VAE(encoder=encoder, decoder=decoder)

    # Initialize optimizer
    optimizer = optax.adam(config['optimizer']['learning_rate'])
    opt_state = optimizer.init(vae.parameters)

    # Training loop
    for epoch in range(config['training']['epochs']):
        for batch in train_data:
            y, h = batch['y'], batch['h']
            with jax.profiler.TraceContext('train_step'):
                reconstructed, z_mean, z_log_var = vae(y, h)
                loss = vae.compute_loss(reconstructed, y, z_mean, z_log_var)

            # Update parameters
            grads = jax.grad(loss)(vae.parameters)
            updates, opt_state = optimizer.update(grads, opt_state)
            vae.parameters = optax.apply_updates(vae.parameters, updates)

        # Log training progress
        print(f'Epoch {epoch + 1}/{config["training"]["epochs"]}, Loss: {loss}')

if __name__ == "__main__":
    train_vae('config/default.yaml')