from flax import linen as nn
import jax.numpy as jnp

class VAE(nn.Module):
    encoder: nn.Module
    decoder: nn.Module
    latent_dim: int

    def setup(self):
        self.encoder_model = self.encoder()
        self.decoder_model = self.decoder()

    def encode(self, y, h):
        # Encode the input data to obtain the latent representation
        z_mean, z_log_var, x = self.encoder_model(y, h)
        return z_mean, z_log_var, x

    def reparameterize(self, z_mean, z_log_var):
        # Reparameterization trick
        epsilon = jnp.random.normal(shape=z_mean.shape)
        return z_mean + jnp.exp(0.5 * z_log_var) * epsilon

    def decode(self, z):
        # Decode the latent representation back to the original space
        return self.decoder_model(z)

    def loss_function(self, y, h, reconstructed, z_mean, z_log_var):
        # Compute the reconstruction loss and KL divergence
        reconstruction_loss = jnp.mean((y - reconstructed) ** 2)
        kl_divergence = -0.5 * jnp.mean(1 + z_log_var - jnp.square(z_mean) - jnp.exp(z_log_var))
        return reconstruction_loss + kl_divergence

    def __call__(self, y, h):
        z_mean, z_log_var, x = self.encode(y, h)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decode(z)
        loss = self.loss_function(y, h, reconstructed, z_mean, z_log_var)
        return reconstructed, loss, z_mean, z_log_var