from flax import linen as nn
import jax
import jax.numpy as jnp
import optax
import numpy as np
import yaml
from src.models import MixRealNVP

# class VAE(nn.Module):
#     encoder: nn.Module
#     decoder: nn.Module

#     def setup(self):
#         self.encoder_model = self.encoder()
#         self.decoder_model = self.decoder()

#     def __call__(self, y, h):
#         # Encode the input data
#         z_mean, z_log_var, x_emb = self.encoder_model(y, h)
#         # Sample from the latent space
#         z = self.reparameterize(z_mean, z_log_var)
#         # Decode the latent representation
#         reconstructed_points = self.decoder_model(z)
#         return reconstructed_points, z_mean, z_log_var

#     def reparameterize(self, mean, log_var):
#         eps = jax.random.normal(jax.random.PRNGKey(0), mean.shape)
#         return mean + jnp.exp(0.5 * log_var) * eps

#     def compute_loss(self, reconstructed, original, z_mean, z_log_var):
#         reconstruction_loss = jnp.mean((reconstructed - original) ** 2)
#         kl_divergence = -0.5 * jnp.mean(1 + z_log_var - jnp.square(z_mean) - jnp.exp(z_log_var))
#         return reconstruction_loss + kl_divergence


# Load data
train_data = jnp.load('datasets/triple_mnist/val_point_cloud.npy')
train_mask = jnp.load('datasets/triple_mnist/val_point_cloud_mask.npy')

# Initialize model
mix_dim = 20
dim = 2
num_nodes = 6
mlp_features = (5,5)
mask_seed = 88

model = MixRealNVP(mix_dim, dim, num_nodes, mlp_features, mask_seed)
# Initialize optimizer

key = jax.random.PRNGKey(0)

y = train_data[:10].copy()
y_mask = train_mask[:10].copy()
del train_data, train_mask

params = model.init(key, y, y_mask)

optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(params)

# Training loop
for epoch in range(1000):
    # Start profiling
    _, log_prob = model(y, y_mask)
    loss = log_prob.sum()
    # Update parameters
    grads = jax.grad(loss)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    # Log training progress
    print(f'Epoch {epoch + 1}, Loss: {loss}')


