from flax import linen as nn
import jax
import jax.numpy as jnp
import jax.random as jr
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

key = jr.PRNGKey(11)

y = train_data[:10].copy()
y_mask = train_mask[:10].copy()
del train_data, train_mask

y = y + jr.uniform(key, y.shape)*0.5
y = y/jnp.std(y)
y = y - jnp.sum(y, (-3,-2), keepdims=True)/jnp.sum(y_mask, -2, keepdims=True)[...,None]

y = y[:1]
y_mask = y_mask[:1]

# Initialize model
mix_dim = 3
dim = 2
num_nodes = 4
mlp_features = (2,2,2)
mask_seed = 88

model = MixRealNVP(mix_dim, dim, num_nodes, mlp_features, mask_seed)
# Initialize optimizer

key = jax.random.PRNGKey(0)
params = model.init(key, y, y_mask)

optimizer = optax.adam(learning_rate=0.1)
opt_state = optimizer.init(params)

def compute_loss(params, model, y, y_mask):
    _, log_prob = model.apply(params, y, y_mask)
    return -log_prob.sum()  # Negative log-likelihood for minimization

# Training loop
from matplotlib import pyplot as plt
for epoch in range(200):
    # Compute gradients
    loss, grads = jax.value_and_grad(compute_loss)(params, model, y, y_mask)
    
    # Apply updates
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    # Log training progress
    print(f'Epoch {epoch + 1}, Loss: {loss}')

    y_hat = model.apply(params, jax.random.PRNGKey(0), y.shape[:-1], method=model.sample)
    x_hat = model.apply(params, y, y_mask)[0]
    plt.scatter(y_hat[0, :, 0], y_hat[0, :, 1])
    plt.scatter(x_hat[0, :, 0], x_hat[0, :, 1])
    plt.show()
