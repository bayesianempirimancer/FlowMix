import jax
from flax import linen as nn
import jax.numpy as jnp
import jax.random as jr
    
class ParallelDense(nn.Module):
    num_par: int  # Number of parallel transformations
    num_units: int  # Number of units in each dense layer

    @nn.compact
    def __call__(self, inputs):
        feature_in = inputs.shape[-1]
        # Initialize the weight matrix with shape (block_size, feature_in, num_units)
        kernel = self.param('kernel',
                            jax.nn.initializers.lecun_normal(),
                            (self.num_par, feature_in, self.num_units))

        # Initialize the bias vector with shape (block_size, num_units)
        bias = self.param('bias',
                          jax.nn.initializers.zeros,
                          (self.num_par, self.num_units))

        # Perform the parallel dense transformation
        if inputs.shape[-2] == self.num_par or inputs.shape[-2] == 1:
            return jnp.sum(inputs[...,None] * kernel,-2) + bias
        else:
            raise ValueError(f"Input shape {inputs.shape} does broadcast to {self.block_size}")

class ParallelMLP(nn.Module):
    num_par: int
    features: tuple

    @nn.compact
    def __call__(self, x):        
        for i, feature in enumerate(self.features):
            x = ParallelDense(self.num_par, feature)(x)
            if i != len(self.features) - 1:
                x = nn.relu(x)
        return x

class ParallelRealNVPNode(nn.Module):
    num_par: int  # Number of parallel transformations
    mlp_features: tuple
    mask: jnp.ndarray # Mask = True determines the dimensions that are changed by the transform

    def setup(self):
        # Create a mask matrix for the parallel transformations
        dim = self.mask.shape[-1]
        mask_dims = jnp.where(~self.mask)
        self.static_mask_mat = jnp.eye(dim)[mask_dims]

        mask_dims = jnp.where(self.mask)
        mask_dim = len(mask_dims[0])
        self.update_mask_mat = jnp.eye(dim)[mask_dims]

        self.s_mlp = ParallelMLP(self.num_par, self.mlp_features + (mask_dim,))
        self.t_mlp = ParallelMLP(self.num_par, self.mlp_features + (mask_dim,))

    @nn.compact
    def __call__(self, inputs):
        x, log_det_jac = inputs

        if not (x.shape[-2] == self.num_par or x.shape[-2] == 1):
            raise ValueError(f"Input/output shape {x.shape} does broadcast to (..., {self.num_par}, :)")
        
        x_static = jnp.sum(x[...,None,:]*self.static_mask_mat,-1)
        x_to_update = jnp.sum(x[...,None,:]*self.update_mask_mat, -1)

        s = self.s_mlp(x_static)
        t = self.t_mlp(x_static)
        log_det_jac += jnp.sum(s,-1)

        x_to_update = x_to_update * (jnp.exp(s)-1.0) + t
        x = jnp.sum(x_to_update[...,:,None]*self.update_mask_mat,-2) + x

        return x, log_det_jac
    
    def inverse(self, inputs):
        x, log_det_jac = inputs

        # Inverse transformation
        x_static = jnp.sum(x[..., None, :] * self.static_mask_mat, -1)
        x_to_update = jnp.sum(x[..., None, :] * self.update_mask_mat, -1)

        s = self.s_mlp(x_static)
        t = self.t_mlp(x_static)
        log_det_jac -= jnp.sum(s, -1)

        x_to_update = (x_to_update - t) * jnp.exp(-s)
        x = jnp.sum(x_to_update[..., :, None] * self.update_mask_mat, -2) + x

        return x, log_det_jac

import numpyro.distributions as dists

class ParallelRealNVP(nn.Module):
    num_par: int  # Number of parallel transformations
    dim: int  # Number of dimensions
    num_nodes: int  # Number of NVP nodes
    mlp_features: tuple  # hidden layer sizes for the MLPs that compute s and t
    mask_seed: int = 88

#    masks: jnp.ndarray  # masks.shape[-2] = number of NVP nodes, 
#                        # masks.shape[-1] = number of dimensions

    # Note that this assumes that x is of shape (batch_size, num_par, dim) or (batch_size, 1, dim)
    def setup(self):        
        key = jr.PRNGKey(self.mask_seed)
        nodes = []
        for i in range(self.num_nodes):
            key, subkey1, subkey2 = jr.split(key,3)
            indices = jr.permutation(subkey2, jnp.arange(self.dim))[:self.dim//2 + jr.bernoulli(subkey1, 0.5)]
            mask = jnp.zeros(self.dim, dtype=bool)
            for j in range(len(indices)):
                mask = mask.at[indices[j]].set(True)    

            nodes.append(ParallelRealNVPNode(self.num_par, self.mlp_features, mask))

        self.nodes = nodes
        self.prior = dists.MultivariateNormal(jnp.zeros(self.dim), jnp.eye(self.dim))
        
    def __call__(self, x):   # assumes x has shape (batch, num_par, dim) or (bathc, 1, dim)
        inputs = (x, 0.0)
        for node in self.nodes:
            inputs = node(inputs)

        y, log_det_jac = inputs
        return y, log_det_jac + self.prior.log_prob(y)
    
    def inverse(self, x):
        # Inverse transformation
        log_det_jac = self.prior.log_prob(x)
        inputs = (x, log_det_jac)
        for node in reversed(self.nodes):
            inputs = node.inverse(inputs)

        return inputs        

class MixRealNVP(nn.Module):
    mix_dim: int #Also equal to numbber of parallel transformations in Parallel* stuff
    dim: int  # Number of dimensions
    num_nodes: int  # Number of NVP nodes
    mlp_features: tuple  # hidden layer sizes for the MLPs that compute s and t
    mask_seed: int = 88

    def setup(self):
        self.dists = ParallelRealNVP(self.mix_dim, self.dim, self.num_nodes, self.mlp_features, self.mask_seed)
        self.mixing_log_probs = self.param("mixing_probs",  
            jax.nn.initializers.normal(stddev=0.1),  (self.mix_dim,))
        
    def __call__(self, x):   # assumes x has shape (batch, num_par, dim) or (bathc, 1, dim)
        y, log_prob = self.dists(x)
        log_prob += self.mixing_log_probs

        log_prob = log_prob - jnp.max(log_prob, axis=-1, keepdims=True)
        return y, jnp.log(jnp.sum(jnp.exp(log_prob), axis=-1)) 


self = ParallelRealNVPNode(4, (5,5), jnp.array([False, True, False]))
x = jnp.ones((5, 1, 3))
params = self.init(jr.PRNGKey(0), (x, 0.0))
y, logp = self.apply(params, (x, 0.0))

self = ParallelRealNVP(4, 3, 10, (5,5))
x = jnp.ones((5, 1, 3))
params = self.init(jr.PRNGKey(1), x)
y, logp = self.apply(params, x)

self = MixRealNVP(4, 3, 10, (5,5))
x = jnp.ones((5, 1, 3))
params = self.init(jr.PRNGKey(2), x)
y, logp = self.apply(params, x)

