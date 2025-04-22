from jax import lax
from flax import linen as nn
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import digamma
    
class ParallelDense(nn.Module):
    num_par: int  # Number of parallel transformations
    num_units: int  # Number of units in each dense layer

    @nn.compact
    def __call__(self, inputs):
        feature_in = inputs.shape[-1]
        # Initialize the weight matrix with shape (block_size, feature_in, num_units)
        kernel = self.param('kernel',
                            nn.initializers.lecun_normal(),
                            (self.num_par, feature_in, self.num_units))

        # Initialize the bias vector with shape (block_size, num_units)
        bias = self.param('bias',
                          nn.initializers.zeros,
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

        mask_dim = jnp.sum(self.mask, -1)
        self.s_mlp = ParallelMLP(self.num_par, self.mlp_features + (mask_dim,))
        self.t_mlp = ParallelMLP(self.num_par, self.mlp_features + (mask_dim,))

    def __call__(self, inputs):
        x, log_det_jac = inputs
        x = jnp.broadcast_to(x, x.shape[:-2] + (self.num_par,) + x.shape[-1:])  # Broadcast to num_par

        if not (x.shape[-2] == self.num_par or x.shape[-2] == 1):
            raise ValueError(f"Input/output shape {x.shape} does broadcast to (..., {self.num_par}, :)")
        
        x_static = x[..., ~self.mask]  # Dimensions where mask is False
        x_to_update = x[..., self.mask]  # Dimensions where mask is True

        s = self.s_mlp(x_static)
        t = self.t_mlp(x_static)
        log_det_jac += jnp.sum(s,-1)

        x_to_update = x_to_update * jnp.exp(s) + t

        # Update x recalling that the mask == True determines the dimensions that are changed by the transform
        x = x.at[..., self.mask].set(x_to_update)
        x = x.at[..., ~self.mask].set(x_static)

        return x, log_det_jac

    @nn.compact
    def inverse(self, x):

        # Inverse transformation
        x_static = x[..., ~self.mask]  # Dimensions where mask is False
        x_to_update = x[..., self.mask]  # Dimensions where mask is True

        s = self.s_mlp(x_static)
        t = self.t_mlp(x_static)

        x_to_update = (x_to_update - t) * jnp.exp(-s)

        # Update x recalling that the mask == True determines the dimensions that are changed by the transform
        x = x.at[..., self.mask].set(x_to_update)
        x = x.at[..., ~self.mask].set(x_static)

        return x

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
    
    @nn.compact
    def inverse(self, x):
        # Inverse transformation
        for node in reversed(self.nodes):
            x = node.inverse(x)
        return x        

def vb_mix_iter(carry, log_prob):  # assumes that log_prob.shape = (sample, batch, mix_dim)
    E_log_pb = carry
    mix_dim = log_prob.shape[-1]
    logpz = log_prob + E_log_pb
    logpz = logpz - jnp.max(logpz, axis=-1, keepdims=True)
    N = nn.softmax(logpz, axis=-1).sum(-2, keepdims=True)
    alpha_0 = 0.5

    E_log_pb = digamma(alpha_0 + N) - digamma(mix_dim*alpha_0 + jnp.sum(N,-1, keepdims=True))  
               # shape = (sample, mix_dim)                                  
    return E_log_pb, None

class MixRealNVP(nn.Module):
    mix_dim: int #Also equal to numbber of parallel transformations in Parallel* stuff
    dim: int  # Number of dimensions
    num_nodes: int  # Number of NVP nodes
    mlp_features: tuple  # hidden layer sizes for the MLPs that compute s and t
    mask_seed: int = 88

    def setup(self):
        self.dists = ParallelRealNVP(self.mix_dim, self.dim, self.num_nodes, self.mlp_features, self.mask_seed)
        
    def __call__(self, x):   # assumes x has shape (batch, sample, dim) or (batch, 1, dim)
        y, log_prob = self.dists(x)  # log_prob.shape = (batch, sample, mix_dim)
        E_log_pb = jnp.zeros(log_prob.shape[:-2] + (1,) + log_prob.shape[-1:]) # shape = batch, mix_dim

        def scan_fn(carry, _):
            return vb_mix_iter(carry, log_prob)
        E_log_pb, _ = lax.scan(scan_fn, E_log_pb, None, length=5)

        # for i in range(5):
        #     E_log_pb, _ = vb_iter(E_log_pb, log_prob)

        log_prob += E_log_pb
        log_prob = log_prob - jnp.max(log_prob, axis=-1, keepdims=True)
        return y, jnp.log(jnp.sum(jnp.exp(log_prob), axis=-1)) 

    @nn.compact
    def inverse(self, x):
        # Inverse transformation
        return self.dists.inverse(x)  
        
    @nn.compact
    def sample(self, key, shape):
        key, subkey1, subkey2 = jr.split(key,3)
        z = jr.categorical(key = subkey1, logits=self.mixing_log_probs, shape = shape)
        z = nn.one_hot(z, self.mix_dim)
        x = jr.normal(subkey2, shape + (self.mix_dim, self.dim))

        return jnp.sum(self.inverse(x)*z[...,None],-2)
        
# self = ParallelRealNVPNode(4, (5,5), jnp.array([False, True, False]))
# x = jnp.ones((5, 4, 3))
# params = self.init(jr.PRNGKey(0), (x, 0.0))
# y, logp = self.apply(params, (x, 0.0))
# xhat = self.apply(params, y, method=self.inverse)

# self = ParallelRealNVP(4, 3, 9, (5,5))
# x = jnp.ones((5, 1, 3))
# params = self.init(jr.PRNGKey(1), x)
# y, logp = self.apply(params, x)
# xhat = self.apply(params, y, method=self.inverse)

self = MixRealNVP(4, 3, 9, (5,5))
x = jnp.ones((2, 10, 1, 3))
params = self.init(jr.PRNGKey(2), x)
y, logp = self.apply(params, x)
xhat = self.apply(params, y, method=self.inverse)
