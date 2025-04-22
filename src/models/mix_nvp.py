import jax.numpy as jnp
import jax.random as jr
from jax import lax
from jax.scipy.special import digamma
from flax import linen as nn
from src.layers import ParallelRealNVP

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
        
self = MixRealNVP(4, 3, 9, (5,5))
x = jnp.ones((2, 10, 1, 3))
params = self.init(jr.PRNGKey(2), x)
y, logp = self.apply(params, x)
xhat = self.apply(params, y, method=self.inverse)