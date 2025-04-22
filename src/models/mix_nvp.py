import jax.numpy as jnp
import jax.random as jr
from jax import lax
from jax.scipy.special import digamma
from flax import linen as nn
from src.layers import ParallelRealNVP

def vbe_mix_iter(E_log_pb, log_prob, x_mask):  # assumes that log_prob.shape = (sample, batch, mix_dim)
                                              # and x_mask.shape = (sample, batch)
    mix_dim = log_prob.shape[-1]
    logpz = log_prob + E_log_pb
    logpz = logpz - jnp.max(logpz, axis=-1, keepdims=True)
    if x_mask is not None:
        N = (nn.softmax(logpz, axis=-1)*x_mask[...,None]).sum(-2, keepdims=True)
    else: 
        N = (nn.softmax(logpz, axis=-1)).sum(-2, keepdims=True)

    alpha_0 = 0.5
    E_log_pb = digamma(alpha_0 + N) - digamma(mix_dim*alpha_0 + jnp.sum(N,-1, keepdims=True))  
               # shape = (sample, mix_dim)                                  
    return E_log_pb

def scan_fn(carry, _):
    E_log_pb, log_prob, x_mask = carry
    E_log_pb = vbe_mix_iter(E_log_pb, log_prob, x_mask)
    return (E_log_pb, log_prob, x_mask), None

class MixRealNVP(nn.Module):
    mix_dim: int #Also equal to numbber of parallel transformations in Parallel* stuff
    dim: int  # Number of dimensions
    num_nodes: int  # Number of NVP nodes
    mlp_features: tuple  # hidden layer sizes for the MLPs that compute s and t
    mask_seed: int = 88

    '''Input x is expected to be of shape (batch, sample, 1, dim) where sample refers to the 
       number of images in the batch.  The output y has shape (batch, sample) and can be turned
       into a loss function by simply summing over the batch and sample dimensions.'''

    def setup(self):
        self.dists = ParallelRealNVP(self.mix_dim, self.dim, self.num_nodes, self.mlp_features, self.mask_seed)
        
    def __call__(self, x, x_mask=None):   # assumes x has shape (batch, sample, dim) 
        # Forward transformation     # and x_mask has shape (batch, sample)
        y, log_prob = self.dists(x[...,None,:])  # log_prob.shape = (batch, sample, mix_dim)
        E_log_pb = jnp.zeros(log_prob.shape[:-2] + (1,) + log_prob.shape[-1:]) # shape = batch, mix_dim

        carry = (E_log_pb, log_prob, x_mask)
        carry, _ = lax.scan(scan_fn, carry, None, length=5)
        E_log_pb, log_prob, x_mask = carry

        # for i in range(5):
        #     E_log_pb, _ = vb_iter(E_log_pb, log_prob)

        log_prob += E_log_pb
        log_prob = log_prob - jnp.max(log_prob, axis=-1, keepdims=True)
        return y, jnp.log(jnp.sum(jnp.exp(log_prob), axis=-1)) 

    @nn.compact
    def inverse(self, y):
        # Inverse transformation
        return self.dists.inverse(y)  
        
    @nn.compact
    def sample(self, key, shape):
        key, subkey1, subkey2 = jr.split(key,3)
        z = jr.categorical(key = subkey1, logits=jnp.zeros(self.mix_dim), shape = shape)
        z = nn.one_hot(z, self.mix_dim)
        x = jr.normal(subkey2, shape + (self.mix_dim, self.dim))

        return jnp.sum(self.inverse(x)*z[...,None],-2)

    @staticmethod
    def loss(log_prob):
        return -jnp.sum(log_prob)
        
self = MixRealNVP(4, 2, 9, (5,5))
x = jnp.ones((2, 10, 2))
x_mask = jnp.ones(x.shape[:-1], dtype=bool)
params = self.init(jr.PRNGKey(2), x, x_mask)
y, logp = self.apply(params, x, x_mask)
xhat = self.apply(params, y, method=self.inverse)
y_hat = self.apply(params, jr.PRNGKey(0), x.shape[:-1], method=self.sample)

# self.loss(self.apply(params, x, x_mask)[1])