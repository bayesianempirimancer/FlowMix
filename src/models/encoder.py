from flax import linen as nn
import jax.numpy as jnp
from ..layers.attention import SelfAttention, CrossAttention
from ..layers.layers import DistanceBasedPooling


    
class Encoder(nn.Module):
    num_layers: int
    num_heads: int
    head_dim: int

    @nn.compact
    def __call__(self, x, y):
        for _ in range(self.num_layers):
            x = CrossAttention(num_heads=self.num_heads, head_dim=self.head_dim)(x,y)
        return x
    
