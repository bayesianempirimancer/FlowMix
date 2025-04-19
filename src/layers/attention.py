from flax import linen as nn
import jax.numpy as jnp

class SelfAttention(nn.Module):
    num_heads: int
    head_dim: int

    def setup(self):
        self.query_dense = nn.Dense(self.num_heads * self.head_dim)
        self.key_dense = nn.Dense(self.num_heads * self.head_dim)
        self.value_dense = nn.Dense(self.num_heads * self.head_dim)
        self.output_dense = nn.Dense(self.num_heads * self.head_dim)

    def __call__(self, x):
        batch_size, seq_length, _ = x.shape

        # Linear transformations
        queries = self.query_dense(x).reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        keys = self.key_dense(x).reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        values = self.value_dense(x).reshape(batch_size, seq_length, self.num_heads, self.head_dim)

        # Scaled dot-product attention
        scores = jnp.einsum('bqhd,bkhd->bhqk', queries, keys) / jnp.sqrt(self.head_dim)
        attention_weights = nn.softmax(scores, axis=-1)

        # Weighted sum of values
        context = jnp.einsum('bhqk,bkhd->bqhd', attention_weights, values)

        # Concatenate heads and pass through output dense layer
        context = context.reshape(batch_size, seq_length, self.num_heads * self.head_dim)
        return self.output_dense(context)


class CrossAttention(nn.Module):
    num_heads: int
    head_dim: int

    @nn.compact  # Use compact to simplify setup
    def __call__(self, x, y):
        batch_size_x, seq_length_x, feature_dim_x = x.shape
        batch_size_y, seq_length_y, feature_dim_y = y.shape

        # Linear transformations
        query_dense = nn.Dense(self.num_heads * self.head_dim, name="query_dense")
        key_dense = nn.Dense(self.num_heads * self.head_dim, name="key_dense")
        value_dense = nn.Dense(self.num_heads * self.head_dim, name="value_dense")
        output_dense = nn.Dense(feature_dim_x, name="output_dense")

        queries = query_dense(x).reshape(batch_size_x, seq_length_x, self.num_heads, self.head_dim)
        keys = key_dense(y).reshape(batch_size_y, seq_length_y, self.num_heads, self.head_dim)
        values = value_dense(y).reshape(batch_size_y, seq_length_y, self.num_heads, self.head_dim)

        # Scaled dot-product attention
        scores = jnp.einsum('bqhd,bkhd->bhqk', queries, keys) / jnp.sqrt(self.head_dim)
        attention_weights = nn.softmax(scores, axis=-1)

        # Weighted sum of values
        context = jnp.einsum('bhqk,bkhd->bqhd', attention_weights, values)

        # Concatenate heads and pass through output dense layer
        context = context.reshape(batch_size_x, seq_length_x, self.num_heads * self.head_dim)
        return x + output_dense(context)  # Residual connection and update x
    
# import jax.random as jr
# key = jr.PRNGKey(0)
# num_heads = 8
# head_dim = 12
# batch_size = 4
# seq_length_x = 4
# seq_length_y = 24
# feature_dim_x = 64
# feature_dim_y = 12

# # Initialize x and y with random values
# key_x, key_y = jr.split(key)
# x = jr.normal(key_x, (batch_size, seq_length_x, feature_dim_x))
# y = jr.normal(key_y, (batch_size, seq_length_y, feature_dim_y))

# # Initialize the CrossAttention layer
# cross_attention = CrossAttention(num_heads=num_heads, head_dim=head_dim)

# # Initialize the parameters (variables) of the layer
# key_params = jr.PRNGKey(1)  # Use a different key for parameters
# variables = cross_attention.init(key_params, x, y)

# # Apply the cross-attention layer
# output = cross_attention.apply(variables, x, y)

# print("Output shape:", output.shape)