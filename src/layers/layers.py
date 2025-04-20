import jax
from flax import linen as nn
import jax.numpy as jnp
import jax.random as jr
    
class ParallelDense(nn.Module):
    num_units: int  # Number of units in each dense layer
    num_par: int  # Number of parallel transformations

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
    block_size: int
    features: tuple

    @nn.compact
    def __call__(self, x):        
        for i, feature in enumerate(self.features):
            x = ParallelDense(feature, self.block_size)(x)
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
        mask_dim = len(mask_dims)
        self.update_mask_mat = jnp.eye(dim)[mask_dims]

        # Initialize MLPs for computing s and t
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
        log_det_jac += jnp.sum(s)
                
        x_to_update = x_to_update * (jnp.exp(s)-1.0) + t
        x = jnp.sum(x_to_update[...,:,None]*self.update_mask_mat,-2) + x

        return x, log_det_jac


class ParallelRealNVP(nn.Module):
    num_par: int  # Number of parallel transformations
    mlp_features: tuple  # hidden layer sizes for the MLPs that compute s and t
    masks: int  # Number of nodes in the RealNVP network

    # Note that this assumes that x is of shape (batch_size, num_par, dim) or (batch_size, 1, dim)
    def setup(self):
        num_nodes = self.masks.shape[0]
        nodes = []
        for i in range(num_nodes):
            mask = self.masks[i]
            if mask.sum() == 0:
                raise ValueError(f"Mask {mask} has no dimensions to transform.")
            nodes.append(ParallelRealNVPNode(self.num_par, self.mlp_features, mask))
        self.nodes = nodes
        
    def __call__(self, x):   # assumes x has shape (batch, num_par, dim) or (bathc, 1, dim)
        inputs = (x, 0.0)
        for node in self.nodes:
            inputs = node(inputs)
        
# self = ParallelRealNVPNode(4, (5,5), jnp.array([False, True, False]))
# x = jnp.ones((5, 1, 3))
# params = self.init(jr.PRNGKey(0), (x, 0.0))
# y = self.apply(params, (x, 0.0))


# masks = jnp.concatenate((1-jnp.eye(3), jnp.eye(3)), axis=0)
# masks = jnp.concatenate((masks, masks), axis=0)>0.5

# self = ParallelRealNVP(4, (5,5), masks)
# x = jnp.ones((5, 1, 3))
# params = self.init(jr.PRNGKey(0), x)
# self.apply(params, x)

