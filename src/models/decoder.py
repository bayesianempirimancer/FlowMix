from flax import linen as nn
import jax.numpy as jnp

def softmax_with_threshold(logits, threshold=0.1):
    """
    Applies softmax to the logits and thresholds the output.
    
    Args:
        logits: Input logits.
        threshold: Threshold for softmax output.
        
    Returns:
        Thresholded softmax output.
    """
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)  # Numerical stability
    logits = jnp.where(logits > jnp.log(threshold), logits, -jnp.inf)  # Apply threshold
    return nn.softmax(logits)

def quat_to_rot_mat(quat):
    """
    Converts a quaternion to a rotation matrix.
    
    Args:
        quat: Quaternion (Batch x N x 4D vector).

    Returns:
        3x3 rotation matrix.
    """
    w = quat[..., 0]
    x = quat[..., 1]
    y = quat[..., 2]
    z = quat[..., 3]
    
    scale = w**2 + x**2 + y**2 + z**2
    
    # Compute rotation matrix elements
    r00 = scale - 2*(y**2 + z**2)
    r01 = 2*(x*y - w*z)
    r02 = 2*(x*z + w*y)
    r10 = 2*(x*y + w*z)
    r11 = scale - 2*(x**2 + z**2)
    r12 = 2*(y*z - w*x)
    r20 = 2*(x*z - w*y)
    r21 = 2*(y*z + w*x)
    r22 = scale - 2*(x**2 + y**2)

    # Stack the elements to form the rotation matrix
    scaled_rot_mat = jnp.stack([
        jnp.stack([r00, r01, r02], axis=-1),
        jnp.stack([r10, r11, r12], axis=-1),
        jnp.stack([r20, r21, r22], axis=-1)
    ], axis=-2)
    
    return scaled_rot_mat, scale  # Note that scale has shape = batch_shape 
                                  # scaled_rotation_matrix has shape = batch_shape + (3, 3)

class Decoder(nn.Module):
    num_objects: int
    embedding_dim: int
    num_classes: int

    def setup(self):
        self.T = nn.Dense(self.num_classes)  # Object type distribution
        self.L = nn.Dense(3)  # Object location
        self.sR = nn.Dense(4)  # Quaternion (4 for orientation)

        self.shape_flows = [None]*self.num_objects  # Placeholder for normalizing flows

    def __call__(self, x, y):
        # Compute object type distribution
        t = nn.softmax(self.T(x))  # batch x num_objects x num_classes
        # Compute object position, rotation, and scale
        pos = self.L(x)   # batch x num_objects x 3
        scale, scaled_rotmat = quat_to_rot_mat(self.sR(x)) # batch x num_objects and batch x num_objects x 3 x 3

        return t, pos, scaled_rotmat, scale, self.shape_flows

class Loss(nn.Module):
    def __call__(self, y, py):

        logp_given_t = jnp.sum(py.t*self.shape_flows.logp((y[...,None,:,:] - py.pos)@py.scaled_rotmat),-1)


        loss = None
        return loss


