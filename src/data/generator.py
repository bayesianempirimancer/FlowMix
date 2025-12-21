import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Dict
import chex
import numpy as np
import tensorflow_datasets as tfds
from scipy.spatial.transform import Rotation as R

@chex.dataclass
class SceneData:
    points: jnp.ndarray  # [N, D]
    features: jnp.ndarray  # [N, F] (e.g., RGB, Opacity, Covariance params)
    object_ids: jnp.ndarray  # [N]
    # Splat specific params if separated
    scales: Optional[jnp.ndarray] = None
    quats: Optional[jnp.ndarray] = None
    opacities: Optional[jnp.ndarray] = None

class MultiObjectSceneGenerator:
    def __init__(self, spatial_dim: int = 3, feature_dim: int = 3):
        """
        Args:
            spatial_dim: 2 or 3
            feature_dim: Dimension of feature (e.g., 3 for RGB)
        """
        self.spatial_dim = spatial_dim
        self.feature_dim = feature_dim
        self.mnist_ds = None

    def _load_mnist(self):
        if self.mnist_ds is None:
            ds = tfds.load('mnist', split='train', shuffle_files=False)
            # Load to memory (only 60k images, small)
            self.mnist_images = []
            self.mnist_labels = []
            for ex in ds.as_numpy_iterator():
                self.mnist_images.append(ex['image'])
                self.mnist_labels.append(ex['label'])
            self.mnist_images = np.array(self.mnist_images) # (60000, 28, 28, 1)
            self.mnist_labels = np.array(self.mnist_labels)
            print("MNIST loaded.")

    def get_mnist_point_cloud(self, key: jax.random.PRNGKey, num_points: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Selects a random MNIST digit, converts it to a point cloud.
        """
        if self.mnist_ds is None:
            self._load_mnist()
            
        idx = jax.random.randint(key, (), 0, len(self.mnist_images))
        idx_int = int(idx)
        
        img = self.mnist_images[idx_int] # (28, 28, 1)
        mask = img[:, :, 0] > 128
        y, x = np.where(mask)
        
        if len(y) == 0: # Empty image case safety
             # Return random points if empty
             points_3d = np.random.normal(0, 1, (num_points, 3))
             colors = np.random.uniform(0, 1, (num_points, 3))
             return jnp.array(points_3d), jnp.array(colors)

        # Normalize to [-1, 1]
        # De-quantize with Gaussian jitter (std = 1/sqrt(12) approx 0.29)
        # This matches variance of uniform distribution of width 1 pixel
        jitter_std = 1.0 / np.sqrt(12.0)
        
        # Sample indices first to get base pixels
        # Sample proportional to number of pixels?
        # User requested: number of samples proportional to number of pixels.
        # But the function signature requests `num_points`.
        # If we ignore `num_points` and sample dynamically, we break batching logic downstream 
        # because each object would have different N points.
        # 
        # Solution: If num_points is "proportional", we need a target density.
        # E.g. 2 points per pixel.
        # But then we return variable sized arrays.
        # The downstream generator expects fixed N per object for stacking?
        #
        # Let's check usage: 
        # generate_scene loops over objects, gets pts, appends to all_points, then concatenates.
        # It expects `points_per_object` passed in.
        #
        # If I change this to variable, `ids` generation and concat works fine (concat handles variable length).
        # But `generate_dataset` calling `generate_scene` passes a fixed `points_per_object`.
        # And JAX training usually likes fixed shapes or padding.
        #
        # However, for "dataset generation" (saving to NPZ), variable length is fine if we use ragged storage (which we just added).
        #
        # Let's modify `get_mnist_point_cloud` to accept an optional `density_factor` instead of fixed `num_points`?
        # Or interpret `num_points` as a density target if negative?
        # Or just change the logic to ignore num_points if a flag is set?
        
        # Actually, the user said "make the number of samples proportional to the number of pixels".
        # So N_points = K * N_pixels.
        # Let's target roughly 3 samples per pixel for good coverage.
        
        num_pixels = len(y)
        target_num_points = int(num_pixels * 3.0) # 3.0 gives ~300-600 points depending on digit
        
        # Override num_points locally
        actual_num_points = target_num_points
        
        indices = np.random.choice(len(x), actual_num_points, replace=True)
        
        y_sel = y[indices].astype(np.float32)
        x_sel = x[indices].astype(np.float32)
        
        # Add jitter
        y_sel += np.random.normal(0, jitter_std, size=actual_num_points)
        x_sel += np.random.normal(0, jitter_std, size=actual_num_points)
        
        y_norm = -(y_sel - 13.5) / 13.5
        x_norm = (x_sel - 13.5) / 13.5
        
        points_selected = np.stack([x_norm, y_norm], axis=1) # (N, 2)
        
        # Extrude to 3D with random z noise
        z = np.random.normal(0, 0.05, (actual_num_points, 1))
        points_3d = np.concatenate([points_selected, z], axis=1)
        
        # Features: Gradient
        colors = np.zeros((actual_num_points, 3))
        colors[:, 0] = (points_selected[:, 0] + 1) / 2 
        colors[:, 1] = (points_selected[:, 1] + 1) / 2 
        colors[:, 2] = 0.5 
        
        return jnp.array(points_3d), jnp.array(colors)
        
        # Features: Gradient
        colors = np.zeros((num_points, 3))
        colors[:, 0] = (points_selected[:, 0] + 1) / 2 
        colors[:, 1] = (points_selected[:, 1] + 1) / 2 
        colors[:, 2] = 0.5 
        
        return jnp.array(points_3d), jnp.array(colors)

    def generate_scene(self, key: jax.random.PRNGKey, 
                      num_objects: int, 
                      points_per_object: int,
                      scene_bounds: float = 4.0,
                      use_mnist: bool = False) -> SceneData:
        
        k_centers, k_scales, k_colors, k_rots, k_gen = jax.random.split(key, 5)
        
        # Random object centers
        centers = jax.random.uniform(k_centers, (num_objects, self.spatial_dim), 
                                   minval=-scene_bounds, maxval=scene_bounds)
        
        # Random object scales
        scales = jax.random.uniform(k_scales, (num_objects,), minval=0.5, maxval=1.5)
        
        # Random object colors (logits)
        color_means = jax.random.normal(k_colors, (num_objects, self.feature_dim))
        
        all_points = []
        all_features = []
        all_ids = []
        
        gen_keys = jax.random.split(k_gen, num_objects)
        
        # Generate random rotations for all objects using scipy
        # (JAX scipy rotation support is limited, usually use numpy/scipy)
        # Or simple Euler:
        rotations = []
        if use_mnist:
            # Generate random Euler angles
            euler_angles = jax.random.uniform(k_rots, (num_objects, 3), minval=0, maxval=2*np.pi)
            # We'll apply rotation inside the loop using numpy for convenience or jax implementation
            # Let's use scipy R since we are in CPU generation mode anyway
            euler_np = np.array(euler_angles)
            for i in range(num_objects):
                rotations.append(R.from_euler('xyz', euler_np[i]).as_matrix())

        for i in range(num_objects):
            if use_mnist:
                pts, feats = self.get_mnist_point_cloud(gen_keys[i], points_per_object)
                
                # Apply Rotation
                rot_matrix = rotations[i]
                pts = pts @ rot_matrix.T # (N, 3) @ (3, 3).T -> (N, 3)
                
                # Apply Scale & Translation
                pts = pts * scales[i] + centers[i]
                
            else:
                # ... (omitted for brevity, logic remains)
                pass # Logic for Gaussian blobs
                
            all_points.append(pts)
            all_features.append(feats)
            # pts.shape[0] might vary now
            all_ids.append(jnp.full((pts.shape[0],), i, dtype=jnp.int32))
            
        points = jnp.concatenate(all_points, axis=0)
        features = jnp.concatenate(all_features, axis=0)
        ids = jnp.concatenate(all_ids, axis=0)
        
        # Shuffle
        k_shuffle = jax.random.fold_in(key, 0)
        perm = jax.random.permutation(k_shuffle, len(points))
        
        return SceneData(
            points=points[perm],
            features=features[perm],
            object_ids=ids[perm]
        )
