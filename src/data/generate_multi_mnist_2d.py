import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds
import tqdm
import os
import matplotlib.pyplot as plt

DATASET_PATH = "data/multi_mnist_2d_dataset.npz"

def generate_multi_mnist_2d(num_scenes=20000, points_per_digit=500, canvas_range=(-4, 4)):
    print("Generating Multi-MNIST 2D Point Cloud Dataset...")
    
    # Load MNIST source
    ds = tfds.load('mnist', split='train', shuffle_files=True)
    # We need random access, so iterate and cache or reservoir sample
    # Caching all 60k is fine (small in memory)
    print("Loading source MNIST images...")
    mnist_data = []
    for ex in tfds.as_numpy(ds):
        mnist_data.append(ex['image'][:, :, 0])
    mnist_data = np.array(mnist_data)
    num_mnist = len(mnist_data)
    
    all_points = []
    all_ids = []
    
    rng = np.random.RandomState(42)
    
    for i in tqdm.tqdm(range(num_scenes)):
        # 1-4 digits
        num_digits = rng.randint(1, 5)
        
        scene_points = []
        scene_ids = []
        
        for obj_idx in range(num_digits):
            # Sample digit
            img_idx = rng.randint(0, num_mnist)
            img = mnist_data[img_idx]
            
            # Convert to points
            y, x = np.where(img > 128)
            
            if len(y) == 0:
                pts = rng.normal(0, 1, (points_per_digit, 2))
            else:
                jitter_std = 1.0 / np.sqrt(12.0)
                indices = rng.choice(len(x), points_per_digit, replace=True)
                
                y_sel = y[indices].astype(np.float32) + rng.normal(0, jitter_std, size=points_per_digit)
                x_sel = x[indices].astype(np.float32) + rng.normal(0, jitter_std, size=points_per_digit)
                
                # Normalize to [-1, 1]
                y_norm = -(y_sel - 13.5) / 13.5
                x_norm = (x_sel - 13.5) / 13.5
                
                pts = np.stack([x_norm, y_norm], axis=1)
                
            # Random Translation within canvas
            # Keep digits roughly inside canvas_range
            # Digit radius approx 1. Canvas [-4, 4].
            # Center can be in [-3, 3]
            border = 1.0
            t_min, t_max = canvas_range[0] + border, canvas_range[1] - border
            t = rng.uniform(t_min, t_max, size=(2,))
            
            pts = pts + t
            
            scene_points.append(pts)
            scene_ids.append(np.full((points_per_digit,), obj_idx))
            
        # Concatenate scene
        scene_points = np.concatenate(scene_points, axis=0)
        scene_ids = np.concatenate(scene_ids, axis=0)
        
        all_points.append(scene_points)
        all_ids.append(scene_ids)
        
    # Save as object array (ragged)
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    np.savez(DATASET_PATH, 
             points=np.array(all_points, dtype=object),
             object_ids=np.array(all_ids, dtype=object))
             
    print(f"Saved {num_scenes} scenes to {DATASET_PATH}")
    
    # Visualize one
    plt.figure(figsize=(5, 5))
    idx = 0
    pts = all_points[idx]
    ids = all_ids[idx]
    plt.scatter(pts[:, 0], pts[:, 1], c=ids, cmap='tab10', s=2)
    plt.title(f"Sample Scene (N={len(np.unique(ids))})")
    plt.xlim(canvas_range)
    plt.ylim(canvas_range)
    plt.savefig("multi_mnist_sample.png")
    print("Saved sample visualization to multi_mnist_sample.png")

if __name__ == "__main__":
    generate_multi_mnist_2d()

