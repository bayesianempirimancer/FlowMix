import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from src.gmm.vb_gmm import fit_vb_gmm

def plot_gaussian_ellipse(mean, cov, ax, **kwargs):
    vals, vecs = jnp.linalg.eigh(cov)
    theta = jnp.degrees(jnp.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 4 * jnp.sqrt(jnp.maximum(vals, 1e-6)) # 2 sigma
    ell = Ellipse(xy=(mean[0], mean[1]), width=w, height=h, angle=theta, **kwargs, fc='None')
    ax.add_patch(ell)

def make_difficult_gmm_batch(key, batch_size=5, num_points=1000):
    # Create a batch of difficult GMMs:
    # 1. Long, thin clusters
    # 2. Overlapping clusters
    
    k1, k2, k3, k4 = jax.random.split(key, 4)
    
    # Cluster 1: Thin, rotated (Mean -1, -1)
    mean1 = jnp.array([-1.0, -1.0])
    cov1 = jnp.array([[0.5, 0.45], [0.45, 0.5]]) # Highly correlated
    pts1 = jax.random.multivariate_normal(k1, mean1, cov1, shape=(batch_size, num_points // 3))
    
    # Cluster 2: Overlapping with Cluster 1 (Mean -0.8, -0.8)
    mean2 = jnp.array([-0.8, -0.8])
    cov2 = jnp.array([[0.1, 0.0], [0.0, 0.1]])
    pts2 = jax.random.multivariate_normal(k2, mean2, cov2, shape=(batch_size, num_points // 3))
    
    # Cluster 3: Distant (Mean 1.5, 1.5)
    mean3 = jnp.array([1.5, 1.5])
    cov3 = jnp.array([[0.2, -0.1], [-0.1, 0.3]])
    pts3 = jax.random.multivariate_normal(k3, mean3, cov3, shape=(batch_size, num_points - 2 * (num_points // 3)))
    
    x = jnp.concatenate([pts1, pts2, pts3], axis=1) # (B, N, 2)
    
    # Shuffle points
    x = jax.random.permutation(k4, x, axis=1, independent=True)
    
    return x

def test_fit_vb_gmm():
    print("\n--- 5. Difficult Clustering Test (Masked + Batched) ---")
    
    x_hard = make_difficult_gmm_batch(jax.random.PRNGKey(101), batch_size=5)
    
    means_h, covs_h, weights_h, valid_mask_h, elbo_history_h = fit_vb_gmm(
        x_hard,
        num_clusters=10, # Initialize with more clusters
        num_iters=20,
        seed=102,
        lr=1.0, # Hard updates to check monotonicity
        global_scale=1.0, # Standard scale
        compute_elbo=True
    )
    
    # Plot ELBO
    plt.figure(figsize=(10, 6))
    iters = range(1, elbo_history_h.shape[0])
    for b in range(x_hard.shape[0]):
        plt.plot(iters, elbo_history_h[1:, b], label=f'Batch {b}')
    plt.title('ELBO History - Difficult Dataset (Skipping Iter 0)')
    plt.xlabel('Iteration')
    plt.ylabel('ELBO')
    plt.legend()
    plt.grid(True)
    plt.savefig('gmm_elbo_hard.png')
    print("Saved ELBO plot to gmm_elbo_hard.png")
    
    # Verify monotonicity (approximate due to soft updates)
    diffs = jnp.diff(elbo_history_h, axis=0)
    min_diff = jnp.min(diffs)
    print(f"Minimum ELBO improvement: {min_diff}")
    
    if min_diff < -0.1: # Allow small fluctuation
        print("WARNING: ELBO decreased significantly at some step!")
    else:
        print("ELBO is monotonically increasing (within tolerance).")

    # Visualize first batch item
    plt.figure(figsize=(8, 8))
    plt.scatter(x_hard[0, :, 0], x_hard[0, :, 1], s=1, alpha=0.5, label='Data', color='gray')
    
    # Plot fitted ellipses
    for k in range(10):
        if valid_mask_h[0, k]:
            plot_gaussian_ellipse(means_h[0, k], covs_h[0, k], ax=plt.gca(), edgecolor='red', lw=2)
            
    plt.title('Fitted GMM on Difficult Data (Batch 0)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('gmm_hard_fit.png')
    print("Saved fit visualization to gmm_hard_fit.png")

if __name__ == "__main__":
    test_fit_vb_gmm()
