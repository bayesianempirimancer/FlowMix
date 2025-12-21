import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
# from src.train_mnist_2d import load_mnist_2d
from src.gmm.vb_gmm import fit_vb_gmm

def viz_gmm_points():
    # Load Data Directly
    DATASET_PATH = "data/mnist_2d_small_dataset.npz"
    try:
        if os.path.exists(DATASET_PATH):
            print(f"Loading dataset from {DATASET_PATH}...")
            data = np.load(DATASET_PATH)
            # Use all 100 samples from the small dataset or a subset
            X = jnp.array(data['points'])
            print(f"Data loaded successfully: {X.shape}")
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        print("Data not found, generating dummy data")
        X = jax.random.normal(jax.random.PRNGKey(0), (10, 500, 2))
        
    # Pick a few samples
    num_viz = 10
    key = jax.random.PRNGKey(42)
    # X is 100 samples, pick num_viz from it
    if len(X) >= num_viz:
        idx = jax.random.choice(key, len(X), shape=(num_viz,), replace=False)
        x_batch = X[idx]
    else:
        x_batch = X[:num_viz]
    
    print("Running fit_vb_gmm directly...")
    
    # Dynamic N_eff
    # Here we know we have valid data (all 500 points valid)
    N = x_batch.shape[1]
    # Use default params (lr=1.0, N_eff=None) as requested
    
    # Compute global std of the batch for scale
    global_std = jnp.std(x_batch)
    print(f"Global Std Dev of Batch: {global_std:.4f}")
    
    # Run with defaults except num_clusters=10 and global_scale=std
    # And init_method='random'
    # N_eff default is mask sum (actual N)
    # lr default is 1.0
    means, covs, weights, valid_mask, elbo_history = fit_vb_gmm(
        x_batch, 
        num_clusters=10, 
        num_iters=20, 
        seed=42,
        global_scale=global_std,
        init_method='random',
        N_eff=None, 
        lr=1.0
    )
    
    # Plot ELBO monotonicity check
    plt.figure(figsize=(10, 6))
    # elbo_history: (num_iters, B)
    iters = range(1, elbo_history.shape[0])
    for b in range(num_viz):
        plt.plot(iters, elbo_history[1:, b], label=f'Sample {b}')
    plt.title('ELBO History per Sample (Skipping Iter 0)')
    plt.xlabel('Iteration')
    plt.ylabel('ELBO')
    plt.grid(True)
    plt.savefig('gmm_elbo_check.png')
    print("Saved ELBO check to gmm_elbo_check.png")
    
    # Convert to geometric points (logic from GMMFeaturizer)
    B, K, D = means.shape
    
    # Eigen decomposition for features
    # covs: (B, K, D, D)
    # eigh returns eigenvalues in ascending order
    eigvals, eigvecs = jnp.linalg.eigh(covs)
    
    # Scale eigenvectors by sqrt(eigenvalues) * 1 (1 sigma)
    # eigvals can be slightly negative due to numerical errors, clip at 0
    eigvals = jnp.maximum(eigvals, 0.0)
    scaled_vecs = eigvecs * (1.0 * jnp.sqrt(eigvals))[:, :, None, :] # (B, K, D, D)
    
    # Create points: Mean, Mean + Vec, Mean - Vec
    # We want a sequence of points.
    # For each cluster k, we have mean, and D axes (pos and neg).
    # Total points per cluster: 1 + 2*D
    
    # means: (B, K, D) -> (B, K, 1, D)
    center_points = means[:, :, None, :]
    
    # axes_pos: Mean + ScaledVec
    # scaled_vecs is (B, K, D, D). The last dim is the eigenvector components. The second to last is which eigenvector.
    # We treat each eigenvector as an axis.
    
    # Broadcasting Correction:
    # means: (B, K, D). We want to add it to each column of scaled_vecs.
    # scaled_vecs: (B, K, Spatial_D, Vector_Index_D)
    # means needs to broadcast to (B, K, Spatial_D, 1)
    
    axes_pos = means[:, :, :, None] + scaled_vecs # (B, K, D, D)
    axes_neg = means[:, :, :, None] - scaled_vecs # (B, K, D, D)
    
    # But now axes_pos is (B, K, Spatial_D, Point_Index_D)
    # We want a list of points (B, K, Point_Index_D, Spatial_D)
    axes_pos_points = jnp.swapaxes(axes_pos, -1, -2)
    axes_neg_points = jnp.swapaxes(axes_neg, -1, -2)
    
    # Concatenate: (B, K, 1+2D, D)
    cluster_points = jnp.concatenate([center_points, axes_pos_points, axes_neg_points], axis=2)
    
    # Flatten to (B, L, D) where L = K * (1+2D)
    points_seq = cluster_points.reshape(B, -1, D)
    
    # Valid mask expansion
    # valid_mask: (B, K)
    # Expand to (B, K, 1+2D)
    valid_mask_seq = jnp.repeat(valid_mask[:, :, None], 1 + 2*D, axis=2)
    valid_mask_seq = valid_mask_seq.reshape(B, -1)
    
    # Scaling Stats
    print("\n--- Scaling Stats ---")
    print(f"Input Range: min={jnp.min(x_batch):.4f}, max={jnp.max(x_batch):.4f}, mean={jnp.mean(x_batch):.4f}")
    
    flat_points = points_seq[valid_mask_seq > 0.5]
    if len(flat_points) > 0:
        print(f"GMM Points Range: min={jnp.min(flat_points):.4f}, max={jnp.max(flat_points):.4f}, mean={jnp.mean(flat_points):.4f}")
    else:
        print("GMM Points: No valid points found!")
        
    # Plotting
    # Show side-by-side comparison: Standard GMM Fit vs Geometric Points
    rows = num_viz
    cols = 3 # Input, Standard Fit, Geometric Features
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    
    from matplotlib.patches import Ellipse
    
    def draw_ellipse(ax, mean, cov, color, alpha=0.2):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * 2 * np.sqrt(vals) # 2 sigma
        
        ell = Ellipse(xy=mean, width=width, height=height, angle=theta, 
                      color=color, alpha=alpha)
        ax.add_patch(ell)
        return ell

    for i in range(num_viz):
        # 1. Input Data
        ax_in = axes[i, 0]
        ax_in.scatter(x_batch[i, :, 0], x_batch[i, :, 1], s=1, c='black', alpha=0.5)
        ax_in.set_title(f"Input {i}")
        ax_in.axis('equal')
        ax_in.set_xlim([-1.5, 1.5])
        ax_in.set_ylim([-1.5, 1.5])
        
        # 2. Standard GMM Fit (Means + Ellipses)
        ax_fit = axes[i, 1]
        ax_fit.scatter(x_batch[i, :, 0], x_batch[i, :, 1], s=1, c='grey', alpha=0.1)
        
        # Iterate over clusters
        valid = valid_mask[i] > 0.5
        m = means[i][valid]
        c = covs[i][valid]
        w = weights[i][valid]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(m)))
        
        for k in range(len(m)):
            ax_fit.scatter(m[k, 0], m[k, 1], s=50, color=colors[k], marker='*')
            try:
                draw_ellipse(ax_fit, m[k], c[k], colors[k], alpha=0.2)
            except:
                pass # Singular covariance
            
        ax_fit.set_title(f"Standard GMM Fit (K={len(m)})")
        ax_fit.axis('equal')
        ax_fit.set_xlim([-1.5, 1.5])
        ax_fit.set_ylim([-1.5, 1.5])

        # 3. Geometric Features
        ax_geo = axes[i, 2]
        ax_geo.scatter(x_batch[i, :, 0], x_batch[i, :, 1], s=1, c='grey', alpha=0.1)
        
        # Filter valid geometric points
        mask = valid_mask_seq[i] > 0.5
        pts = points_seq[i][mask]
        
        if len(pts) > 0:
            # We can color code them by cluster if we reshape back or iterate
            # Let's iterate to match colors
            # points_seq is flattened (B, K*(1+2D), D)
            # Reconstruct structure: (K, 1+2D, D)
            # 1 center + 2 axes (pos) + 2 axes (neg) = 5 points per cluster
            
            # The mask is also flattened. We need to match valid clusters.
            # pts contains all valid points.
            # Assuming K valid clusters, we have K*5 points.
            
            cluster_pts = pts.reshape(-1, 5, 2) # (K_valid, 5, 2)
            
            for k in range(len(cluster_pts)):
                if k < len(colors):
                    col = colors[k]
                else:
                    col = 'red'
                    
                c_pts = cluster_pts[k]
                center = c_pts[0]
                # Plot center
                ax_geo.scatter(center[0], center[1], s=50, color=col, marker='*')
                # Plot axes points
                ax_geo.scatter(c_pts[1:, 0], c_pts[1:, 1], s=20, color=col, marker='x')
                # Draw lines
                # Axis 1: Ax1- (3) -> Mean (0) -> Ax1+ (1)
                ax_geo.plot([c_pts[3,0], c_pts[1,0]], [c_pts[3,1], c_pts[1,1]], color=col, alpha=0.5)
                # Axis 2: Ax2- (4) -> Mean (0) -> Ax2+ (2)
                ax_geo.plot([c_pts[4,0], c_pts[2,0]], [c_pts[4,1], c_pts[2,1]], color=col, alpha=0.5)
        
        ax_geo.set_title("Geometric Features")
        ax_geo.axis('equal')
        ax_geo.set_xlim([-1.5, 1.5])
        ax_geo.set_ylim([-1.5, 1.5])
        
    plt.tight_layout()
    save_path = "gmm_comparison.png"
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    viz_gmm_points()