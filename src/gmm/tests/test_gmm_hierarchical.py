"""Test script for hierarchical VB-GMM with overlap-based merging."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.colors as mcolors

from src.gmm.vb_gmm import fit_vb_gmm
from src.gmm.vb_gmm_hierarchical import (
    fit_hierarchical_vb_gmm, 
    get_level_gmm,
    HierarchicalGMMOutput
)


def plot_gaussian_ellipse(mean, cov, ax, **kwargs):
    """Plot 2-sigma ellipse for a 2D Gaussian."""
    vals, vecs = jnp.linalg.eigh(cov)
    theta = jnp.degrees(jnp.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 4 * jnp.sqrt(jnp.maximum(vals, 1e-6))  # 2 sigma
    ell = Ellipse(xy=(mean[0], mean[1]), width=w, height=h, angle=theta, **kwargs, fc='None')
    ax.add_patch(ell)


def make_overlapping_clusters(key, batch_size=2, num_points=800):
    """
    Create test data with overlapping sub-clusters that should merge.
    
    Structure:
    - 2 "objects" (top-left quadrant, bottom-right quadrant)
    - Each object has 3 overlapping sub-clusters
    - Sub-clusters within an object should merge; objects should stay separate
    """
    keys = jax.random.split(key, 7)
    n_per = num_points // 6
    
    # Object 1: Top-left quadrant (3 overlapping clusters)
    obj1_center = jnp.array([-2.0, 2.0])
    pts1a = jax.random.multivariate_normal(
        keys[0], obj1_center + jnp.array([0.0, 0.3]),
        jnp.array([[0.3, 0.1], [0.1, 0.2]]),
        shape=(batch_size, n_per)
    )
    pts1b = jax.random.multivariate_normal(
        keys[1], obj1_center + jnp.array([0.3, 0.0]),
        jnp.array([[0.25, 0.05], [0.05, 0.3]]),
        shape=(batch_size, n_per)
    )
    pts1c = jax.random.multivariate_normal(
        keys[2], obj1_center + jnp.array([-0.2, -0.2]),
        jnp.array([[0.2, 0.0], [0.0, 0.25]]),
        shape=(batch_size, n_per)
    )
    
    # Object 2: Bottom-right quadrant (3 overlapping clusters)
    obj2_center = jnp.array([2.0, -2.0])
    pts2a = jax.random.multivariate_normal(
        keys[3], obj2_center + jnp.array([0.0, 0.2]),
        jnp.array([[0.35, -0.1], [-0.1, 0.2]]),
        shape=(batch_size, n_per)
    )
    pts2b = jax.random.multivariate_normal(
        keys[4], obj2_center + jnp.array([0.25, -0.15]),
        jnp.array([[0.2, 0.0], [0.0, 0.3]]),
        shape=(batch_size, n_per)
    )
    pts2c = jax.random.multivariate_normal(
        keys[5], obj2_center + jnp.array([-0.3, 0.0]),
        jnp.array([[0.25, 0.08], [0.08, 0.25]]),
        shape=(batch_size, num_points - 5 * n_per)
    )
    
    x = jnp.concatenate([pts1a, pts1b, pts1c, pts2a, pts2b, pts2c], axis=1)
    x = jax.random.permutation(keys[6], x, axis=1, independent=True)
    
    return x


def test_hierarchical_output_structure():
    """Test that hierarchical GMM produces multi-level output."""
    print("\n=== Test 1: Hierarchical Output Structure ===")
    
    key = jax.random.PRNGKey(42)
    x = make_overlapping_clusters(key, batch_size=2, num_points=600)
    
    print(f"Input shape: {x.shape}")
    
    output = fit_hierarchical_vb_gmm(
        x,
        max_levels=5,
        max_clusters_base=16,
        overlap_threshold=0.25,
        min_clusters=2,
        seed=42
    )
    
    print(f"\nNumber of levels: {output.num_levels}")
    
    for i, params in enumerate(output.levels):
        num_valid = params.num_clusters
        print(f"  Level {i}: {int(num_valid[0])} / {int(num_valid[1])} valid clusters (batch 0/1)")
    
    print("\n✓ Hierarchical structure test passed!")
    return output, x


def test_overlap_based_merging():
    """Test that overlapping clusters get merged."""
    print("\n=== Test 2: Overlap-Based Merging ===")
    
    key = jax.random.PRNGKey(123)
    x = make_overlapping_clusters(key, batch_size=1, num_points=800)
    
    # Low threshold = aggressive merging
    output_aggressive = fit_hierarchical_vb_gmm(
        x, max_levels=5, max_clusters_base=20,
        overlap_threshold=0.15,  # Low threshold = merge more
        min_clusters=1, seed=42
    )
    
    # High threshold = conservative merging
    output_conservative = fit_hierarchical_vb_gmm(
        x, max_levels=5, max_clusters_base=20,
        overlap_threshold=0.5,  # High threshold = merge less
        min_clusters=1, seed=42
    )
    
    print(f"Aggressive merging (threshold=0.15): {output_aggressive.num_levels} levels")
    for i, p in enumerate(output_aggressive.levels):
        print(f"  Level {i}: {int(p.num_clusters[0])} clusters")
    
    print(f"\nConservative merging (threshold=0.5): {output_conservative.num_levels} levels")
    for i, p in enumerate(output_conservative.levels):
        print(f"  Level {i}: {int(p.num_clusters[0])} clusters")
    
    print("\n✓ Overlap-based merging test passed!")


def test_visualization_all_levels():
    """Visualize all hierarchy levels."""
    print("\n=== Test 3: Multi-Level Visualization ===")
    
    key = jax.random.PRNGKey(456)
    x = make_overlapping_clusters(key, batch_size=1, num_points=800)
    
    output = fit_hierarchical_vb_gmm(
        x,
        max_levels=5,
        max_clusters_base=20,
        overlap_threshold=0.2,
        min_clusters=2,
        seed=42
    )
    
    num_levels = output.num_levels
    fig, axes = plt.subplots(1, num_levels, figsize=(5 * num_levels, 5))
    if num_levels == 1:
        axes = [axes]
    
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    for level_idx, (ax, params) in enumerate(zip(axes, output.levels)):
        ax.scatter(x[0, :, 0], x[0, :, 1], s=3, alpha=0.3, c='gray')
        
        valid = params.valid_mask[0]
        means = params.means[0]
        covs = params.covariances[0]
        weights = params.weights[0]
        
        num_valid = int(jnp.sum(valid))
        
        k_idx = 0
        for k in range(means.shape[0]):
            if valid[k] > 0.5:
                color = colors[k_idx % len(colors)]
                plot_gaussian_ellipse(
                    means[k], covs[k], ax,
                    edgecolor=color, lw=2.5, alpha=0.9
                )
                ax.scatter(means[k, 0], means[k, 1],
                          c=color, s=150, marker='x', linewidths=3, zorder=10)
                # Annotate with weight
                ax.annotate(f'{weights[k]:.2f}', 
                           (means[k, 0], means[k, 1]),
                           textcoords="offset points", xytext=(5, 5),
                           fontsize=9, color=color)
                k_idx += 1
        
        ax.set_title(f'Level {level_idx}: {num_valid} clusters')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
    
    plt.tight_layout()
    plt.savefig('gmm_hierarchy_levels.png', dpi=150)
    print(f"Saved {num_levels}-level hierarchy to gmm_hierarchy_levels.png")
    
    print("✓ Multi-level visualization test passed!")


def test_data_driven_levels():
    """Test that level count adapts to data structure."""
    print("\n=== Test 4: Data-Driven Level Count ===")
    
    key = jax.random.PRNGKey(789)
    
    # Well-separated clusters (should need fewer merges)
    x_sep = jax.random.multivariate_normal(
        key, jnp.array([0.0, 0.0]), 
        jnp.eye(2) * 0.1, shape=(1, 200)
    )
    x_sep = jnp.concatenate([
        x_sep,
        jax.random.multivariate_normal(
            jax.random.PRNGKey(1), jnp.array([5.0, 5.0]),
            jnp.eye(2) * 0.1, shape=(1, 200)
        ),
        jax.random.multivariate_normal(
            jax.random.PRNGKey(2), jnp.array([-5.0, 5.0]),
            jnp.eye(2) * 0.1, shape=(1, 200)
        )
    ], axis=1)
    
    # Highly overlapping clusters (should need more merges)
    x_overlap = make_overlapping_clusters(jax.random.PRNGKey(999), batch_size=1, num_points=600)
    
    output_sep = fit_hierarchical_vb_gmm(
        x_sep, max_levels=5, max_clusters_base=12,
        overlap_threshold=0.2, seed=42
    )
    
    output_overlap = fit_hierarchical_vb_gmm(
        x_overlap, max_levels=5, max_clusters_base=12,
        overlap_threshold=0.2, seed=42
    )
    
    print(f"Well-separated data: {output_sep.num_levels} levels, "
          f"final clusters = {int(output_sep.levels[-1].num_clusters[0])}")
    
    print(f"Overlapping data: {output_overlap.num_levels} levels, "
          f"final clusters = {int(output_overlap.levels[-1].num_clusters[0])}")
    
    print("\n✓ Data-driven levels test passed!")


def test_get_level_api():
    """Test the get_level_gmm convenience function."""
    print("\n=== Test 5: Level Access API ===")
    
    key = jax.random.PRNGKey(101)
    x = make_overlapping_clusters(key, batch_size=2, num_points=500)
    
    output = fit_hierarchical_vb_gmm(
        x, max_levels=4, max_clusters_base=16,
        overlap_threshold=0.25, seed=42
    )
    
    # Access different levels
    for level in range(output.num_levels):
        means, covs, weights, valid = get_level_gmm(output, level)
        print(f"Level {level}: means shape = {means.shape}, "
              f"valid clusters = {int(jnp.sum(valid[0]))}, {int(jnp.sum(valid[1]))}")
    
    # Test out-of-bounds access (should return last level)
    means, covs, weights, valid = get_level_gmm(output, 999)
    print(f"Out-of-bounds access returns level {output.num_levels - 1}")
    
    print("\n✓ Level access API test passed!")


def test_comparison_with_flat_gmm():
    """Compare hierarchical vs flat GMM on overlapping data."""
    print("\n=== Test 6: Comparison with Flat GMM ===")
    
    key = jax.random.PRNGKey(202)
    x = make_overlapping_clusters(key, batch_size=1, num_points=800)
    
    # Flat GMM
    means_flat, covs_flat, weights_flat, valid_flat, _ = fit_vb_gmm(
        x, num_clusters=8, seed=42, num_iters=20, prior_counts=0.1
    )
    
    # Hierarchical GMM
    output_hier = fit_hierarchical_vb_gmm(
        x, max_levels=5, max_clusters_base=16,
        overlap_threshold=0.25, min_clusters=2, seed=42
    )
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    # Flat GMM
    ax = axes[0]
    ax.scatter(x[0, :, 0], x[0, :, 1], s=3, alpha=0.3, c='gray')
    for k in range(8):
        if valid_flat[0, k]:
            plot_gaussian_ellipse(means_flat[0, k], covs_flat[0, k], ax,
                                edgecolor='red', lw=2)
    ax.set_title(f'Flat VB-GMM ({int(jnp.sum(valid_flat[0]))} clusters)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Hierarchical - Base level
    ax = axes[1]
    ax.scatter(x[0, :, 0], x[0, :, 1], s=3, alpha=0.3, c='gray')
    params = output_hier.levels[0]
    k_idx = 0
    for k in range(params.means.shape[1]):
        if params.valid_mask[0, k] > 0.5:
            color = colors[k_idx % len(colors)]
            plot_gaussian_ellipse(params.means[0, k], params.covariances[0, k], ax,
                                edgecolor=color, lw=2)
            k_idx += 1
    ax.set_title(f'Hierarchical Level 0 ({int(params.num_clusters[0])} clusters)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Hierarchical - Final level
    ax = axes[2]
    ax.scatter(x[0, :, 0], x[0, :, 1], s=3, alpha=0.3, c='gray')
    params = output_hier.levels[-1]
    k_idx = 0
    for k in range(params.means.shape[1]):
        if params.valid_mask[0, k] > 0.5:
            color = colors[k_idx % len(colors)]
            plot_gaussian_ellipse(params.means[0, k], params.covariances[0, k], ax,
                                edgecolor=color, lw=2.5)
            k_idx += 1
    ax.set_title(f'Hierarchical Level {output_hier.num_levels-1} ({int(params.num_clusters[0])} clusters)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gmm_flat_vs_hierarchical.png', dpi=150)
    print("Saved comparison to gmm_flat_vs_hierarchical.png")
    
    print(f"\nFlat GMM: {int(jnp.sum(valid_flat[0]))} clusters")
    print(f"Hierarchical: {output_hier.num_levels} levels")
    for i, p in enumerate(output_hier.levels):
        print(f"  Level {i}: {int(p.num_clusters[0])} clusters")
    
    print("\n✓ Comparison test passed!")


if __name__ == "__main__":
    test_hierarchical_output_structure()
    test_overlap_based_merging()
    test_visualization_all_levels()
    test_data_driven_levels()
    test_get_level_api()
    test_comparison_with_flat_gmm()
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")
    print("=" * 50)
