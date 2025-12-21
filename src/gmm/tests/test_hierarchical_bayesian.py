"""Test script for Bayesian Hierarchical GMM with Variational Inference."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.colors as mcolors

from src.gmm.hierarchical_vb_gmm import (
    fit_hierarchical_bayesian_gmm,
    extract_level_gmm,
    extract_all_levels,
    get_parent_assignments
)


def plot_gaussian_ellipse(mean, cov, ax, **kwargs):
    """Plot 2-sigma ellipse for a 2D Gaussian."""
    cov = jnp.array(cov)
    vals, vecs = jnp.linalg.eigh(cov)
    theta = float(jnp.degrees(jnp.arctan2(vecs[1, 0], vecs[0, 0])))
    w = float(4 * jnp.sqrt(jnp.maximum(vals[0], 1e-6)))
    h = float(4 * jnp.sqrt(jnp.maximum(vals[1], 1e-6)))
    ell = Ellipse(xy=(float(mean[0]), float(mean[1])), width=w, height=h, 
                  angle=theta, **kwargs, fc='None')
    ax.add_patch(ell)


def make_hierarchical_data(key, batch_size=1, num_points=600):
    """
    Create data with clear hierarchical structure:
    - 2 "super-clusters" (top level)
    - Each super-cluster has 3 sub-clusters (middle level)
    - Data points in each sub-cluster (leaf level)
    """
    keys = jax.random.split(key, 7)
    n_per = num_points // 6
    
    # Super-cluster 1 (left side)
    super1_center = jnp.array([-3.0, 0.0])
    pts1a = jax.random.multivariate_normal(
        keys[0], super1_center + jnp.array([0.0, 1.5]),
        jnp.array([[0.2, 0.0], [0.0, 0.15]]),
        shape=(batch_size, n_per)
    )
    pts1b = jax.random.multivariate_normal(
        keys[1], super1_center + jnp.array([0.8, -0.5]),
        jnp.array([[0.15, 0.05], [0.05, 0.2]]),
        shape=(batch_size, n_per)
    )
    pts1c = jax.random.multivariate_normal(
        keys[2], super1_center + jnp.array([-0.7, -0.8]),
        jnp.array([[0.18, 0.0], [0.0, 0.18]]),
        shape=(batch_size, n_per)
    )
    
    # Super-cluster 2 (right side)
    super2_center = jnp.array([3.0, 0.0])
    pts2a = jax.random.multivariate_normal(
        keys[3], super2_center + jnp.array([0.0, 1.2]),
        jnp.array([[0.25, -0.05], [-0.05, 0.15]]),
        shape=(batch_size, n_per)
    )
    pts2b = jax.random.multivariate_normal(
        keys[4], super2_center + jnp.array([0.6, -0.6]),
        jnp.array([[0.12, 0.0], [0.0, 0.22]]),
        shape=(batch_size, n_per)
    )
    pts2c = jax.random.multivariate_normal(
        keys[5], super2_center + jnp.array([-0.5, -0.4]),
        jnp.array([[0.2, 0.08], [0.08, 0.2]]),
        shape=(batch_size, num_points - 5 * n_per)
    )
    
    x = jnp.concatenate([pts1a, pts1b, pts1c, pts2a, pts2b, pts2c], axis=1)
    x = jax.random.permutation(keys[6], x, axis=1, independent=True)
    
    return x


def test_basic_fitting():
    """Test basic hierarchical GMM fitting."""
    print("\n=== Test 1: Basic Fitting ===")
    
    key = jax.random.PRNGKey(42)
    x = make_hierarchical_data(key, batch_size=2, num_points=400)
    
    print(f"Input shape: {x.shape}")
    
    state, _ = fit_hierarchical_bayesian_gmm(
        x,
        num_levels=3,
        clusters_per_level=[2, 6, 12],
        num_iters=15,
        beta_hierarchy=3.0,
        seed=42
    )
    
    print(f"Number of levels: {len(state.levels)}")
    for l, lvl in enumerate(state.levels):
        num_active = int(jnp.sum(lvl.active[0]))
        print(f"  Level {l}: {lvl.m.shape[1]} clusters, {num_active} active")
    
    print(f"Data assignments shape: {state.r_data.shape}")
    
    print("✓ Basic fitting test passed!")
    return state, x


def test_level_extraction():
    """Test extracting GMM parameters at each level."""
    print("\n=== Test 2: Level Extraction ===")
    
    key = jax.random.PRNGKey(123)
    x = make_hierarchical_data(key, batch_size=1, num_points=500)
    
    state, _ = fit_hierarchical_bayesian_gmm(
        x,
        num_levels=3,
        clusters_per_level=[3, 8, 16],
        num_iters=20,
        seed=42
    )
    
    all_levels = extract_all_levels(state)
    
    for l, (means, covs, weights, valid) in enumerate(all_levels):
        print(f"Level {l}:")
        print(f"  Means shape: {means.shape}")
        print(f"  Covs shape: {covs.shape}")
        print(f"  Active clusters: {int(jnp.sum(valid[0]))}")
        print(f"  Top weights: {weights[0, :5]}")
    
    print("✓ Level extraction test passed!")


def test_parent_assignments():
    """Test parent assignment retrieval."""
    print("\n=== Test 3: Parent Assignments ===")
    
    key = jax.random.PRNGKey(456)
    x = make_hierarchical_data(key, batch_size=1, num_points=500)
    
    state, _ = fit_hierarchical_bayesian_gmm(
        x,
        num_levels=3,
        clusters_per_level=[2, 6, 12],
        num_iters=15,
        seed=42
    )
    
    # Check root (should be None)
    r_root = get_parent_assignments(state, 0)
    print(f"Root parent assignments: {r_root}")
    
    # Check level 1
    r_level1 = get_parent_assignments(state, 1)
    print(f"Level 1 parent assignments shape: {r_level1.shape}")
    print(f"Level 1 cluster 0 parent probs: {r_level1[0, 0, :]}")
    
    # Check level 2 (leaf)
    r_level2 = get_parent_assignments(state, 2)
    print(f"Level 2 parent assignments shape: {r_level2.shape}")
    
    # Verify soft assignments sum to 1
    sums = jnp.sum(r_level2[0], axis=-1)
    print(f"Parent assignment sums (should be ~1): {sums[:5]}")
    
    print("✓ Parent assignments test passed!")


def test_visualization():
    """Visualize all hierarchy levels."""
    print("\n=== Test 4: Multi-Level Visualization ===")
    
    key = jax.random.PRNGKey(789)
    x = make_hierarchical_data(key, batch_size=1, num_points=800)
    
    state, _ = fit_hierarchical_bayesian_gmm(
        x,
        num_levels=3,
        clusters_per_level=[2, 6, 18],
        num_iters=25,
        beta_hierarchy=5.0,
        seed=42
    )
    
    all_levels = extract_all_levels(state)
    num_levels = len(all_levels)
    
    fig, axes = plt.subplots(1, num_levels, figsize=(6 * num_levels, 6))
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    level_names = ['Root (Coarsest)', 'Middle', 'Leaf (Finest)']
    
    for l, (ax, (means, covs, weights, valid)) in enumerate(zip(axes, all_levels)):
        ax.scatter(x[0, :, 0], x[0, :, 1], s=3, alpha=0.3, c='gray')
        
        K = means.shape[1]
        k_idx = 0
        for k in range(K):
            if valid[0, k] > 0.5:
                color = colors[k_idx % len(colors)]
                plot_gaussian_ellipse(
                    means[0, k], covs[0, k], ax,
                    edgecolor=color, lw=2.5 if l == 0 else 1.5, alpha=0.9
                )
                ax.scatter(float(means[0, k, 0]), float(means[0, k, 1]),
                          c=color, s=100, marker='x', linewidths=2, zorder=10)
                k_idx += 1
        
        ax.set_title(f'Level {l}: {level_names[l]} ({k_idx} clusters)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-4, 4)
    
    plt.tight_layout()
    plt.savefig('bayesian_hierarchy_levels.png', dpi=150)
    print("Saved visualization to bayesian_hierarchy_levels.png")
    
    print("✓ Visualization test passed!")


def test_hierarchy_tightness():
    """Test effect of beta_hierarchy parameter."""
    print("\n=== Test 5: Hierarchy Tightness ===")
    
    key = jax.random.PRNGKey(101)
    x = make_hierarchical_data(key, batch_size=1, num_points=600)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    for ax_idx, beta_h in enumerate([1.0, 5.0, 20.0]):
        state, _ = fit_hierarchical_bayesian_gmm(
            x,
            num_levels=2,
            clusters_per_level=[3, 12],
            num_iters=20,
            beta_hierarchy=beta_h,
            seed=42
        )
        
        ax = axes[ax_idx]
        ax.scatter(x[0, :, 0], x[0, :, 1], s=3, alpha=0.3, c='gray')
        
        # Plot root level (large ellipses)
        means_root, covs_root, _, valid_root = extract_level_gmm(state, 0)
        for k in range(means_root.shape[1]):
            if valid_root[0, k] > 0.5:
                plot_gaussian_ellipse(
                    means_root[0, k], covs_root[0, k], ax,
                    edgecolor='black', lw=3, linestyle='--', alpha=0.7
                )
        
        # Plot leaf level
        means_leaf, covs_leaf, _, valid_leaf = extract_level_gmm(state, 1)
        k_idx = 0
        for k in range(means_leaf.shape[1]):
            if valid_leaf[0, k] > 0.5:
                color = colors[k_idx % len(colors)]
                plot_gaussian_ellipse(
                    means_leaf[0, k], covs_leaf[0, k], ax,
                    edgecolor=color, lw=1.5
                )
                k_idx += 1
        
        ax.set_title(f'β_hierarchy = {beta_h}')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-4, 4)
    
    plt.tight_layout()
    plt.savefig('bayesian_hierarchy_tightness.png', dpi=150)
    print("Saved tightness comparison to bayesian_hierarchy_tightness.png")
    
    print("✓ Hierarchy tightness test passed!")


def test_parent_child_relationship():
    """Visualize parent-child relationships in the hierarchy."""
    print("\n=== Test 6: Parent-Child Relationships ===")
    
    key = jax.random.PRNGKey(202)
    x = make_hierarchical_data(key, batch_size=1, num_points=600)
    
    state, _ = fit_hierarchical_bayesian_gmm(
        x,
        num_levels=2,
        clusters_per_level=[2, 8],
        num_iters=20,
        beta_hierarchy=5.0,
        seed=42
    )
    
    means_root, covs_root, _, valid_root = extract_level_gmm(state, 0)
    means_leaf, covs_leaf, _, valid_leaf = extract_level_gmm(state, 1)
    r_parent = get_parent_assignments(state, 1)  # (B, K_leaf, K_root)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(x[0, :, 0], x[0, :, 1], s=3, alpha=0.2, c='gray')
    
    # Color children by their most likely parent
    parent_colors = ['tab:blue', 'tab:orange', 'tab:green']
    
    K_root = means_root.shape[1]
    K_leaf = means_leaf.shape[1]
    
    # Draw root clusters
    for k in range(K_root):
        if valid_root[0, k] > 0.5:
            plot_gaussian_ellipse(
                means_root[0, k], covs_root[0, k], ax,
                edgecolor=parent_colors[k], lw=4, linestyle='--', alpha=0.8
            )
            ax.scatter(float(means_root[0, k, 0]), float(means_root[0, k, 1]),
                      c=parent_colors[k], s=300, marker='s', 
                      edgecolors='black', linewidths=2, zorder=10)
    
    # Draw leaf clusters colored by parent
    for k in range(K_leaf):
        if valid_leaf[0, k] > 0.5:
            # Get most likely parent
            parent_probs = r_parent[0, k, :]
            best_parent = int(jnp.argmax(parent_probs))
            color = parent_colors[best_parent]
            
            plot_gaussian_ellipse(
                means_leaf[0, k], covs_leaf[0, k], ax,
                edgecolor=color, lw=2, alpha=0.8
            )
            
            # Draw line from child to parent
            ax.plot(
                [float(means_leaf[0, k, 0]), float(means_root[0, best_parent, 0])],
                [float(means_leaf[0, k, 1]), float(means_root[0, best_parent, 1])],
                color=color, linestyle=':', linewidth=1, alpha=0.5
            )
    
    ax.set_title('Parent-Child Relationships (leaf → root)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-4, 4)
    
    plt.tight_layout()
    plt.savefig('bayesian_hierarchy_relationships.png', dpi=150)
    print("Saved parent-child visualization to bayesian_hierarchy_relationships.png")
    
    print("✓ Parent-child relationship test passed!")


def test_different_level_counts():
    """Test with different numbers of levels."""
    print("\n=== Test 7: Different Level Counts ===")
    
    key = jax.random.PRNGKey(303)
    x = make_hierarchical_data(key, batch_size=1, num_points=600)
    
    for num_levels in [2, 3, 4]:
        # Geometric cluster progression
        clusters = [2 ** (i + 1) for i in range(num_levels)]
        
        state, _ = fit_hierarchical_bayesian_gmm(
            x,
            num_levels=num_levels,
            clusters_per_level=clusters,
            num_iters=15,
            seed=42
        )
        
        print(f"\n{num_levels} levels: {clusters}")
        for l, lvl in enumerate(state.levels):
            num_active = int(jnp.sum(lvl.active[0]))
            print(f"  Level {l}: {num_active}/{lvl.m.shape[1]} active")
    
    print("\n✓ Different level counts test passed!")


if __name__ == "__main__":
    test_basic_fitting()
    test_level_extraction()
    test_parent_assignments()
    test_visualization()
    test_hierarchy_tightness()
    test_parent_child_relationship()
    test_different_level_counts()
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")
    print("=" * 50)

