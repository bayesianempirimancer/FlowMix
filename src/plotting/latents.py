"""
Plotting utilities for latent space analysis.

Includes t-SNE visualization of latent codes.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE


def create_tsne_plot(z_all, labels, output_path):
    """Create t-SNE visualization colored by digit identity.
    
    Args:
        z_all: Latent codes, shape (N, latent_dim)
        labels: Digit labels, shape (N,)
        output_path: Path to save the plot
    """
    print(f"\nComputing t-SNE embedding...")
    
    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    z_2d = tsne.fit_transform(z_all)
    
    print(f"t-SNE embedding shape: {z_2d.shape}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color map for digits
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for digit in range(10):
        digit_mask = labels == digit
        if np.any(digit_mask):
            ax.scatter(z_2d[digit_mask, 0], z_2d[digit_mask, 1], 
                      c=[colors[digit]], label=f'Digit {digit}', 
                      alpha=0.6, s=10)
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('t-SNE Visualization of Encoder Latent Codes\n(Colored by Digit Identity)', 
                 fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved t-SNE plot to {output_path}")
    plt.close()



