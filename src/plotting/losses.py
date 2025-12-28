"""
Plotting utilities for loss trajectory visualization.
"""

import matplotlib.pyplot as plt
from pathlib import Path
from typing import List


def plot_loss_trajectory(loss_history, chamfer_history, flow_loss_history, 
                         prior_flow_loss_history, vae_kl_history, marginal_kl_history,
                         output_dir):
    """Plot comprehensive loss trajectory with all components.
    
    Args:
        loss_history: List of total loss values per epoch
        chamfer_history: List of Chamfer distance values per evaluation
        flow_loss_history: List of flow loss values per epoch
        prior_flow_loss_history: List of prior flow loss values per epoch
        vae_kl_history: List of VAE KL loss values per epoch
        marginal_kl_history: List of marginal KL loss values per epoch
        output_dir: Directory to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top left: Total loss
    axes[0, 0].plot(loss_history, label='Total Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Top right: Individual loss components (log scale)
    axes[0, 1].plot(flow_loss_history, label='Flow Loss', linewidth=1.5, alpha=0.8)
    if any(v > 0 for v in prior_flow_loss_history):
        axes[0, 1].plot(prior_flow_loss_history, label='Prior Flow Loss', linewidth=1.5, alpha=0.8)
    if any(v > 0 for v in vae_kl_history):
        axes[0, 1].plot(vae_kl_history, label='VAE KL Loss', linewidth=1.5, alpha=0.8)
    if any(v > 0 for v in marginal_kl_history):
        axes[0, 1].plot(marginal_kl_history, label='Marginal KL Loss', linewidth=1.5, alpha=0.8)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Loss Components (Log Scale)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')  # Log scale for better visibility
    
    # Bottom left: Chamfer distance
    axes[1, 0].plot(chamfer_history, color='green', linewidth=2)
    axes[1, 0].set_xlabel('Evaluation Step')
    axes[1, 0].set_ylabel('Chamfer Distance')
    axes[1, 0].set_title('Chamfer Distance (Lower is Better)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Bottom right: Loss components (linear scale)
    axes[1, 1].plot(flow_loss_history, label='Flow Loss', linewidth=1.5, alpha=0.8)
    if any(v > 0 for v in prior_flow_loss_history):
        axes[1, 1].plot(prior_flow_loss_history, label='Prior Flow Loss', linewidth=1.5, alpha=0.8)
    if any(v > 0 for v in vae_kl_history):
        axes[1, 1].plot(vae_kl_history, label='VAE KL Loss', linewidth=1.5, alpha=0.8)
    if any(v > 0 for v in marginal_kl_history):
        axes[1, 1].plot(marginal_kl_history, label='Marginal KL Loss', linewidth=1.5, alpha=0.8)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Loss Components (Linear Scale)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    loss_path = output_dir / "loss_trajectory.png"
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved loss trajectory to {loss_path}")



