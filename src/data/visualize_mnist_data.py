"""
Visualization routines for MNIST Point Cloud Data

This module provides plotting and visualization functions for MNIST point cloud datasets.
All visualization code is separated from the data generation module.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, List, Union
from pathlib import Path
import os

try:
    from .mnist_point_cloud_data import load_mnist_point_clouds, MNISTPointCloudDataset
except ImportError:
    # Fallback for when running as script
    from mnist_point_cloud_data import load_mnist_point_clouds, MNISTPointCloudDataset


# ============================================================================
# Single Point Cloud Visualization
# ============================================================================

def plot_point_cloud(
    points: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    color: Union[str, np.ndarray] = 'blue',
    alpha: float = 0.6,
    s: float = 1.0,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    show_axes: bool = True,
    figsize: Tuple[int, int] = (5, 5)
) -> plt.Figure:
    """Plot a single point cloud.
    
    Args:
        points: Point cloud array (N, 2)
        ax: Matplotlib axes (if None, creates new figure)
        title: Plot title
        color: Color for points (string or array for per-point colors)
        alpha: Transparency
        s: Point size
        xlim: X-axis limits (auto if None)
        ylim: Y-axis limits (auto if None)
        show_axes: Whether to show axes
        figsize: Figure size if creating new figure
        
    Returns:
        matplotlib Figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    ax.scatter(points[:, 0], points[:, 1], c=color, alpha=alpha, s=s)
    
    if title:
        ax.set_title(title)
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    if not show_axes:
        ax.axis('off')
    else:
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    return fig


def plot_point_cloud_with_labels(
    points: np.ndarray,
    labels: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    cmap: str = 'tab10',
    alpha: float = 0.6,
    s: float = 1.0,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    show_axes: bool = True,
    figsize: Tuple[int, int] = (5, 5)
) -> plt.Figure:
    """Plot a point cloud with per-point labels (for multi-object scenes).
    
    Args:
        points: Point cloud array (N, 2)
        labels: Object IDs for each point (N,)
        ax: Matplotlib axes (if None, creates new figure)
        title: Plot title
        cmap: Colormap for labels
        alpha: Transparency
        s: Point size
        xlim: X-axis limits (auto if None)
        ylim: Y-axis limits (auto if None)
        show_axes: Whether to show axes
        figsize: Figure size if creating new figure
        
    Returns:
        matplotlib Figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    if labels is not None:
        scatter = ax.scatter(points[:, 0], points[:, 1], c=labels, cmap=cmap, 
                           alpha=alpha, s=s)
        # Add colorbar if labels are provided
        if len(np.unique(labels)) > 1:
            plt.colorbar(scatter, ax=ax, label='Object ID')
    else:
        ax.scatter(points[:, 0], points[:, 1], alpha=alpha, s=s)
    
    if title:
        ax.set_title(title)
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    if not show_axes:
        ax.axis('off')
    else:
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    return fig


# ============================================================================
# Dataset Visualization
# ============================================================================

def visualize_dataset_samples(
    file_path: str,
    num_samples: int = 16,
    num_cols: int = 4,
    file_format: Optional[str] = None,
    output_path: Optional[str] = None,
    title_prefix: str = "Sample",
    figsize: Optional[Tuple[int, int]] = None
) -> plt.Figure:
    """Visualize multiple samples from a dataset in a grid.
    
    Args:
        file_path: Path to dataset file
        num_samples: Number of samples to visualize
        num_cols: Number of columns in grid
        file_format: "npz" or "pkl" (auto-detected if None)
        output_path: Path to save figure (if None, doesn't save)
        title_prefix: Prefix for sample titles
        figsize: Figure size (auto if None)
        
    Returns:
        matplotlib Figure object
    """
    dataset = MNISTPointCloudDataset(file_path, file_format)
    
    num_samples = min(num_samples, len(dataset))
    num_rows = (num_samples + num_cols - 1) // num_cols
    
    if figsize is None:
        figsize = (4 * num_cols, 4 * num_rows)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Sample random indices
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        sample = dataset[idx]
        points = sample['points']
        
        if dataset.dataset_type == 'multi_scene' and 'object_ids' in sample:
            plot_point_cloud_with_labels(
                points, 
                labels=sample['object_ids'],
                ax=ax,
                title=f"{title_prefix} {i+1}",
                show_axes=False
            )
        else:
            label = sample.get('label', None)
            title = f"{title_prefix} {i+1}"
            if label is not None:
                title += f" (Digit: {label})"
            plot_point_cloud(
                points,
                ax=ax,
                title=title,
                show_axes=False
            )
    
    # Hide unused axes
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    
    return fig


def visualize_simple_dataset(
    file_path: str,
    num_samples: int = 16,
    num_cols: int = 4,
    file_format: Optional[str] = None,
    output_path: Optional[str] = None,
    show_labels: bool = True
) -> plt.Figure:
    """Visualize samples from a simple MNIST point cloud dataset.
    
    Args:
        file_path: Path to dataset file
        num_samples: Number of samples to visualize
        num_cols: Number of columns in grid
        file_format: "npz" or "pkl" (auto-detected if None)
        output_path: Path to save figure
        show_labels: Whether to show digit labels
        
    Returns:
        matplotlib Figure object
    """
    return visualize_dataset_samples(
        file_path=file_path,
        num_samples=num_samples,
        num_cols=num_cols,
        file_format=file_format,
        output_path=output_path,
        title_prefix="Digit"
    )


def visualize_multi_scene_dataset(
    file_path: str,
    num_samples: int = 9,
    num_cols: int = 3,
    file_format: Optional[str] = None,
    output_path: Optional[str] = None,
    canvas_range: Tuple[float, float] = (-4, 4)
) -> plt.Figure:
    """Visualize samples from a multi-digit scene dataset.
    
    Args:
        file_path: Path to dataset file
        num_samples: Number of scenes to visualize
        num_cols: Number of columns in grid
        file_format: "npz" or "pkl" (auto-detected if None)
        output_path: Path to save figure
        canvas_range: Canvas range for axis limits
        
    Returns:
        matplotlib Figure object
    """
    dataset = MNISTPointCloudDataset(file_path, file_format)
    
    if dataset.dataset_type != 'multi_scene':
        raise ValueError("Dataset is not a multi-scene dataset")
    
    num_samples = min(num_samples, len(dataset))
    num_rows = (num_samples + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Sample random indices
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        sample = dataset[idx]
        points = sample['points']
        object_ids = sample.get('object_ids', None)
        
        plot_point_cloud_with_labels(
            points,
            labels=object_ids,
            ax=ax,
            title=f"Scene {i+1}",
            xlim=canvas_range,
            ylim=canvas_range,
            show_axes=False
        )
    
    # Hide unused axes
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    
    return fig


# ============================================================================
# Comparison and Analysis Visualizations
# ============================================================================

def plot_comparison(
    points1: np.ndarray,
    points2: np.ndarray,
    title1: str = "Original",
    title2: str = "Generated",
    figsize: Tuple[int, int] = (10, 5),
    output_path: Optional[str] = None
) -> plt.Figure:
    """Compare two point clouds side by side.
    
    Args:
        points1: First point cloud (N, 2)
        points2: Second point cloud (M, 2)
        title1: Title for first plot
        title2: Title for second plot
        figsize: Figure size
        output_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    plot_point_cloud(points1, ax=ax1, title=title1, show_axes=True)
    plot_point_cloud(points2, ax=ax2, title=title2, show_axes=True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {output_path}")
    
    return fig


def visualize_label_distribution(
    file_path: str,
    file_format: Optional[str] = None,
    output_path: Optional[str] = None
) -> plt.Figure:
    """Visualize the distribution of digit labels in a simple dataset.
    
    Args:
        file_path: Path to dataset file
        file_format: "npz" or "pkl" (auto-detected if None)
        output_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    dataset = load_mnist_point_clouds(file_path, file_format)
    
    if 'labels' not in dataset:
        raise ValueError("Dataset does not contain labels")
    
    labels = dataset['labels']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    bars = ax.bar(unique_labels, counts, color='steelblue', alpha=0.7)
    ax.set_xlabel('Digit Label')
    ax.set_ylabel('Count')
    ax.set_title('Label Distribution in Dataset')
    ax.set_xticks(unique_labels)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved label distribution to {output_path}")
    
    return fig


def visualize_scene_statistics(
    file_path: str,
    file_format: Optional[str] = None,
    output_path: Optional[str] = None
) -> plt.Figure:
    """Visualize statistics about multi-digit scenes (number of digits per scene, etc.).
    
    Args:
        file_path: Path to dataset file
        file_format: "npz" or "pkl" (auto-detected if None)
        output_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    dataset = load_mnist_point_clouds(file_path, file_format)
    
    if 'object_ids' not in dataset:
        raise ValueError("Dataset is not a multi-scene dataset")
    
    object_ids = dataset['object_ids']
    
    # Count number of unique objects per scene
    num_objects_per_scene = []
    for scene_ids in object_ids:
        num_objects_per_scene.append(len(np.unique(scene_ids)))
    
    num_objects_per_scene = np.array(num_objects_per_scene)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram of number of objects per scene
    unique_counts, counts = np.unique(num_objects_per_scene, return_counts=True)
    ax1.bar(unique_counts, counts, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Number of Digits per Scene')
    ax1.set_ylabel('Number of Scenes')
    ax1.set_title('Distribution of Digits per Scene')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add count labels
    for count_val, freq in zip(unique_counts, counts):
        ax1.text(count_val, freq, f'{freq}',
                ha='center', va='bottom')
    
    # Points per scene distribution
    points_per_scene = [len(ids) for ids in object_ids]
    ax2.hist(points_per_scene, bins=30, color='coral', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Number of Points per Scene')
    ax2.set_ylabel('Number of Scenes')
    ax2.set_title('Distribution of Points per Scene')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved scene statistics to {output_path}")
    
    return fig


# ============================================================================
# Interactive and Detailed Visualizations
# ============================================================================

def plot_digit_grid(
    file_path: str,
    digits: Optional[List[int]] = None,
    samples_per_digit: int = 5,
    file_format: Optional[str] = None,
    output_path: Optional[str] = None
) -> plt.Figure:
    """Plot a grid showing multiple samples of each digit.
    
    Args:
        file_path: Path to dataset file
        digits: List of digits to show (0-9, None = all)
        samples_per_digit: Number of samples per digit
        file_format: "npz" or "pkl" (auto-detected if None)
        output_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    dataset = load_mnist_point_clouds(file_path, file_format)
    
    if 'labels' not in dataset:
        raise ValueError("Dataset does not contain labels")
    
    points = dataset['points']
    labels = dataset['labels']
    
    if digits is None:
        digits = list(range(10))
    
    num_digits = len(digits)
    fig, axes = plt.subplots(num_digits, samples_per_digit, 
                            figsize=(2 * samples_per_digit, 2 * num_digits))
    
    if num_digits == 1:
        axes = axes.reshape(1, -1)
    
    for row, digit in enumerate(digits):
        # Find all samples of this digit
        digit_indices = np.where(labels == digit)[0]
        if len(digit_indices) == 0:
            continue
        
        # Sample random indices
        num_available = min(samples_per_digit, len(digit_indices))
        selected_indices = np.random.choice(digit_indices, num_available, replace=False)
        
        for col, idx in enumerate(selected_indices):
            ax = axes[row, col]
            plot_point_cloud(
                points[idx],
                ax=ax,
                title=f"Digit {digit}",
                show_axes=False
            )
    
    # Hide unused axes
    for row in range(num_digits):
        for col in range(samples_per_digit):
            if col >= len(selected_indices) if row == num_digits - 1 else True:
                if row < len(axes) and col < len(axes[row]):
                    axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved digit grid to {output_path}")
    
    return fig


def plot_single_scene_detailed(
    file_path: str,
    scene_idx: int = 0,
    file_format: Optional[str] = None,
    output_path: Optional[str] = None,
    canvas_range: Tuple[float, float] = (-4, 4)
) -> plt.Figure:
    """Plot a single multi-digit scene with detailed information.
    
    Args:
        file_path: Path to dataset file
        scene_idx: Index of scene to plot
        file_format: "npz" or "pkl" (auto-detected if None)
        output_path: Path to save figure
        canvas_range: Canvas range for axis limits
        
    Returns:
        matplotlib Figure object
    """
    dataset = MNISTPointCloudDataset(file_path, file_format)
    
    if dataset.dataset_type != 'multi_scene':
        raise ValueError("Dataset is not a multi-scene dataset")
    
    if scene_idx >= len(dataset):
        raise ValueError(f"Scene index {scene_idx} out of range (dataset has {len(dataset)} scenes)")
    
    sample = dataset[scene_idx]
    points = sample['points']
    object_ids = sample.get('object_ids', None)
    
    num_objects = len(np.unique(object_ids)) if object_ids is not None else 1
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    plot_point_cloud_with_labels(
        points,
        labels=object_ids,
        ax=ax,
        title=f"Scene {scene_idx} - {num_objects} digits, {len(points)} points",
        xlim=canvas_range,
        ylim=canvas_range,
        show_axes=True
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved scene visualization to {output_path}")
    
    return fig


# ============================================================================
# Quick Visualization Functions
# ============================================================================

def quick_visualize(
    file_path: str,
    file_format: Optional[str] = None,
    output_path: Optional[str] = None,
    num_samples: int = 16
) -> plt.Figure:
    """Quick visualization of a dataset (auto-detects dataset type).
    
    Args:
        file_path: Path to dataset file
        file_format: "npz" or "pkl" (auto-detected if None)
        output_path: Path to save figure
        num_samples: Number of samples to show
        
    Returns:
        matplotlib Figure object
    """
    dataset = MNISTPointCloudDataset(file_path, file_format)
    
    if dataset.dataset_type == 'simple':
        return visualize_simple_dataset(
            file_path=file_path,
            num_samples=num_samples,
            file_format=file_format,
            output_path=output_path
        )
    else:  # multi_scene
        return visualize_multi_scene_dataset(
            file_path=file_path,
            num_samples=num_samples,
            file_format=file_format,
            output_path=output_path
        )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize MNIST point cloud datasets")
    parser.add_argument("file_path", type=str, help="Path to dataset file")
    parser.add_argument("--file_format", type=str, default=None, choices=['npz', 'pkl'],
                       help="File format (auto-detected if not specified)")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Path to save visualization (if not specified, displays interactively)")
    parser.add_argument("--num_samples", type=int, default=16,
                       help="Number of samples to visualize")
    parser.add_argument("--mode", type=str, default="quick",
                       choices=['quick', 'simple', 'multi_scene', 'labels', 'statistics', 'grid'],
                       help="Visualization mode")
    parser.add_argument("--scene_idx", type=int, default=0,
                       help="Scene index (for detailed scene visualization)")
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        quick_visualize(
            file_path=args.file_path,
            file_format=args.file_format,
            output_path=args.output_path,
            num_samples=args.num_samples
        )
    elif args.mode == 'simple':
        visualize_simple_dataset(
            file_path=args.file_path,
            num_samples=args.num_samples,
            file_format=args.file_format,
            output_path=args.output_path
        )
    elif args.mode == 'multi_scene':
        visualize_multi_scene_dataset(
            file_path=args.file_path,
            num_samples=args.num_samples,
            file_format=args.file_format,
            output_path=args.output_path
        )
    elif args.mode == 'labels':
        visualize_label_distribution(
            file_path=args.file_path,
            file_format=args.file_format,
            output_path=args.output_path
        )
    elif args.mode == 'statistics':
        visualize_scene_statistics(
            file_path=args.file_path,
            file_format=args.file_format,
            output_path=args.output_path
        )
    elif args.mode == 'grid':
        plot_digit_grid(
            file_path=args.file_path,
            file_format=args.file_format,
            output_path=args.output_path
        )
    
    if args.output_path is None:
        plt.show()
    else:
        print(f"Visualization saved to {args.output_path}")

