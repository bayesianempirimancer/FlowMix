"""
Script to generate all MNIST point cloud datasets and their visualizations.
"""

import sys
import os
sys.path.insert(0, '.')

from src.data.mnist_point_cloud_data import (
    generate_simple_mnist_point_clouds,
    generate_simple_mnist_rotated,
    generate_simple_mnist_translated,
    generate_simple_mnist_rotated_translated,
    generate_multi_digit_scenes_normal,
    generate_multi_digit_scenes_rotated,
    generate_multi_digit_scenes_rotated_scaled
)
from src.data.visualize_mnist_data import (
    visualize_simple_dataset,
    visualize_multi_scene_dataset,
    visualize_label_distribution,
    visualize_scene_statistics
)

def main():
    print("=" * 80)
    print("Generating All MNIST Point Cloud Datasets")
    print("=" * 80)
    print()
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/visualizations", exist_ok=True)
    
    # Single digit datasets
    print("=" * 80)
    print("SINGLE DIGIT DATASETS")
    print("=" * 80)
    print()
    
    # 1. Single digits (no transformations)
    print("1. Generating single digits (no transformations)...")
    print("-" * 80)
    single_path = "data/mnist_2d_single.npz"
    generate_simple_mnist_point_clouds(
        num_points=500,
        output_path=single_path,
        file_format="npz",
        force_overwrite=False,
        seed=42
    )
    print()
    print("   Creating visualizations...")
    visualize_simple_dataset(
        file_path=single_path,
        num_samples=16,
        num_cols=4,
        file_format="npz",
        output_path="data/visualizations/single_samples.png"
    )
    visualize_label_distribution(
        file_path=single_path,
        file_format="npz",
        output_path="data/visualizations/single_labels.png"
    )
    print("   ✓ Single dataset visualizations saved")
    print()
    
    # 2. Single digits with rotation only
    print("2. Generating single digits (rotated only)...")
    print("-" * 80)
    single_rotated_path = "data/mnist_2d_single_rotated.npz"
    generate_simple_mnist_rotated(
        num_points=500,
        rotation_range=(0, 2 * 3.14159),
        output_path=single_rotated_path,
        file_format="npz",
        force_overwrite=False,
        seed=42
    )
    print()
    print("   Creating visualizations...")
    visualize_simple_dataset(
        file_path=single_rotated_path,
        num_samples=16,
        num_cols=4,
        file_format="npz",
        output_path="data/visualizations/single_rotated_samples.png"
    )
    print("   ✓ Single rotated dataset visualizations saved")
    print()
    
    # 3. Single digits with translation only
    print("3. Generating single digits (translated only)...")
    print("-" * 80)
    single_translated_path = "data/mnist_2d_single_translated.npz"
    generate_simple_mnist_translated(
        num_points=500,
        translation_range=(-1.5, 1.5),
        output_path=single_translated_path,
        file_format="npz",
        force_overwrite=False,
        seed=42
    )
    print()
    print("   Creating visualizations...")
    visualize_simple_dataset(
        file_path=single_translated_path,
        num_samples=16,
        num_cols=4,
        file_format="npz",
        output_path="data/visualizations/single_translated_samples.png"
    )
    print("   ✓ Single translated dataset visualizations saved")
    print()
    
    # 4. Single digits with rotation and translation
    print("4. Generating single digits (rotated and translated)...")
    print("-" * 80)
    single_rotated_translated_path = "data/mnist_2d_single_rotated_translated.npz"
    generate_simple_mnist_rotated_translated(
        num_points=500,
        rotation_range=(0, 2 * 3.14159),
        translation_range=(-1.5, 1.5),
        output_path=single_rotated_translated_path,
        file_format="npz",
        force_overwrite=False,
        seed=42
    )
    print()
    print("   Creating visualizations...")
    visualize_simple_dataset(
        file_path=single_rotated_translated_path,
        num_samples=16,
        num_cols=4,
        file_format="npz",
        output_path="data/visualizations/single_rotated_translated_samples.png"
    )
    print("   ✓ Single rotated/translated dataset visualizations saved")
    print()
    
    # Multi-digit scene datasets
    print("=" * 80)
    print("MULTI-DIGIT SCENE DATASETS")
    print("=" * 80)
    print()
    
    # 5. Multi-digit scenes (no transformations)
    print("5. Generating multi-digit scenes (no transformations)...")
    print("-" * 80)
    multi_path = "data/multi_mnist_2d.npz"
    generate_multi_digit_scenes_normal(
        num_scenes=20000,
        points_per_digit=500,
        min_digits=1,
        max_digits=4,
        canvas_range=(-4, 4),
        output_path=multi_path,
        file_format="npz",
        force_overwrite=False,
        seed=42
    )
    print()
    print("   Creating visualizations...")
    visualize_multi_scene_dataset(
        file_path=multi_path,
        num_samples=9,
        num_cols=3,
        file_format="npz",
        output_path="data/visualizations/multi_samples.png"
    )
    visualize_scene_statistics(
        file_path=multi_path,
        file_format="npz",
        output_path="data/visualizations/multi_statistics.png"
    )
    print("   ✓ Multi dataset visualizations saved")
    print()
    
    # 6. Multi-digit scenes with rotation only
    print("6. Generating multi-digit scenes (rotated only)...")
    print("-" * 80)
    multi_rotated_path = "data/multi_mnist_2d_rotated.npz"
    generate_multi_digit_scenes_rotated(
        num_scenes=20000,
        points_per_digit=500,
        min_digits=1,
        max_digits=4,
        canvas_range=(-4, 4),
        rotation_range=(0, 2 * 3.14159),
        output_path=multi_rotated_path,
        file_format="npz",
        force_overwrite=False,
        seed=42
    )
    print()
    print("   Creating visualizations...")
    visualize_multi_scene_dataset(
        file_path=multi_rotated_path,
        num_samples=9,
        num_cols=3,
        file_format="npz",
        output_path="data/visualizations/multi_rotated_samples.png"
    )
    visualize_scene_statistics(
        file_path=multi_rotated_path,
        file_format="npz",
        output_path="data/visualizations/multi_rotated_statistics.png"
    )
    print("   ✓ Multi rotated dataset visualizations saved")
    print()
    
    # 7. Multi-digit scenes with rotation and scaling
    print("7. Generating multi-digit scenes (rotated and scaled)...")
    print("-" * 80)
    multi_rotated_scaled_path = "data/multi_mnist_2d_rotated_scaled.npz"
    generate_multi_digit_scenes_rotated_scaled(
        num_scenes=20000,
        points_per_digit=500,
        min_digits=1,
        max_digits=4,
        canvas_range=(-4, 4),
        rotation_range=(0, 2 * 3.14159),
        scale_range=(0.5, 1.5),
        output_path=multi_rotated_scaled_path,
        file_format="npz",
        force_overwrite=False,
        seed=42
    )
    print()
    print("   Creating visualizations...")
    visualize_multi_scene_dataset(
        file_path=multi_rotated_scaled_path,
        num_samples=9,
        num_cols=3,
        file_format="npz",
        output_path="data/visualizations/multi_rotated_scaled_samples.png"
    )
    visualize_scene_statistics(
        file_path=multi_rotated_scaled_path,
        file_format="npz",
        output_path="data/visualizations/multi_rotated_scaled_statistics.png"
    )
    print("   ✓ Multi rotated/scaled dataset visualizations saved")
    print()
    
    print("=" * 80)
    print("All datasets and visualizations generated successfully!")
    print("=" * 80)
    print()
    print("Generated datasets:")
    print("  Single digit datasets:")
    print(f"    1. {single_path}")
    print(f"    2. {single_rotated_path}")
    print(f"    3. {single_translated_path}")
    print(f"    4. {single_rotated_translated_path}")
    print("  Multi-digit scene datasets:")
    print(f"    5. {multi_path}")
    print(f"    6. {multi_rotated_path}")
    print(f"    7. {multi_rotated_scaled_path}")
    print()
    print("Visualizations saved to: data/visualizations/")

if __name__ == "__main__":
    main()

