"""
Consolidated MNIST Point Cloud Data Generation and Loading

This module provides functions to generate various MNIST point cloud datasets:
1. Simple MNIST point clouds: Single digits with N points per digit
2. Multi-digit scenes: Scenes with multiple digits (normal or rotated/scaled)

All datasets can be saved as either .pkl or .npz files and include modern ML data loaders.
"""

import os
import pickle
import numpy as np
import jax.numpy as jnp
import tensorflow_datasets as tfds
from typing import Tuple, Optional, Dict, Union, List
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


# ============================================================================
# Core MNIST Loading and Point Cloud Conversion
# ============================================================================

class MNISTPointCloudGenerator:
    """Core class for generating MNIST point clouds."""
    
    def __init__(self, seed: int = 42):
        """Initialize the generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.mnist_images = None
        self.mnist_labels = None
        self._load_mnist()
    
    def _load_mnist(self):
        """Load MNIST dataset from tensorflow_datasets."""
        if self.mnist_images is None:
            print("Loading MNIST dataset...")
            ds = tfds.load('mnist', split='train', shuffle_files=False)
            self.mnist_images = []
            self.mnist_labels = []
            for ex in tqdm(ds.as_numpy_iterator(), desc="Loading MNIST", total=60000):
                self.mnist_images.append(ex['image'][:, :, 0])  # (28, 28)
                self.mnist_labels.append(ex['label'])
            self.mnist_images = np.array(self.mnist_images)  # (60000, 28, 28)
            self.mnist_labels = np.array(self.mnist_labels)  # (60000,)
            print(f"Loaded {len(self.mnist_images)} MNIST images")
    
    def digit_to_point_cloud(self, img: np.ndarray, num_points: int, 
                             jitter: bool = True) -> np.ndarray:
        """Convert a single MNIST digit image to a point cloud.
        
        Args:
            img: MNIST image array (28, 28)
            num_points: Number of points to generate
            jitter: Whether to add Gaussian jitter for sub-pixel accuracy
            
        Returns:
            points: Point cloud array (num_points, 2) in [-1, 1] range
        """
        # Find pixel locations where digit is present
        mask = img > 128
        y, x = np.where(mask)
        
        if len(y) == 0:
            # Empty image: return random points
            return self.rng.normal(0, 0.5, (num_points, 2))
        
        # Sample indices with replacement
        indices = self.rng.choice(len(x), num_points, replace=True)
        
        y_sel = y[indices].astype(np.float32)
        x_sel = x[indices].astype(np.float32)
        
        # Add jitter for sub-pixel accuracy
        if jitter:
            jitter_std = 1.0 / np.sqrt(12.0)  # Matches uniform distribution variance
            y_sel += self.rng.normal(0, jitter_std, size=num_points)
            x_sel += self.rng.normal(0, jitter_std, size=num_points)
        
        # Normalize to [-1, 1] range (centered at origin)
        y_norm = -(y_sel - 13.5) / 13.5
        x_norm = (x_sel - 13.5) / 13.5
        
        points = np.stack([x_norm, y_norm], axis=1)  # (num_points, 2)
        return points
    
    def apply_transform(self, points: np.ndarray, rotation: Optional[float] = None,
                       scale: Optional[float] = None, translation: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply rotation, scale, and translation to a point cloud.
        
        Args:
            points: Point cloud (N, 2)
            rotation: Rotation angle in radians (2D rotation)
            scale: Scale factor
            translation: Translation vector (2,)
            
        Returns:
            transformed_points: Transformed point cloud (N, 2)
        """
        transformed = points.copy()
        
        # Apply rotation (2D rotation matrix)
        if rotation is not None:
            cos_r = np.cos(rotation)
            sin_r = np.sin(rotation)
            rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
            transformed = transformed @ rot_matrix.T
        
        # Apply scale
        if scale is not None:
            transformed = transformed * scale
        
        # Apply translation
        if translation is not None:
            transformed = transformed + translation
        
        return transformed


# ============================================================================
# Dataset Generation Functions
# ============================================================================

def generate_simple_mnist_point_clouds(
    num_points: int = 500,
    output_path: str = "data/mnist_2d_single.npz",
    file_format: str = "npz",
    force_overwrite: bool = False,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """Generate simple MNIST point clouds: one point cloud per digit (no transformations).
    
    Args:
        num_points: Number of points per digit
        output_path: Path to save the dataset
        file_format: "npz" or "pkl"
        force_overwrite: If False, skip generation if file exists
        seed: Random seed
        
    Returns:
        Dictionary containing 'points' (N, num_points, 2) and 'labels' (N,)
    """
    # Check if file exists
    if os.path.exists(output_path) and not force_overwrite:
        print(f"Dataset already exists at {output_path}. Use force_overwrite=True to regenerate.")
        return load_mnist_point_clouds(output_path, file_format)
    
    print(f"Generating simple MNIST point clouds ({num_points} points per digit)...")
    generator = MNISTPointCloudGenerator(seed=seed)
    
    all_points = []
    all_labels = []
    
    for i in tqdm(range(len(generator.mnist_images)), desc="Converting digits"):
        img = generator.mnist_images[i]
        label = generator.mnist_labels[i]
        
        points = generator.digit_to_point_cloud(img, num_points)
        
        all_points.append(points)
        all_labels.append(label)
    
    # Stack into arrays
    points_array = np.array(all_points)  # (60000, num_points, 2)
    labels_array = np.array(all_labels)  # (60000,)
    
    dataset = {
        'points': points_array,
        'labels': labels_array
    }
    
    # Save dataset
    _save_dataset(dataset, output_path, file_format)
    print(f"Saved {len(points_array)} point clouds to {output_path}")
    
    return dataset


def generate_multi_digit_scenes(
    num_scenes: int = 20000,
    points_per_digit: int = 500,
    min_digits: int = 1,
    max_digits: int = 4,
    canvas_range: Tuple[float, float] = (-4, 4),
    use_rotation: bool = False,
    use_scaling: bool = False,
    rotation_range: Tuple[float, float] = (0, 2 * np.pi),
    scale_range: Tuple[float, float] = (0.5, 1.5),
    output_path: str = "data/multi_mnist_2d.npz",
    file_format: str = "npz",
    force_overwrite: bool = False,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """Generate multi-digit scenes with multiple MNIST digits.
    
    Args:
        num_scenes: Number of scenes to generate
        points_per_digit: Number of points per digit
        min_digits: Minimum number of digits per scene
        max_digits: Maximum number of digits per scene
        canvas_range: Range for digit placement (min, max)
        use_rotation: Whether to randomly rotate digits
        use_scaling: Whether to randomly scale digits
        rotation_range: (min_angle, max_angle) in radians
        scale_range: (min_scale, max_scale)
        output_path: Path to save the dataset
        file_format: "npz" or "pkl"
        force_overwrite: If False, skip generation if file exists
        seed: Random seed
        
    Returns:
        Dictionary containing 'points' (ragged array) and 'object_ids' (ragged array)
    """
    # Check if file exists
    if os.path.exists(output_path) and not force_overwrite:
        print(f"Dataset already exists at {output_path}. Use force_overwrite=True to regenerate.")
        return load_mnist_point_clouds(output_path, file_format)
    
    print(f"Generating multi-digit scenes (with rotation={use_rotation}, scaling={use_scaling})...")
    generator = MNISTPointCloudGenerator(seed=seed)
    
    all_points = []
    all_object_ids = []
    all_rotations = []
    all_translations = []
    all_scales = []
    all_digit_labels = []  # Store which digit (0-9) each object is
    
    for scene_idx in tqdm(range(num_scenes), desc="Generating scenes"):
        # Random number of digits in this scene
        num_digits = generator.rng.randint(min_digits, max_digits + 1)
        
        scene_points = []
        scene_ids = []
        scene_rotations = []
        scene_translations = []
        scene_scales = []
        scene_digit_labels = []
        
        for obj_idx in range(num_digits):
            # Sample a random digit
            img_idx = generator.rng.randint(0, len(generator.mnist_images))
            img = generator.mnist_images[img_idx]
            digit_label = generator.mnist_labels[img_idx]
            
            # Convert to point cloud
            points = generator.digit_to_point_cloud(img, points_per_digit)
            
            # Apply transformations
            rotation = None
            scale = None
            if use_rotation:
                rotation = generator.rng.uniform(rotation_range[0], rotation_range[1])
            else:
                rotation = 0.0  # Store 0.0 if no rotation
            if use_scaling:
                scale = generator.rng.uniform(scale_range[0], scale_range[1])
            else:
                scale = 1.0  # Store 1.0 if no scaling
            
            # Random translation within canvas
            border = 1.0  # Keep digits away from edges
            t_min = canvas_range[0] + border
            t_max = canvas_range[1] - border
            translation = generator.rng.uniform(t_min, t_max, size=(2,))
            
            # Apply transformations
            points = generator.apply_transform(points, rotation=rotation, 
                                             scale=scale, translation=translation)
            
            scene_points.append(points)
            scene_ids.append(np.full((points_per_digit,), obj_idx, dtype=np.int32))
            scene_rotations.append(rotation)
            scene_translations.append(translation)
            scene_scales.append(scale)
            scene_digit_labels.append(digit_label)
        
        # Concatenate all digits in scene
        scene_points = np.concatenate(scene_points, axis=0)  # (total_points, 2)
        scene_ids = np.concatenate(scene_ids, axis=0)  # (total_points,)
        
        all_points.append(scene_points)
        all_object_ids.append(scene_ids)
        all_rotations.append(np.array(scene_rotations))  # (num_digits,)
        all_translations.append(np.array(scene_translations))  # (num_digits, 2)
        all_scales.append(np.array(scene_scales))  # (num_digits,)
        all_digit_labels.append(np.array(scene_digit_labels))  # (num_digits,)
    
    # Save as ragged arrays (object arrays)
    dataset = {
        'points': np.array(all_points, dtype=object),
        'object_ids': np.array(all_object_ids, dtype=object),
        'rotations': np.array(all_rotations, dtype=object),
        'translations': np.array(all_translations, dtype=object),
        'scales': np.array(all_scales, dtype=object),
        'digit_labels': np.array(all_digit_labels, dtype=object)  # Which digit (0-9) each object is
    }
    
    # Save dataset
    _save_dataset(dataset, output_path, file_format)
    print(f"Saved {num_scenes} scenes to {output_path}")
    
    return dataset


def generate_multi_digit_scenes_normal(
    num_scenes: int = 20000,
    points_per_digit: int = 500,
    min_digits: int = 1,
    max_digits: int = 4,
    canvas_range: Tuple[float, float] = (-4, 4),
    output_path: str = "data/multi_mnist_2d.npz",
    file_format: str = "npz",
    force_overwrite: bool = False,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """Generate multi-digit scenes without rotation or scaling (normal digits).
    
    Convenience wrapper around generate_multi_digit_scenes with use_rotation=False, use_scaling=False.
    """
    return generate_multi_digit_scenes(
        num_scenes=num_scenes,
        points_per_digit=points_per_digit,
        min_digits=min_digits,
        max_digits=max_digits,
        canvas_range=canvas_range,
        use_rotation=False,
        use_scaling=False,
        output_path=output_path,
        file_format=file_format,
        force_overwrite=force_overwrite,
        seed=seed
    )


def generate_multi_digit_scenes_rotated(
    num_scenes: int = 20000,
    points_per_digit: int = 500,
    min_digits: int = 1,
    max_digits: int = 4,
    canvas_range: Tuple[float, float] = (-4, 4),
    rotation_range: Tuple[float, float] = (0, 2 * np.pi),
    output_path: str = "data/multi_mnist_2d_rotated.pkl",
    file_format: str = "pkl",
    force_overwrite: bool = False,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """Generate multi-digit scenes with rotation but no scaling.
    
    Convenience wrapper around generate_multi_digit_scenes with use_rotation=True, use_scaling=False.
    """
    return generate_multi_digit_scenes(
        num_scenes=num_scenes,
        points_per_digit=points_per_digit,
        min_digits=min_digits,
        max_digits=max_digits,
        canvas_range=canvas_range,
        use_rotation=True,
        use_scaling=False,
        rotation_range=rotation_range,
        output_path=output_path,
        file_format=file_format,
        force_overwrite=force_overwrite,
        seed=seed
    )


def generate_multi_digit_scenes_rotated_scaled(
    num_scenes: int = 20000,
    points_per_digit: int = 500,
    min_digits: int = 1,
    max_digits: int = 4,
    canvas_range: Tuple[float, float] = (-4, 4),
    rotation_range: Tuple[float, float] = (0, 2 * np.pi),
    scale_range: Tuple[float, float] = (0.5, 1.5),
    output_path: str = "data/multi_mnist_2d_rotated_scaled.pkl",
    file_format: str = "pkl",
    force_overwrite: bool = False,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """Generate multi-digit scenes with rotation and scaling.
    
    Convenience wrapper around generate_multi_digit_scenes with use_rotation=True, use_scaling=True.
    """
    return generate_multi_digit_scenes(
        num_scenes=num_scenes,
        points_per_digit=points_per_digit,
        min_digits=min_digits,
        max_digits=max_digits,
        canvas_range=canvas_range,
        use_rotation=True,
        use_scaling=True,
        rotation_range=rotation_range,
        scale_range=scale_range,
        output_path=output_path,
        file_format=file_format,
        force_overwrite=force_overwrite,
        seed=seed
    )


def generate_full_mnist_dataset(
    num_points: int = 500,
    output_path: str = "data/mnist_2d_full_dataset.npz",
    file_format: str = "npz",
    force_overwrite: bool = False,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """Generate the full MNIST dataset (all 60k digits) in the format expected by training scripts.
    
    This is a convenience function that generates the standard "full dataset" format used
    by the training code. It's equivalent to generate_simple_mnist_point_clouds but with
    a default path matching the training script expectations.
    
    Args:
        num_points: Number of points per digit
        output_path: Path to save the dataset (default matches training script)
        file_format: "npz" or "pkl"
        force_overwrite: If False, skip generation if file exists
        seed: Random seed
        
    Returns:
        Dictionary containing 'points' (60000, num_points, 2)
    """
    return generate_simple_mnist_point_clouds(
        num_points=num_points,
        output_path=output_path,
        file_format=file_format,
        force_overwrite=force_overwrite,
        seed=seed
    )


def generate_simple_mnist_rotated(
    num_points: int = 500,
    rotation_range: Tuple[float, float] = (0, 2 * np.pi),
    output_path: str = "data/mnist_2d_single_rotated.npz",
    file_format: str = "npz",
    force_overwrite: bool = False,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """Generate simple MNIST point clouds with random rotation applied (no translation).
    
    Each digit is individually rotated, creating a dataset of rotated single digits.
    
    Args:
        num_points: Number of points per digit
        rotation_range: (min_angle, max_angle) in radians for rotation
        output_path: Path to save the dataset
        file_format: "npz" or "pkl"
        force_overwrite: If False, skip generation if file exists
        seed: Random seed
        
    Returns:
        Dictionary containing 'points' (N, num_points, 2), 'labels' (N,), and 'rotations' (N,)
    """
    # Check if file exists
    if os.path.exists(output_path) and not force_overwrite:
        print(f"Dataset already exists at {output_path}. Use force_overwrite=True to regenerate.")
        return load_mnist_point_clouds(output_path, file_format)
    
    print(f"Generating simple MNIST point clouds with rotation...")
    print(f"  Rotation range: [{rotation_range[0]:.2f}, {rotation_range[1]:.2f}] radians")
    generator = MNISTPointCloudGenerator(seed=seed)
    
    all_points = []
    all_labels = []
    all_rotations = []
    
    for i in tqdm(range(len(generator.mnist_images)), desc="Converting digits"):
        img = generator.mnist_images[i]
        label = generator.mnist_labels[i]
        
        # Convert to point cloud
        points = generator.digit_to_point_cloud(img, num_points)
        
        # Apply random rotation
        rotation = generator.rng.uniform(rotation_range[0], rotation_range[1])
        
        # Apply transformation
        points = generator.apply_transform(points, rotation=rotation, translation=None)
        
        all_points.append(points)
        all_labels.append(label)
        all_rotations.append(rotation)
    
    # Stack into arrays
    points_array = np.array(all_points)  # (60000, num_points, 2)
    labels_array = np.array(all_labels)  # (60000,)
    rotations_array = np.array(all_rotations)  # (60000,)
    
    dataset = {
        'points': points_array,
        'labels': labels_array,
        'rotations': rotations_array
    }
    
    # Save dataset
    _save_dataset(dataset, output_path, file_format)
    print(f"Saved {len(points_array)} point clouds to {output_path}")
    
    return dataset


def generate_simple_mnist_translated(
    num_points: int = 500,
    translation_range: Tuple[float, float] = (-1.5, 1.5),
    output_path: str = "data/mnist_2d_single_translated.npz",
    file_format: str = "npz",
    force_overwrite: bool = False,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """Generate simple MNIST point clouds with random translation applied (no rotation).
    
    Each digit is individually translated, creating a dataset of translated single digits.
    
    Args:
        num_points: Number of points per digit
        translation_range: (min_translation, max_translation) for x and y translation
        output_path: Path to save the dataset
        file_format: "npz" or "pkl"
        force_overwrite: If False, skip generation if file exists
        seed: Random seed
        
    Returns:
        Dictionary containing 'points' (N, num_points, 2), 'labels' (N,), and 'translations' (N, 2)
    """
    # Check if file exists
    if os.path.exists(output_path) and not force_overwrite:
        print(f"Dataset already exists at {output_path}. Use force_overwrite=True to regenerate.")
        return load_mnist_point_clouds(output_path, file_format)
    
    print(f"Generating simple MNIST point clouds with translation...")
    print(f"  Translation range: [{translation_range[0]:.2f}, {translation_range[1]:.2f}]")
    generator = MNISTPointCloudGenerator(seed=seed)
    
    all_points = []
    all_labels = []
    all_translations = []
    
    for i in tqdm(range(len(generator.mnist_images)), desc="Converting digits"):
        img = generator.mnist_images[i]
        label = generator.mnist_labels[i]
        
        # Convert to point cloud
        points = generator.digit_to_point_cloud(img, num_points)
        
        # Apply random translation
        translation = generator.rng.uniform(
            translation_range[0], translation_range[1], size=(2,)
        )
        
        # Apply transformation
        points = generator.apply_transform(points, rotation=None, translation=translation)
        
        all_points.append(points)
        all_labels.append(label)
        all_translations.append(translation)
    
    # Stack into arrays
    points_array = np.array(all_points)  # (60000, num_points, 2)
    labels_array = np.array(all_labels)  # (60000,)
    translations_array = np.array(all_translations)  # (60000, 2)
    
    dataset = {
        'points': points_array,
        'labels': labels_array,
        'translations': translations_array
    }
    
    # Save dataset
    _save_dataset(dataset, output_path, file_format)
    print(f"Saved {len(points_array)} point clouds to {output_path}")
    
    return dataset


def generate_simple_mnist_rotated_translated(
    num_points: int = 500,
    rotation_range: Tuple[float, float] = (0, 2 * np.pi),
    translation_range: Tuple[float, float] = (-1.5, 1.5),
    output_path: str = "data/mnist_2d_single_rotated_translated.npz",
    file_format: str = "npz",
    force_overwrite: bool = False,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """Generate simple MNIST point clouds with random rotation and translation applied.
    
    Each digit is individually rotated and translated, creating a dataset of transformed
    single digits. This is useful for training models that need to be robust to these
    transformations.
    
    Args:
        num_points: Number of points per digit
        rotation_range: (min_angle, max_angle) in radians for rotation
        translation_range: (min_translation, max_translation) for x and y translation
        output_path: Path to save the dataset
        file_format: "npz" or "pkl"
        force_overwrite: If False, skip generation if file exists
        seed: Random seed
        
    Returns:
        Dictionary containing 'points' (N, num_points, 2) and 'labels' (N,)
    """
    # Check if file exists
    if os.path.exists(output_path) and not force_overwrite:
        print(f"Dataset already exists at {output_path}. Use force_overwrite=True to regenerate.")
        return load_mnist_point_clouds(output_path, file_format)
    
    print(f"Generating simple MNIST point clouds with rotation and translation...")
    print(f"  Rotation range: [{rotation_range[0]:.2f}, {rotation_range[1]:.2f}] radians")
    print(f"  Translation range: [{translation_range[0]:.2f}, {translation_range[1]:.2f}]")
    generator = MNISTPointCloudGenerator(seed=seed)
    
    all_points = []
    all_labels = []
    all_rotations = []
    all_translations = []
    
    for i in tqdm(range(len(generator.mnist_images)), desc="Converting digits"):
        img = generator.mnist_images[i]
        label = generator.mnist_labels[i]
        
        # Convert to point cloud
        points = generator.digit_to_point_cloud(img, num_points)
        
        # Apply random rotation and translation
        rotation = generator.rng.uniform(rotation_range[0], rotation_range[1])
        translation = generator.rng.uniform(
            translation_range[0], translation_range[1], size=(2,)
        )
        
        # Apply transformations
        points = generator.apply_transform(
            points, rotation=rotation, translation=translation
        )
        
        all_points.append(points)
        all_labels.append(label)
        all_rotations.append(rotation)
        all_translations.append(translation)
    
    # Stack into arrays
    points_array = np.array(all_points)  # (60000, num_points, 2)
    labels_array = np.array(all_labels)  # (60000,)
    rotations_array = np.array(all_rotations)  # (60000,)
    translations_array = np.array(all_translations)  # (60000, 2)
    
    dataset = {
        'points': points_array,
        'labels': labels_array,
        'rotations': rotations_array,
        'translations': translations_array
    }
    
    # Save dataset
    _save_dataset(dataset, output_path, file_format)
    print(f"Saved {len(points_array)} point clouds to {output_path}")
    
    return dataset


# ============================================================================
# Dataset Saving and Loading
# ============================================================================

def _save_dataset(dataset: Dict[str, np.ndarray], output_path: str, file_format: str):
    """Save dataset to file.
    
    Args:
        dataset: Dictionary of arrays to save
        output_path: Path to save file
        file_format: "npz" or "pkl"
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    if file_format.lower() == "npz":
        np.savez(output_path, **dataset)
    elif file_format.lower() == "pkl":
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f)
    else:
        raise ValueError(f"Unsupported file format: {file_format}. Use 'npz' or 'pkl'.")


def load_mnist_point_clouds(
    file_path: str,
    file_format: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """Load MNIST point cloud dataset from file.
    
    Args:
        file_path: Path to dataset file
        file_format: "npz" or "pkl" (auto-detected from extension if None)
        
    Returns:
        Dictionary containing dataset arrays
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    # Auto-detect format from extension
    if file_format is None:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.npz':
            file_format = 'npz'
        elif ext == '.pkl':
            file_format = 'pkl'
        else:
            raise ValueError(f"Could not auto-detect file format from extension {ext}. Specify file_format.")
    
    if file_format.lower() == "npz":
        data = np.load(file_path, allow_pickle=True)
        dataset = {key: data[key] for key in data.keys()}
    elif file_format.lower() == "pkl":
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_format}. Use 'npz' or 'pkl'.")
    
    return dataset


def load_mnist_point_clouds_jax(
    file_path: str,
    max_samples: Optional[int] = None,
    seed: int = 42,
    file_format: Optional[str] = None
) -> jnp.ndarray:
    """Load MNIST point clouds in JAX format (compatible with existing training code).
    
    This function provides compatibility with the existing load_all_digits_data function.
    
    Args:
        file_path: Path to dataset file
        max_samples: Maximum number of samples to use (None = use all)
        seed: Random seed for shuffling
        file_format: "npz" or "pkl" (auto-detected from extension if None)
        
    Returns:
        X: JAX array of point clouds (num_samples, num_points, 2)
    """
    dataset = load_mnist_point_clouds(file_path, file_format)
    
    if 'points' not in dataset:
        raise ValueError("Dataset must contain 'points' key")
    
    points = dataset['points']
    
    # Handle ragged arrays (multi-scene datasets)
    if points.dtype == object:
        raise ValueError("This function only supports simple point cloud datasets (not multi-scene). "
                       "Use load_mnist_point_clouds() for multi-scene datasets.")
    
    # Optionally limit number of samples
    if max_samples is not None and len(points) > max_samples:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(points), max_samples, replace=False)
        points = points[indices]
    
    return jnp.array(points)


# ============================================================================
# Modern ML Data Loaders
# ============================================================================

class MNISTPointCloudDataset:
    """PyTorch-style dataset for MNIST point clouds.
    
    Compatible with PyTorch DataLoader and can be adapted for JAX.
    """
    
    def __init__(self, file_path: str, file_format: Optional[str] = None, 
                 transform: Optional[callable] = None):
        """Initialize dataset.
        
        Args:
            file_path: Path to dataset file
            file_format: "npz" or "pkl" (auto-detected if None)
            transform: Optional transform function to apply to samples
        """
        self.data = load_mnist_point_clouds(file_path, file_format)
        self.transform = transform
        
        # Determine dataset type
        if 'points' in self.data and 'labels' in self.data:
            self.dataset_type = 'simple'  # Simple point clouds
            self.points = self.data['points']
            self.labels = self.data['labels']
            self.num_samples = len(self.points)
        elif 'points' in self.data and 'object_ids' in self.data:
            self.dataset_type = 'multi_scene'  # Multi-digit scenes
            self.points = self.data['points']
            self.object_ids = self.data['object_ids']
            self.num_samples = len(self.points)
        else:
            raise ValueError("Unknown dataset format. Expected 'points' and 'labels' or 'object_ids'.")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with 'points' and optionally 'label' or 'object_ids'
        """
        if self.dataset_type == 'simple':
            sample = {
                'points': self.points[idx].astype(np.float32),
                'label': self.labels[idx]
            }
        else:  # multi_scene
            sample = {
                'points': self.points[idx].astype(np.float32),
                'object_ids': self.object_ids[idx]
            }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def create_data_loader(
    file_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    train_split: float = 0.9,
    seed: int = 42,
    file_format: Optional[str] = None,
    use_pytorch: bool = False
):
    """Create a data loader for training.
    
    Args:
        file_path: Path to dataset file
        batch_size: Batch size
        shuffle: Whether to shuffle data
        train_split: Train/test split ratio
        seed: Random seed
        file_format: "npz" or "pkl" (auto-detected if None)
        use_pytorch: If True, return PyTorch DataLoader; else return JAX-compatible generator
        
    Returns:
        train_loader, test_loader: Data loaders for train and test sets
    """
    dataset = MNISTPointCloudDataset(file_path, file_format)
    
    # Split into train/test
    n = len(dataset)
    n_train = int(n * train_split)
    
    rng = np.random.RandomState(seed)
    indices = np.arange(n)
    if shuffle:
        rng.shuffle(indices)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    if use_pytorch:
        try:
            import torch
            from torch.utils.data import DataLoader, Subset
            
            train_dataset = Subset(dataset, train_indices)
            test_dataset = Subset(dataset, test_indices)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            return train_loader, test_loader
        except ImportError:
            print("PyTorch not available, falling back to JAX-compatible generator")
            use_pytorch = False
    
    # JAX-compatible generator
    def get_batches(indices, batch_size, shuffle_batches=True):
        """Generator for batches."""
        n_samples = len(indices)
        batch_indices = np.arange(n_samples)
        if shuffle_batches:
            rng_batch = np.random.RandomState(seed)
            rng_batch.shuffle(batch_indices)
        
        for i in range(0, n_samples, batch_size):
            batch_idx = batch_indices[i:i+batch_size]
            batch_samples = [dataset[indices[j]] for j in batch_idx]
            
            # Stack into batches
            if dataset.dataset_type == 'simple':
                batch = {
                    'points': np.stack([s['points'] for s in batch_samples]),
                    'labels': np.array([s['label'] for s in batch_samples])
                }
            else:  # multi_scene
                # For ragged arrays, we can't stack directly
                # Return as list or pad to fixed size
                batch = {
                    'points': [s['points'] for s in batch_samples],
                    'object_ids': [s['object_ids'] for s in batch_samples]
                }
            
            yield batch
    
    train_loader = lambda: get_batches(train_indices, batch_size, shuffle)
    test_loader = lambda: get_batches(test_indices, batch_size, False)
    
    return train_loader, test_loader


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate MNIST point cloud datasets")
    parser.add_argument("--dataset_type", type=str, required=True,
                       choices=['single', 'single_rotated', 'single_translated', 'single_rotated_translated',
                               'multi', 'multi_rotated', 'multi_rotated_scaled'],
                       help="Type of dataset to generate")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Output file path (default: auto-generated)")
    parser.add_argument("--file_format", type=str, default="npz", choices=['npz', 'pkl'],
                       help="File format (npz or pkl)")
    parser.add_argument("--num_points", type=int, default=500,
                       help="Number of points per digit (for simple dataset)")
    parser.add_argument("--num_scenes", type=int, default=20000,
                       help="Number of scenes (for multi-digit datasets)")
    parser.add_argument("--points_per_digit", type=int, default=500,
                       help="Number of points per digit (for multi-digit datasets)")
    parser.add_argument("--force_overwrite", action="store_true",
                       help="Force overwrite existing files")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set default output paths
    if args.output_path is None:
        if args.dataset_type == 'single':
            args.output_path = f"data/mnist_2d_single.npz"
        elif args.dataset_type == 'single_rotated':
            args.output_path = f"data/mnist_2d_single_rotated.npz"
        elif args.dataset_type == 'single_translated':
            args.output_path = f"data/mnist_2d_single_translated.npz"
        elif args.dataset_type == 'single_rotated_translated':
            args.output_path = f"data/mnist_2d_single_rotated_translated.npz"
        elif args.dataset_type == 'multi':
            args.output_path = f"data/multi_mnist_2d.npz"
        elif args.dataset_type == 'multi_rotated':
            args.output_path = f"data/multi_mnist_2d_rotated.npz"
        elif args.dataset_type == 'multi_rotated_scaled':
            args.output_path = f"data/multi_mnist_2d_rotated_scaled.npz"
    
    # Generate dataset
    if args.dataset_type == 'single':
        generate_simple_mnist_point_clouds(
            num_points=args.num_points,
            output_path=args.output_path,
            file_format=args.file_format,
            force_overwrite=args.force_overwrite,
            seed=args.seed
        )
    elif args.dataset_type == 'single_rotated':
        generate_simple_mnist_rotated(
            num_points=args.num_points,
            output_path=args.output_path,
            file_format=args.file_format,
            force_overwrite=args.force_overwrite,
            seed=args.seed
        )
    elif args.dataset_type == 'single_translated':
        generate_simple_mnist_translated(
            num_points=args.num_points,
            output_path=args.output_path,
            file_format=args.file_format,
            force_overwrite=args.force_overwrite,
            seed=args.seed
        )
    elif args.dataset_type == 'single_rotated_translated':
        generate_simple_mnist_rotated_translated(
            num_points=args.num_points,
            output_path=args.output_path,
            file_format=args.file_format,
            force_overwrite=args.force_overwrite,
            seed=args.seed
        )
    elif args.dataset_type == 'multi':
        generate_multi_digit_scenes_normal(
            num_scenes=args.num_scenes,
            points_per_digit=args.points_per_digit,
            output_path=args.output_path,
            file_format=args.file_format,
            force_overwrite=args.force_overwrite,
            seed=args.seed
        )
    elif args.dataset_type == 'multi_rotated':
        generate_multi_digit_scenes_rotated(
            num_scenes=args.num_scenes,
            points_per_digit=args.points_per_digit,
            output_path=args.output_path,
            file_format=args.file_format,
            force_overwrite=args.force_overwrite,
            seed=args.seed
        )
    elif args.dataset_type == 'multi_rotated_scaled':
        generate_multi_digit_scenes_rotated_scaled(
            num_scenes=args.num_scenes,
            points_per_digit=args.points_per_digit,
            output_path=args.output_path,
            file_format=args.file_format,
            force_overwrite=args.force_overwrite,
            seed=args.seed
        )
    
    print(f"\nDataset generation complete!")
    print(f"Dataset saved to: {args.output_path}")

