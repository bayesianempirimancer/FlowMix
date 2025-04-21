import os
from PIL import Image
import numpy as np

def load_images_and_labels(base_directory):
    """
    Load PNG images and their corresponding labels from a directory structure.

    Args:
        base_directory (str): The base directory containing subdirectories with images.

    Returns:
        tuple: A tuple containing:
            - data (np.ndarray): A NumPy array of shape (num_images, height, width, channels) with image data.
            - labels (np.ndarray): A NumPy array of shape (num_images, 3) with labels extracted from directory names.
    """
    data = []
    labels = []

    # Traverse the directory structure
    for label_dir in os.listdir(base_directory):
        label_path = os.path.join(base_directory, label_dir)
        if os.path.isdir(label_path):  # Ensure it's a directory
            try:
                # Extract the label (###) as a list of integers
                label = [int(digit) for digit in label_dir]
            except ValueError:
                print(f"Skipping invalid directory name: {label_dir}")
                continue

            # Load all PNG images in the directory
            for file_name in os.listdir(label_path):
                if file_name.endswith('.png'):
                    file_path = os.path.join(label_path, file_name)
                    image = Image.open(file_path).convert('RGB')  # Convert to RGB
                    image_array = np.array(image)
                    data.append(image_array)
                    labels.append(label)

    # Convert lists to NumPy arrays
    data = np.array(data)
    labels = np.array(labels)

    print("Data shape:", data.shape)  # (num_images, height, width, channels)
    print("Labels shape:", labels.shape)  # (num_images, 3)

    return data, labels

# TRAINING DATA
train_image_directory = 'datasets/triple_mnist/train/'
data, labels = load_images_and_labels(train_image_directory)

data_save_path = 'datasets/triple_mnist/train_data.npy'
labels_save_path = 'datasets/triple_mnist/train_labels.npy'

np.save(data_save_path, data)
np.save(labels_save_path, labels)

print(f"Data saved to {data_save_path}")
print(f"Labels saved to {labels_save_path}")

# TEST DATA
test_image_directory = 'datasets/triple_mnist/test/'
data, labels = load_images_and_labels(test_image_directory)

data_save_path = 'datasets/triple_mnist/test_data.npy'
labels_save_path = 'datasets/triple_mnist/test_labels.npy'

np.save(data_save_path, data)
np.save(labels_save_path, labels)

print(f"Data saved to {data_save_path}")
print(f"Labels saved to {labels_save_path}")

# VALIDATION DATA
train_image_directory = 'datasets/triple_mnist/val/'
data, labels = load_images_and_labels(train_image_directory)

data_save_path = 'datasets/triple_mnist/val_data.npy'
labels_save_path = 'datasets/triple_mnist/val_labels.npy'

np.save(data_save_path, data)
np.save(labels_save_path, labels)

print(f"Data saved to {data_save_path}")
print(f"Labels saved to {labels_save_path}")

