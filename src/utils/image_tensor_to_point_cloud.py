import jax.numpy as jnp
import jax.random as jr
import jax

def image_tensor_to_point_cloud(image_tensor, save_fname=None):
    image_tensor = image_tensor.sum(-1)  # Convert to grayscale
    image_tensor = image_tensor / 255.0/3.0  # Normalize to [0, 1]
    image_tensor = jnp.clip(image_tensor, 0, 1)  # Ensure values are in [0, 1]

    image_tensor = image_tensor > 0.2  # Binarize the image
    image_indices, y_coords, x_coords = jnp.where(image_tensor)

    # Stack the coordinates into a single array of shape (num_points, 3)
    all_points = jnp.stack([image_indices, x_coords, y_coords], axis=-1)

    # Create a mask to group points by image index
    num_images = image_tensor.shape[0]
    point_counts = jnp.bincount(image_indices, minlength=num_images)  # Number of points per image
    max_points_per_image = jnp.max(point_counts)  # Limit to max_points

    # Create an array to store the point clouds
    point_cloud = jnp.zeros((num_images, max_points_per_image, 2), dtype=jnp.int32)

    temp = 0
    m=0
    for i in range(num_images):
        temp = jnp.sum(image_indices==i)
        if temp>m:
            m = temp

    max_points = m
    point_cloud = jnp.zeros((num_images, max_points, 2), dtype=jnp.int32)
    point_cloud_mask = jnp.zeros((num_images, max_points), dtype=jnp.bool)

    for i in range(num_images):
        mask = image_indices == i
        point_cloud = point_cloud.at[i, :jnp.sum(mask), :].set(all_points[mask, 1:])
        point_cloud_mask = point_cloud_mask.at[i, :jnp.sum(mask)].set(True)

    # Save the point cloud
    if save_fname is not None:
        jnp.save(save_fname+'_point_cloud.npy', point_cloud)
        jnp.save(save_fname+'_point_cloud_mask.npy', point_cloud)

    return point_cloud, point_cloud_mask


