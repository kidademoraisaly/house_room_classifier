# -----------------------------------------------------------------
# Count Images in the Training, Validation and Test Sets
# -----------------------------------------------------------------
import hashlib
import numpy as np

def count_images_by_class(dataset):
    class_counts = {}
    for images, labels in dataset:
        unique_classes, counts = np.unique(labels.numpy(), return_counts=True)
        for cls, count in zip(unique_classes, counts):
            class_counts[cls] = class_counts.get(cls, 0) + count
    return class_counts

# -----------------------------------------------------------------
# Image Channel Counter
# -----------------------------------------------------------------

# def image_color(dataset):
#     """
#     Prints the number of color channels in images from the dataset and checks if images are RGB, grayscale, or other.

#     Args:
#         dataset (tf.data.Dataset): The TensorFlow dataset to inspect.
#     """
#     # Iterate through the dataset and inspect one batch
#     for images, _ in dataset.take(1):
#         # Check the number of channels
#         num_channels = images.shape[-1]
#         print(f"Number of channels in the images: {num_channels}")

#         # Determine if the dataset is RGB or grayscale
#         if num_channels == 3:
#             print("Images are RGB (3 channels).")
#         elif num_channels == 1:
#             print("Images are grayscale (1 channel).")
#         else:
#             print(f"Images have an unusual number of channels: {num_channels}")

def image_color_distribution(dataset, dataset_name):
    """
    Counts and prints the number of RGB, grayscale, and unusual images in the dataset.

    Args:
        dataset (tf.data.Dataset): The TensorFlow dataset to inspect.
        dataset_name (str): The name of the dataset (e.g., "Training", "Validation") for logging purposes.
    """
    total_rgb = 0  # Counter for RGB images
    total_grayscale = 0  # Counter for grayscale images
    total_unusual = 0  # Counter for images with unusual channels

    for images, _ in dataset:
        # Check the number of channels for the current batch
        num_channels = images.shape[-1]
        if num_channels == 3:
            total_rgb += images.shape[0]
        elif num_channels == 1:
            total_grayscale += images.shape[0]
        else:
            total_unusual += images.shape[0]

    print(f"{dataset_name} DATASET:")
    print(f"  Total RGB images: {total_rgb}")
    print(f"  Total grayscale images: {total_grayscale}")
    if total_unusual > 0:
        print(f"  Total images with unusual channels: {total_unusual}")
    print("-" * 40)

# -----------------------------------------------------------------
# DUPLICATES
# -----------------------------------------------------------------
def compute_image_hash(image):
    """Compute the hash of an image to identify duplicates."""
    # Convert the image tensor to bytes
    image_bytes = image.numpy().tobytes()
    # Compute a hash value for the image
    return hashlib.md5(image_bytes).hexdigest()

def find_duplicates(dataset):
    """Find duplicate images in a TensorFlow dataset."""
    hashes = set()
    duplicates = []

    # Iterate through the dataset
    for batch_idx, (images, labels) in enumerate(dataset):
        for i in range(images.shape[0]):  # Process each image in the batch
            image = images[i]
            image_hash = compute_image_hash(image)

            # Check if hash already exists
            if image_hash in hashes:
                duplicates.append((batch_idx, i))  # Store batch index and image index
            else:
                hashes.add(image_hash)

    return duplicates

# -----------------------------------------------------------------
#
# -----------------------------------------------------------------