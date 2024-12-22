import tensorflow as tf
import os
import hashlib
from collections import defaultdict

def load_dataset(data_dir, img_height=150,img_width=150, batch_size=20,subset=None, validation_split=0.2,seed=123, shuffle=True):
    return tf.keras.utils.image_dataset_from_directory(
            data_dir,
            image_size=(img_height,img_width),
            batch_size=batch_size,
            shuffle=shuffle,
            validation_split=validation_split if subset else None,
            subset=subset,
            seed=seed
    )

def load_datasets(train_dir,val_dir=None,test_dir=None,img_height=150,img_width=150,batch_size=20,validation_split=0.2,seed=123):

    if val_dir and test_dir:
        # If separate directories are provided for all sets
        train_ds = load_dataset(train_dir, img_height, img_width, batch_size, seed=seed)
        val_ds = load_dataset(val_dir, img_height, img_width, batch_size, shuffle=False, seed=seed)
        test_ds = load_dataset(test_dir, img_height, img_width, batch_size, shuffle=False, seed=seed)
    else:
        train_ds = load_dataset( 
            train_dir,
            img_height,
            img_width,
            batch_size,
            subset='training',
            validation_split=validation_split,
            seed=seed
        )
        val_ds = load_dataset(
            train_dir, 
            img_height, 
            img_width, 
            batch_size, 
            subset='validation', 
            validation_split=validation_split,
            shuffle=False, 
            seed=seed
        )
        test_ds=None

    return train_ds, val_ds, test_ds

def apply_augmentations(dataset,data_augmentation):
    dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y)
                          ,num_parallel_calls=tf.data.AUTOTUNE
                          )
    return dataset

def apply_augmentations_image(image, data_augmentation):
    if len(image.shape) == 3:  # Assuming image is (H, W, C)
        image = tf.expand_dims(image, axis=0)
    augmented_image = data_augmentation(image, training=True)
    return tf.squeeze(augmented_image, axis=0)

def apply_normalization(dataset,normalization):
    return dataset.map(lambda x, y: (normalization(x), y)
                       ,num_parallel_calls=tf.data.AUTOTUNE
                       )

def compute_image_hash(image_path):
    """Compute the hash of an image file."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    return hashlib.md5(image_bytes).hexdigest()

def find_and_remove_duplicates(dataset_path):
    """Find and remove duplicate image files in a dataset."""
    hash_dict = defaultdict(list)
    duplicate_set = set()
    for root, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                file_hash = compute_image_hash(file_path)
                hash_dict[file_hash].append(file_path)
    duplicates = {hash_value: paths for hash_value, paths in hash_dict.items() if len(paths) > 1}
    for hash_value, paths in duplicates.items():
        duplicate_set.update(paths[1:])  # Keep the first occurrence, remove others
        for path in paths[1:]:
            os.remove(path)

    return duplicate_set, len(duplicate_set)