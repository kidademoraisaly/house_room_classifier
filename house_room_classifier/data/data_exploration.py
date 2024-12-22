import hashlib
import numpy as np
import os

def count_images_by_class(dataset):
    class_counts = {}
    for images, labels in dataset:
        unique_classes, counts = np.unique(labels.numpy(), return_counts=True)
        for cls, count in zip(unique_classes, counts):
            class_counts[cls] = class_counts.get(cls, 0) + count
    return class_counts

def count_files_per_class(directory):
    return {class_name: len(os.listdir(os.path.join(directory, class_name))) 
            for class_name in os.listdir(directory) 
            if os.path.isdir(os.path.join(directory, class_name))}

def image_color_distribution(dataset, dataset_name):
    total_rgb = 0  
    total_grayscale = 0  
    total_unusual = 0  

    for images, _ in dataset:
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

