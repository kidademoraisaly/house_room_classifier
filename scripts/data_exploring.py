# =====================================================
# 1. Importing Libraries
# =====================================================

import pathlib
import os
import PIL
import PIL.Image
import matplotlib.pyplot as plt
from house_room_classifier.data.preprocessing import prepare_dataset, load_datasets
from house_room_classifier.utils.visualization_data import visualize_first_images
import tensorflow  as tf

# =====================================================
# 2. Classification Task
# =====================================================

# ---------------------------------------------
# 2.1. Define Dataset Directories
# ---------------------------------------------

def main():
    DATA_DIR="data"     # Root directory where all datasets are stored
    train_ds_dir=pathlib.Path(os.path.join(DATA_DIR,"train"))  # Subfolders: "train", "valid", and "test" and pathlib.Path converts
    val_ds_dir=pathlib.Path(os.path.join(DATA_DIR,"valid"))    # these directory paths into Path objects, which allow easy file and folder manipulations
    test_ds_dir=pathlib.Path(os.path.join(DATA_DIR,"test"))

# ---------------------------------------------
# 2.2. Count Images in the Training, Validation and Test Sets
# ---------------------------------------------
    
    # Training set
    image_count=len(list(train_ds_dir.glob('*/*.jpg')))          # Counts all images in the training set
    print(f"Total number of images on training: {image_count}")

    baths_count=len(list(train_ds_dir.glob("Bathroom/*")))       # Counts bathroom images in the training set
    print(f"Total number of Bathrooms on training : {baths_count}")

    # Validation set
    image_count=len(list(val_ds_dir.glob('*/*.jpg')))            # Counts all images in the validation set
    print(f"Total number of images on validation: {image_count}")

    baths_count=len(list(val_ds_dir.glob("Bathroom/*")))         # Counts bathroom images in the validation set
    print(f"Total number of Bathrooms on validation : {baths_count}")

    # Test set
    image_count=len(list(test_ds_dir.glob('*/*.jpg')))           # Counts all images in the test set
    print(f"Total number of images on test: {image_count}")

    baths_count=len(list(test_ds_dir.glob("Bathroom/*")))        # Counts bathroom images in the test set
    print(f"Total number of Bathrooms on test : {baths_count}")

# ---------------------------------------------
# 2.3. Visualize One Image
# ---------------------------------------------

    bath_rooms=list(val_ds_dir.glob("Bathroom/*"))
    image=PIL.Image.open(str(bath_rooms[0]))
     
    #image.show() 
    train_ds, val_ds,test_ds=load_datasets(
        train_ds_dir,
        val_dir=val_ds_dir,
        test_dir=test_ds_dir,
        img_height=150,
        img_width=150,
        batch_size=20,
        seed=123
    )
    class_names=train_ds.class_names
    print(f"class_names", class_names)
    total_train_batches=tf.data.experimental.cardinality(train_ds).numpy()
    total_val_batches=tf.data.experimental.cardinality(val_ds).numpy()
    total_test_batches=tf.data.experimental.cardinality(test_ds).numpy()
    
    print(f"Total train batches: {total_train_batches}")
    print(f"Total validation batches: {total_val_batches}")
    print(f"Total test batches: { total_test_batches}")

    visualize_first_images(train_ds, class_names=train_ds.class_names,num_images=16)

    visualize_first_images(val_ds, class_names=val_ds.class_names,num_images=12)


# ---------------------------------------------
# 2.4. Execution
# ---------------------------------------------

# This ensures that the main() function runs when the script is executed directly (not imported)
if __name__=="__main__":
    main()