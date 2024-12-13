import tensorflow as tf
import os

# The function prepare_dataset() prepares image datasets for training and validation
def prepare_dataset(
        data_dir,                    # Path to the directory containing the image data
        img_height=150,              # Dimensions to resize the images to (default is 150x150 pixels)
        img_width=150,
        batch_size=20,               # Number of images to process in one step (default is 20)
        validation_split=0.2,        # Proportion of the data to use for validation (default is 20%), in case we dont have a validation folder (but we have)
        seed=123                     # Ensures reproducibility of random operations like shuffling
):
    train_ds=tf.keras.utils.image_dataset_from_directory(
        data_dir,
        shuffle=True,
        validation_split=validation_split,
        subset='training',
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds=tf.keras.utils.image_dataset_from_directory(
        data_dir,
        shuffle=True,
        validation_split=validation_split,
        subset='validation',
        seed=seed,
        image_size=(img_height
                    ,img_width),
        batch_size=batch_size
    )
    
    # Applies random transformations (data augmentation) to the training images to simulate a more diverse dataset
    data_augmentation=tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),     # Randomly flips images horizontally
            tf.keras.layers.RandomRotation(0.1),          # Randomly rotates images by up to 10% of 360°
            tf.keras.layers.RandomZoom(0.1),              # Randomly zooms into the images by 10%
        ]
    )
  
    # Normalizes pixel values of the images to be between 0 and 1 (instead of 0 to 255). This is necessary for faster and more efficient training. 
    normalization_layer=tf.keras.layers.Rescaling(1./255)

    # Applying Data Augumentation
    train_ds=train_ds.map(
        lambda x, y:(data_augmentation(x,training=True),y)
    )

    # Applying Normalization
    train_ds=train_ds.map(
        lambda x, y:(normalization_layer(x,training=True),y)
    )

    val_ds=val_ds.map(
        lambda x, y:(normalization_layer(x,training=True),y)
    )

    # Prefetching loads the data in parallel while the model is training, making the training process faster and more efficient
    AUTOTUNE=tf.data.AUTOTUNE
    train_ds.prefetch(AUTOTUNE)
    val_ds=val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds            # The function returns the prepared training dataset (train_ds) and validation dataset (val_ds)


def __init__():
    prepare_dataset("/data/preprocessing.py")
  

#    test_generator = test_datagen.flow_from_directory(
#         data_dir,
#         target_size=(img_height, img_width),
#         batch_size=batch_size,
#         class_mode='categorical',
#         shuffle=False
#     )

#     return train_generator, validation_generator, test_generator