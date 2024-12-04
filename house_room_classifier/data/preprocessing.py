import tensorflow as tf


def load_dataset(data_dir, img_height,img_width, batch_size,subset=None, validation_split=0.2,seed=123, shuffle=True):
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
        # Split train directory into training, validation, and test sets
        train_ds = load_dataset( 
            train_dir,
            img_height,
            img_width,
            batch_size,
            subset='training',
            validation_split=validation_split *0.2,  # Combined split for validation and test
            seed=seed
        )
        val_ds = load_dataset(
            train_dir, 
            img_height, 
            img_width, 
            batch_size, 
            subset='validation', 
            validation_split=validation_split *2,  # Combined split for validation and test
            shuffle=False, 
            seed=seed
        )
        test_ds = load_dataset(
            train_dir, 
            img_height, 
            img_width, 
            batch_size, 
            subset='test',  # Use 'test' subset
            validation_split=validation_split *2,  # Combined split for validation and test
            shuffle=False, 
            seed=seed
        )

    return train_ds, val_ds, test_ds

def apply_augmentations(dataset):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])
#     advanced_augmentations = tf.keras.Sequential([
#     tf.keras.layers.RandomFlip("horizontal_and_vertical"),
#     tf.keras.layers.RandomRotation(0.2),
#     tf.keras.layers.RandomZoom(0.2),
#     tf.keras.layers.RandomContrast(0.1),
#     tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1)
# ])
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    # Apply augmentations
    dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
    # Apply normalization
    return dataset.map(lambda x, y: (normalization_layer(x, training=True), y))

def prepare_dataset(
        train_dir=None,
        val_dir=None,
        test_dir=None,
        train_ds=None,
        val_ds=None,
        test_ds=None,
        img_height=150,
        img_width=150,
        batch_size=20,
        validation_split=0.2,
        seed=123
):
    if train_ds is None:
        if train_dir is None:
            raise ValueError("Either train_ds or train_dir must be provided")
        
        train_ds, val_ds,test_ds = load_datasets(
            train_dir, 
            val_dir, 
            test_dir,
            img_height, 
            img_width, 
            batch_size, 
            validation_split, 
            seed
        )
    
    #train_ds, val_ds,test_ds=load_datasets(train_dir,val_dir,img_height,img_width,batch_size,validation_split,seed)
    # Apply preprocessing
    train_ds = apply_augmentations(train_ds)
  
    #we do not apply augumentation on validation data
    val_ds = val_ds.map(lambda x, y: (tf.keras.layers.Rescaling(1. / 255)(x, training=True), y)) 
    test_ds = test_ds.map(lambda x, y: (tf.keras.layers.Rescaling(1. / 255)(x, training=True), y))

    
    AUTOTUNE=tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds,test_ds


