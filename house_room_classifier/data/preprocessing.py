import tensorflow as tf


def prepare_dataset(
        data_dir,
        img_height=150,
        img_width=150,
        batch_size=20,
        validation_split=0.2,
        seed=123
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

    data_augmentation=tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ]
    )
  
    normalization_layer=tf.keras.layers.Rescaling(1./255)
    train_ds=train_ds.map(
        lambda x, y:(data_augmentation(x,training=True),y)
    )

    train_ds=train_ds.map(
        lambda x, y:(normalization_layer(x,training=True),y)
    )


    val_ds=val_ds.map(
        lambda x, y:(normalization_layer(x,training=True),y)
    )
    AUTOTUNE=tf.data.AUTOTUNE
    train_ds.prefetch(AUTOTUNE)
    val_ds=val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds


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