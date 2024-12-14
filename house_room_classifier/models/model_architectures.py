from tensorflow.keras import models,layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from house_room_classifier.models.training_config import TrainingConfig

# [ModelType]_[BaseArchitecture]_[Complexity]_[Version]
# Examples:

# custom_cnn_simple_v1
# custom_cnn_complex_v2
# pretrained_mobilenet_base_v1
# pretrained_resnet50_fine_v2
# custom_small_augmented_v1
class ModelArchitectures:

    @staticmethod
    def custom_cnn_simple_v1(img_height, img_width, num_classes):
        """Basic custom CNN with minimal layers"""
        model = models.Sequential(
                [
                    layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_height,img_width,3), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                    layers.MaxPooling2D((2,2)),
                    layers.Conv2D(64, (3,3), activation='relu',  kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                    layers.MaxPooling2D((2,2)),
                    layers.Conv2D(128, (3,3), activation='relu',  kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                    layers.MaxPooling2D((2,2)),
                    layers.Flatten(),
                    layers.Dropout(0.6),
                    layers.Dense(512, activation='relu' ,kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                    layers.Dropout(0.6),
                    layers.Dense(num_classes, activation="softmax")
                    #layers.Dense(self.num_classes, name="outputs")
                    
                ]
            )
        return model

    @staticmethod
    def custom_cnn_complex_v1(img_height, img_width, num_classes):
        """More complex custom CNN with multiple layers and regularization"""
        model = models.Sequential(
                [
                    layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_height,img_width,3), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                    layers.MaxPooling2D((2,2)),
                    layers.Conv2D(64, (3,3), activation='relu',  kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                    layers.MaxPooling2D((2,2)),
                    layers.Conv2D(128, (3,3), activation='relu',  kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                    layers.MaxPooling2D((2,2)),
                    layers.Conv2D(256, (3,3), activation='relu',  kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                    layers.MaxPooling2D((2,2)),
                    layers.Flatten(),
                    layers.Dropout(0.6),
                    layers.Dense(512, activation='relu' ,kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                    layers.Dropout(0.6),
                    layers.Dense(num_classes, activation="softmax")
                    #layers.Dense(self.num_classes, name="outputs")
                    
                ]
            )
        return model

    @staticmethod
    def pretrained_mobilenet_base_v1(img_height, img_width, num_classes):
        """MobileNetV2 with frozen base layers"""
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(img_width, img_height, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model

    @staticmethod
    def pretrained_resnet50_fine_v1(img_height, img_width, num_classes):
        """ResNet50 with fine-tuning of later layers"""
        base_model = tf.keras.applications.ResNet50(
            input_shape=(img_width, img_height, 3),
            include_top=False,
            weights='imagenet'
        )
        # Fine-tune later layers
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model
    
    @staticmethod
    def get_training_config(archicteture):
        configs={
            'custom_cnn_simple_v1': TrainingConfig(
                epochs=30,
                learning_rate=0.001,
                early_stopping_patience=5,
                use_data_augmentation=False
            ),
            'custom_cnn_complex_v1': TrainingConfig(
                epochs=20,
                learning_rate=0.0001,
                early_stopping_patience=10,
                use_data_augmentation=True
            ),
            'pretrained_mobilenet_base_v1': TrainingConfig(
                epochs=15,
                learning_rate=0.00001,
                early_stopping_patience=8,
                use_data_augmentation=True
            ),
            'pretrained_resnet50_fine_v1': TrainingConfig(
                epochs=25,
                learning_rate=0.00001,
                early_stopping_patience=12,
                use_data_augmentation=True
            )

        }
        return configs.get(archicteture, {}) 
    
    @staticmethod
    def get_augmentation_strategy(architecture):
        augmentation_strategies={
            'custom_cnn_simple_v1': tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1), 
            ]),
            'custom_cnn_complex_v1': tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.RandomRotation(0.2),
                tf.keras.layers.RandomZoom(0.2),
                tf.keras.layers.RandomContrast(0.1),
                 tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1)
            ]),
            'pretrained_mobilenet_base_v1': tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.RandomRotation(0.2),
                tf.keras.layers.RandomZoom(0.1),
                tf.keras.layers.RandomContrast(0.1),
                tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1)
            ]),
            'pretrained_resnet50_fine_v1': tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.RandomRotation(0.2),
                tf.keras.layers.RandomZoom(0.1),
                tf.keras.layers.RandomContrast(0.1),
                tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1)
            ])
        }
        return augmentation_strategies.get(
            architecture,
            tf.keras.Sequential([])
        )