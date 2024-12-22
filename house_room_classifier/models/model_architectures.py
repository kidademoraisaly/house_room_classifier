from tensorflow.keras import models,layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from house_room_classifier.models.training_config import TrainingConfig

class ModelArchitectures:
    @staticmethod
    def custom_cnn_simple_v1(img_height, img_width, num_classes):
        """Basic custom CNN with minimal layers"""
        model = models.Sequential(
                [
                    layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_height,img_width,3)),
                    layers.MaxPooling2D((2,2)),
                    layers.Conv2D(64, (3,3), activation='relu'),
                    layers.MaxPooling2D((2,2)),
                    layers.Conv2D(128, (3,3), activation='relu'),
                    layers.MaxPooling2D((2,2)),
                    layers.Flatten(),
                    layers.Dropout(0.6),
                    layers.Dense(512, activation='relu'),
                    layers.Dropout(0.6),
                    layers.Dense(num_classes, activation="softmax")
                    
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
                    
                ]
            )
        return model
    
    @staticmethod
    def custom_cnn_complex_v2(img_height, img_width, num_classes):
        """More complex custom CNN with multiple layers and regularization"""
        model = models.Sequential(
                [
                    layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_height,img_width,3), kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
                    layers.MaxPooling2D((2,2)),
                    layers.Conv2D(64, (3,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)), # smaller penalty to allow the weights to learn while maintaining some regularization
                    layers.MaxPooling2D((2,2)),
                    layers.Conv2D(128, (3,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
                    layers.MaxPooling2D((2,2)),
                    layers.Conv2D(128, (3,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
                    layers.MaxPooling2D((2,2)),
                    layers.Flatten(),
                    layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
                    layers.Dropout(0.4),
                    layers.Dense(num_classes, activation="softmax")                   
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
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False

        
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
    
    def pretrained_resnet50_full_training(img_height, img_width, num_classes):
        """ResNet50 with training of all layers"""
        base_model = tf.keras.applications.ResNet50(
            input_shape=(img_height, img_width, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Make all layers trainable
        base_model.trainable = True
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])        
        return model
    
    def pretrained_resnet50_full_training_v2(img_height, img_width, num_classes):
        """ResNet50 with training of all layers"""
        base_model = tf.keras.applications.ResNet50(
            input_shape=(img_height, img_width, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Make all layers trainable
        base_model.trainable = True
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])        
        return model
    
    def pretrained_resnet50_full_training_v3(img_height, img_width, num_classes):
        """ResNet50 with training of all layers"""
        base_model = tf.keras.applications.ResNet50(
            input_shape=(img_height, img_width, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Make all layers trainable
        base_model.trainable = True
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])        
        return model
    
    @staticmethod
    def custom_cnn_simple_v2(img_height, img_width, num_classes):
        """Basic custom CNN with minimal layers"""
        model = models.Sequential(
                [
                    layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_height,img_width,3)),
                    layers.MaxPooling2D((2,2)),
                    layers.Conv2D(64, (3,3), activation='relu'),
                    layers.MaxPooling2D((2,2)),
                    layers.Conv2D(128, (3,3), activation='relu'),
                    layers.MaxPooling2D((2,2)),
                    layers.Flatten(),
                    layers.Dropout(0.6),
                    layers.Dense(512, activation='relu'),
                    layers.Dropout(0.6),
                    layers.Dense(num_classes, activation="softmax")
                    #layers.Dense(self.num_classes, name="outputs")
                    
                ]
            )
        return model
    
    @staticmethod
    def get_training_config(archicteture):
        configs={
            'custom_cnn_simple_v1': TrainingConfig(
                epochs=15,
                learning_rate=0.0001,
                early_stopping_patience=10,
                use_data_augmentation=False
            ),
            'custom_cnn_complex_v1': TrainingConfig(
                epochs=20,
                learning_rate=0.0001,
                early_stopping_patience=10,
                use_data_augmentation=True
            ),
            'custom_cnn_complex_v2': TrainingConfig(
                epochs=30,
                learning_rate=0.00001,
                early_stopping_patience=5, # to prevent unnecessary computation
                use_data_augmentation=True
            ),
            'pretrained_mobilenet_base_v1': TrainingConfig(
                epochs=20,
                learning_rate=0.00001,
                early_stopping_patience=3,
                use_data_augmentation=True
            ),
            'pretrained_resnet50_fine_v1': TrainingConfig(
                epochs=25,
                learning_rate=0.00001,
                early_stopping_patience=12,
                use_data_augmentation=True
            ),
            'pretrained_resnet50_full_training': TrainingConfig(
                epochs=20,
                learning_rate=0.00001,
                early_stopping_patience=3,
                use_data_augmentation=True
            ),
            'pretrained_resnet50_full_training_v2': TrainingConfig(
            epochs=19,
            learning_rate=0.00001,
            early_stopping_patience=3,
            use_data_augmentation=True
            ),
            'pretrained_resnet50_full_training_v3': TrainingConfig(
            epochs=100,
            learning_rate=0.00001,
            early_stopping_patience=5,
            use_data_augmentation=True
            ),
            'custom_cnn_simple_v2': TrainingConfig(
                epochs=25,
                learning_rate=0.0001,
                early_stopping_patience=3,
                use_data_augmentation=False
            ),
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
            'custom_cnn_complex_v2': tf.keras.Sequential([
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
            ]),
            'pretrained_resnet50_full_training': tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.RandomRotation(0.2),
                tf.keras.layers.RandomZoom(0.1),
                tf.keras.layers.RandomContrast(0.1),
                tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1)
            ]),
            'pretrained_resnet50_full_training_v2': tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.RandomRotation(0.2),
                tf.keras.layers.RandomZoom(0.1),
                tf.keras.layers.RandomContrast(0.1),
                tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1)
            ]),
            'pretrained_resnet50_full_training_v3': tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.RandomRotation(0.2),
                tf.keras.layers.RandomZoom(0.1),
                tf.keras.layers.RandomContrast(0.1),
                tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1)
            ]),
            'custom_cnn_simple_v2': tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.RandomRotation(0.2),
                tf.keras.layers.RandomZoom(0.1),
                tf.keras.layers.RandomContrast(0.1),
                tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1)
            ]),
        }
        return augmentation_strategies.get(
            architecture,
            tf.keras.Sequential([])
        )