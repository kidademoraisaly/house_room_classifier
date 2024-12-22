import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers # layers provides building blocks for neural networks
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from house_room_classifier.models.model_architectures import ModelArchitectures
from house_room_classifier.data.preprocessing import apply_normalization,apply_augmentations

class RoomClassificationModel:
    def __init__(self, img_height=150,img_width=150, num_classes=5, architecture="custom_cnn_simple_1"):
        self.img_height=img_height
        self.img_width=img_width
        self.num_classes=num_classes
        self.model=None
        self.architecture=architecture
        self.model=None
        self.training_config=None
    
    def build_model(self):
        # Dynamically select model architecture
        model_func = getattr(ModelArchitectures, self.architecture, None)
        if model_func is None:
            raise ValueError(f"Architecture {self.architecture} not found")
        
        self.model = model_func(self.img_height, self.img_width, self.num_classes)
        
        self.training_config=ModelArchitectures.get_training_config(self.architecture)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.training_config.learning_rate,
            decay_steps=self.training_config.learning_rate_decay_steps,
            decay_rate=self.training_config.learning_rate_decay
        )
        optimizer=getattr(tf.keras.optimizers,
                          self.training_config.optimizer.capitalize())(learning_rate=self.training_config.learning_rate)
             
        self.model.compile(
            optimizer=optimizer,
            loss=self.training_config.loss,
            metrics=['accuracy']
        )
    
    @tf.autograph.experimental.do_not_convert
    def prepare_dataset(self, train_ds, val_ds, test_ds):
        # Get augmentation strategy
        augmentation_strategy = ModelArchitectures.get_augmentation_strategy(self.architecture)
        normalization=tf.keras.layers.Rescaling(1./255)

        if self.training_config.use_data_augmentation:
            train_ds = apply_augmentations(train_ds,augmentation_strategy)     
        # Normalize datasets
        train_ds = apply_normalization(train_ds,normalization)
        val_ds = apply_normalization(val_ds,normalization)
        test_ds=apply_normalization(test_ds,normalization)
        
        return train_ds, val_ds, test_ds

    def train(self, train_ds, val_ds, class_weights=None):      
        train_ds, val_ds, _=self.prepare_dataset(train_ds,val_ds,val_ds)  
        early_stopping=EarlyStopping(
            monitor='val_loss',
            patience=self.training_config.early_stopping_patience,
            restore_best_weights=True
        )
        lr_reducer=ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        history=self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.training_config.epochs,         
            callbacks=[early_stopping,lr_reducer],
            class_weight=class_weights
        )
        return history
    
    def evaluate(self, test_ds):
        test_loss, test_accuracy=self.model.evaluate(test_ds)
        return{
            'test_loss':test_loss,
            'test_accuracy':test_accuracy
        }
    
    def predit(self, image):
        return self.model.predict(image)
    
    def save_model(self, file_path="models/room_classifier_model.keras"):
        self.model.save(file_path)
    
    def load_model(self, file_path="models/room_classifier_model.Keras"):
        self.model=models.load_model(file_path)




        
