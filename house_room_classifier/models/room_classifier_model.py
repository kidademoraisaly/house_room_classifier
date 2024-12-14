# =====================================================
# 1. Importing Libraries
# =====================================================

# from keras.api.applications import MobileNetV2
# from keras.api.callbacks 
# from keras.api import layers, models, optimizers

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers # layers provides building blocks for neural networks
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from house_room_classifier.models.model_architectures import ModelArchitectures
from house_room_classifier.data.preprocessing import apply_normalization,apply_augmentations




# =====================================================
# 2. Class: RoomClassificationModel
# (This class builds, trains, evaluates, and saves a CNN to classify images - like "bedroom", "kitchen", etc.)
# =====================================================

class RoomClassificationModel:
    def __init__(self, img_height=150,img_width=150, num_classes=5, architecture="custom_cnn_simple_1"):
        self.img_height=img_height
        self.img_width=img_width
        self.num_classes=num_classes
        self.model=None
        self.architecture=architecture
        self.model=None
        self.training_config=None
    
    # def build_model(self, base_model=None):      
    #     #to use pretrained model, now we are not going to using this part,
    #     #first we are going to start simple
    #     if base_model=='mobilenet':
    #         base=tf.keras.applications.MobileNetV2(
    #             input_shape=(self.img_width, self.img_height,3
    #                          ),
    #             include_top=False, #what is this parameter ??~
    #             weights='imagenet' # and this??                    
    #         )
    #         base.trainable=False #and this??
    #         self.model=models.Sequential(
    #             [
    #                 base,
    #                 layers.GlobalAveragePooling2D(),
    #                 layers.Dense(256, activation='relu'),
    #                 layers.Dropout(0.5),
    #                 layers.Dense(self.num_classes, activation="softmax")
    #             ]
    #         )
    #     # customer CNN arquitecture
    #     else:
    #         self.model=models.Sequential(
    #             [
    #                 layers.Conv2D(32, (3,3), activation='relu', input_shape=(self.img_height,self.img_width,3), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    #                 layers.MaxPooling2D((2,2)),
    #                 layers.Conv2D(64, (3,3), activation='relu',  kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    #                 layers.MaxPooling2D((2,2)),
    #                 layers.Conv2D(128, (3,3), activation='relu',  kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    #                 layers.MaxPooling2D((2,2)),
    #                 layers.Conv2D(256, (3,3), activation='relu',  kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    #                 layers.MaxPooling2D((2,2)),
    #                 layers.Flatten(),
    #                 layers.Dropout(0.6),
    #                 layers.Dense(512, activation='relu' ,kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    #                 layers.Dropout(0.6),
    #                 layers.Dense(self.num_classes, activation="softmax")
    #                 #layers.Dense(self.num_classes, name="outputs")
                    
    #             ]
    #         )

    #     self.model.compile(optimizer='adam',
    #           #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #           optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    #           metrics=['accuracy'])
         
    #     # self.model.compile(loss='categorical_crossentropy',
    #     #                    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
    #     #                    # could be #optimizers.Adam(learning_rate=0.0001)
    #     #                    metrics=['accuracy']
    #     #                    )
    
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
        #Using fixed learning rate, Now using the scheduler for now
        optimizer=getattr(tf.keras.optimizers,
                          self.training_config.optimizer.capitalize())(learning_rate=self.training_config.learning_rate)
        
        # optimizer=getattr(tf.keras.optimizers,
        #                   self.training_config.optimizer.capitalize())(learning_rate=lr_schedule)
                          
        self.model.compile(
            optimizer=optimizer,
            loss=self.training_config.loss,
            metrics=['accuracy']
        )
    
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

    def train(self, train_ds, val_ds):      
        train_ds, val_ds, _=self.prepare_dataset(train_ds,val_ds,val_ds)  
        # #WHAT is a scheduler
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        # initial_learning_rate=0.0001,
        # decay_steps=100,
        # decay_rate=0.9
        # )
        # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)  
        # self.model.compile(
        #     optimizer=optimizer,
        #     loss='sparse_categorical_crossentropy',
        #     metrics=['accuracy']
        # )

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
            #train_generator.batch_size,
            #steps_per_epoch=steps_per_epoch,
            epochs=self.training_config.epochs,         
            #validation_steps=validation_steps,
            callbacks=[early_stopping,lr_reducer]
            #callbacks=[early_stopping,lr_reducer]
        )
        return history
    
    # ---------------------------------------------
    # 2.4. Evaluating the Model
    # --------------------------------------------- 

    # This function is used to test the model's performance on the test dataset (that the model hasn’t seen during training or validation)
    # It evaluates the loss and accuracy of the trained model on the provided test dataset
    def evaluate(self, test_ds):
        test_loss, test_accuracy=self.model.evaluate(test_ds)
        return{
            'test_loss':test_loss,
            'test_accuracy':test_accuracy
        }

    # ---------------------------------------------
    # 2.4. Predicting the Model
    # --------------------------------------------- 

    # The predict() method of the model generates predictions based on the input data (the image in this case).
    def predit(self, image):
        return self.model.predict(image)
    
    def save_model(self, file_path="models/room_classifier_model.keras"):
        self.model.save(file_path)
    
    def load_model(self, file_path="models/room_classifier_model.Keras"):
        self.model=models.load_model(file_path)




        
