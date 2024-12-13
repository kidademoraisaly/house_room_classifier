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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # Special tools (callbacks) to control training and improve performance


# =====================================================
# 2. Class: RoomClassificationModel
# (This class builds, trains, evaluates, and saves a CNN to classify images - like "bedroom", "kitchen", etc.)
# =====================================================

class RoomClassificationModel:
    # ---------------------------------------------
    # 2.1. Constructor: Initialize Image and Class Parameters
    # ---------------------------------------------
    
    # it initializes the class with default image dimensions 150x150 and 6 classes (categories to classify, like "bedroom", "kitchen", etc.)
    def __init__(self, img_height=150,img_width=150, num_classes=5):
        self.img_height=img_height
        self.img_width=img_width
        self.num_classes=num_classes
        self.model=None                   # A placeholder to hold the neural network model
    

    # ---------------------------------------------
    # 2.2. Model Building: Pretrained MobileNetV2 or Custom CNN (default option)
    # ---------------------------------------------

    def build_model(self, base_model=None):      
        
        # 2.2.1 Pretrained Model (MobileNetV2)
        # ---------------------------------------------
            
        if base_model=='mobilenet':
            base=tf.keras.applications.MobileNetV2(                # MobileNetV2 is a pre-trained model trained on the ImageNet dataset (a large image database). It’s very good at recognizing images.
                input_shape=(self.img_width, self.img_height, 3),  # 3 represents the number of color channels in the image (RGB)
                include_top=False,    # It excludes the final layer of the MobileNetV2 because we will add our own layer for classification
                weights='imagenet'    # Loads pre-trained weights (the parameters MobileNet learned from the ImageNet dataset)                    
            )
            base.trainable=False      # Freezes the pre-trained MobileNet weights so they are not changed during training
            
            # Adding custom layers
            self.model=models.Sequential([
                    base,                                 # Use the MobileNetV2 model
                    layers.GlobalAveragePooling2D(),      # Reduces spatial dimensions into a vector
                    layers.Dense(256, activation='relu'), # Dense (fully connected) layer
                    layers.Dropout(0.5),                  # Prevents overfitting by randomly "dropping out" neurons
                    layers.Dense(self.num_classes, activation="softmax")  # Final classification layer                
            ])

        # 2.2.2 Default CNN Architecture
        # ---------------------------------------------

        else:
            self.model=models.Sequential([
                    layers.Conv2D(32, (3,3), activation='relu', input_shape=(self.img_height,self.img_width,3)), 
                    layers.MaxPooling2D((2,2)),                   # Reduces image size to make learning faster                                                
                    layers.Conv2D(64, (3,3), activation='relu'),  # Extracts features from images using small filters (e.g., edges, corners).
                    layers.MaxPooling2D((2,2)),
                    layers.Conv2D(128, (3,3), activation='relu'),
                    layers.MaxPooling2D((2,2)),
                    layers.Flatten(),
                    layers.Dropout(0.5),    # Prevents overfitting by randomly deactivating 50% of neurons
                    layers.Dense(512, activation='relu'),
                    layers.Dense(self.num_classes, activation="softmax") # Produces probabilities for each class                
            ])

            # Custom CNN architecture, more simplified
            # self.model = models.Sequential([
            #     layers.Conv2D(32, (3, 3), activation='relu', 
            #                   input_shape=(self.img_height, self.img_width, 3)),
            #     layers.MaxPooling2D((2, 2)),
            #     layers.Conv2D(64, (3, 3), activation='relu'),
            #     layers.MaxPooling2D((2, 2)),
            #     layers.Conv2D(64, (3, 3), activation='relu'),
            #     layers.Flatten(),
            #     layers.Dense(64, activation='relu'),
            #     layers.Dropout(0.5),
            #     layers.Dense(self.num_classes, activation='softmax')
            # ])

        # Compile the model
        self.model.compile(loss='categorical_crossentropy',    # Crossentropy is a loss function used for multi-class classification
                           optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),  # Optimizer: Adjusts model weights. RMSprop is used here.
                           # (we could also apply #optimizers.Adam(learning_rate=0.0001))
                           metrics=['acc'] # Tracks the model’s accuracy ('acc')
        )
    
    # ---------------------------------------------
    # 2.3. Training the Model
    # ---------------------------------------------    

    def train(self, train_ds, val_ds, epochs=20, steps_per_epoch=100, validation_steps=50):
        # steps_per_epoch (Optional): Number of batches the model processes during one epoch of training
        # validation_steps (Optional): Number of batches of validation data to evaluate at the end of each epoch        
        early_stopping=EarlyStopping(     # Callback the section / stop training when the model’s performance stops improving (prevent overfitting)
            monitor='val_loss',           # If the validation loss stops decreasing, it means the model might be starting to overfit
            patience=10,                  # The callback will wait for 10 epochs
            restore_best_weights=True     # This restores the model to the best weights (the state with the lowest validation loss) before stopping
        )

        # lr_reducer=ReduceLROnPlateau(
        #     monitor='val_loss',
        #     factor=0.5,
        #     patience=5
        # )

        # The fit() method returns a History object that contains training and validation metrics (e.g., accuracy and loss) for each epoch
        history=self.model.fit(          
            train_ds,
            validation_data=val_ds,
            #train_generator.batch_size,
            #steps_per_epoch=steps_per_epoch,
            epochs=epochs,         
            #validation_steps=validation_steps,
            callbacks=[early_stopping]
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
    
    # ---------------------------------------------
    # 2.5. Saving and Loading the Model
    # --------------------------------------------- 
    
    # The save_model method is used to save the trained model to a file so that it can be reloaded later without retraining
    def save_model(self, file_path="models/room_classifier_model.h5"):
        self.model.save(file_path)
    
    # The load_model method is used to reload a previously saved model from a file. It restores the model's architecture, weights, 
    # optimizer state, and configuration, making it ready for use
    def load_model(self, file_path="models/room_classifier_model.h5"):
        self.model=models.load_model(file_path)




        
