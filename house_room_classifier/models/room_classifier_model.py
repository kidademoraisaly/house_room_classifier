
# from keras.api.applications import MobileNetV2
# from keras.api.callbacks 
# from keras.api import layers, models, optimizers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models,layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class RoomClassificationModel:
    def __init__(self, img_height=150,img_width=150, num_classes=5):
        self.img_height=img_height
        self.img_width=img_width
        self.num_classes=num_classes
        self.model=None
    
    def build_model(self, base_model=None):      
        #to use pretrained model, now we are not going to using this part,
        #first we are going to start simple
        if base_model=='mobilenet':
            base=tf.keras.applications.MobileNetV2(
                input_shape=(self.img_width, self.img_height,3
                             ),
                include_top=False, #what is this parameter ??~
                weights='imagenet' # and this??                    
            )
            base.trainable=False #and this??
            self.model=models.Sequential(
                [
                    base,
                    layers.GlobalAveragePooling2D(),
                    layers.Dense(256, activation='relu'),
                    layers.Dropout(0.5),
                    layers.Dense(self.num_classes, activation="softmax")
                ]
            )
        # customer CNN arquitecture
        else:
            self.model=models.Sequential(
                [
                    layers.Conv2D(32, (3,3), activation='relu', input_shape=(self.img_height,self.img_width,3)),
                    layers.MaxPooling2D((2,2)),
                    layers.Conv2D(64, (3,3), activation='relu'),
                    layers.MaxPooling2D((2,2)),
                    layers.Conv2D(128, (3,3), activation='relu'),
                    layers.MaxPooling2D((2,2)),
                    layers.Flatten(),
                    layers.Dropout(0.5),
                    layers.Dense(512, activation='relu'),
                    layers.Dense(self.num_classes, activation="softmax")

                    
                ]
            )
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
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
                           # could be #optimizers.Adam(learning_rate=0.0001)
                           metrics=['acc']
                           )
    
    def train(self, train_ds, val_ds, epochs=20, steps_per_epoch=100, validation_steps=50):
        early_stopping=EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # lr_reducer=ReduceLROnPlateau(
        #     monitor='val_loss',
        #     factor=0.5,
        #     patience=5
        # )
        history=self.model.fit(
            train_ds,
            validation_data= val_ds,
            #train_generator.batch_size,
            #steps_per_epoch=steps_per_epoch,
            epochs=epochs,         
            #validation_steps=validation_steps,
            callbacks=[early_stopping]
            #callbacks=[early_stopping,lr_reducer]
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
    
    def save_model(self, file_path="models/room_classifier_model.h5"):
        self.model.save(file_path)
    
    def load_model(self, file_path="models/room_classifier_model.h5"):
        self.model=models.load_model(file_path)




        
