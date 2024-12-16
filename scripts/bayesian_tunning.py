import tensorflow as tf
from tensorflow import keras
from kerastuner import BayesianOptimization
from kerastuner.engine.hyperparameters import HyperParameters
from house_room_classifier.data.preprocessing import  load_datasets
import pathlib
import os

class CNNClassificationTuner:
    @staticmethod
    def build_model(hp, input_shape, num_classes):
        """
        Dynamically build and compile a CNN model with tunable hyperparameters
        
        Args:
            hp (HyperParameters): Hyperparameters to tune
            input_shape (tuple): Input image shape
            num_classes (int): Number of classification classes
        
        Returns:
            tf.keras.Model: Compiled neural network model
        """
        # Dynamically choose number of convolutional layers
        num_conv_layers = hp.Int('num_conv_layers', 1, 4)
        
        model = keras.Sequential()
        
        # Add convolutional layers dynamically
        for i in range(num_conv_layers):
            # Dynamically choose number of filters
            filters = hp.Int(f'filters_{i}', 32, 256, step=32)
            
            # First layer needs input_shape
            if i == 0:
                model.add(keras.layers.Conv2D(
                    filters, 
                    hp.Choice(f'kernel_size_{i}', [3, 5]), 
                    activation='relu', 
                    input_shape=input_shape,
                    kernel_regularizer=keras.regularizers.l2(
                        hp.Float('l2_reg', 1e-4, 1e-2, sampling='log')
                    )
                ))
            else:
                model.add(keras.layers.Conv2D(
                    filters, 
                    hp.Choice(f'kernel_size_{i}', [3, 5]), 
                    activation='relu',
                    kernel_regularizer=keras.regularizers.l2(
                        hp.Float('l2_reg', 1e-4, 1e-2, sampling='log')
                    )
                ))
            
            # Add pooling
            model.add(keras.layers.MaxPooling2D((2, 2)))
            
            # Optional dropout after conv layers
            if hp.Boolean(f'dropout_conv_{i}'):
                model.add(keras.layers.Dropout(
                    hp.Float(f'dropout_rate_conv_{i}', 0.2, 0.5, step=0.1)
                ))
        
        # Flatten and add dense layers
        model.add(keras.layers.Flatten())
        
        # Dynamically choose number of dense layers
        num_dense_layers = hp.Int('num_dense_layers', 1, 3)
        for i in range(num_dense_layers):
            model.add(keras.layers.Dense(
                hp.Int(f'dense_units_{i}', 64, 512, step=64),
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(
                    hp.Float('l2_reg_dense', 1e-4, 1e-2, sampling='log')
                )
            ))
            
            # Optional dropout after dense layers
            if hp.Boolean(f'dropout_dense_{i}'):
                model.add(keras.layers.Dropout(
                    hp.Float(f'dropout_rate_dense_{i}', 0.2, 0.5, step=0.1)
                ))
        
        # Output layer
        model.add(keras.layers.Dense(num_classes, activation='softmax'))
        
        # Tune learning rate
        learning_rate = hp.Float(
            'learning_rate', 
            min_value=1e-4, 
            max_value=1e-2, 
            sampling='log'
        )
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    @staticmethod
    def tune_hyperparameters(
        train_ds, 
        val_ds, 
        input_shape, 
        num_classes, 
        max_trials=50, 
        epochs=20
    ):
        """
        Perform Bayesian optimization for hyperparameter tuning
        
        Args:
            train_ds (tf.data.Dataset): Training dataset
            val_ds (tf.data.Dataset): Validation dataset
            input_shape (tuple): Input image shape
            num_classes (int): Number of classification classes
            max_trials (int): Maximum number of hyperparameter combinations to try
            epochs (int): Number of epochs to train each model
        
        Returns:
            dict: Best hyperparameters and model
        """
        # Initialize Bayesian Optimization Tuner
        tuner = BayesianOptimization(
            lambda hp: CNNClassificationTuner.build_model(hp, input_shape, num_classes),
            objective='val_accuracy',
            max_trials=max_trials,
            directory='bayesian_tuning_results',
            project_name='image_classification_tuning'
        )
        
        # Define tuning method
        #tuner.hyperparameters = HyperParameters()
        #tuner.hyperparameters.Fixed('input_shape', input_shape)
       # tuner.hyperparameters.Fixed('num_classes', num_classes)
        
        # Search for best hyperparameters
        tuner.search(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_accuracy', 
                    patience=5, 
                    restore_best_weights=True
                )
            ]
        )
        
        # Get the optimal hyperparameters and model
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = tuner.get_best_models(num_models=1)[0]
        
        # Print best hyperparameters
        print("\nBest Hyperparameters:")
        for param, value in best_hps.values.items():
            print(f"{param}: {value}")
        
        return {
            'best_model': best_model,
            'best_hyperparameters': best_hps
        }
    
    @staticmethod
    def evaluate_model(model, test_ds):
        """
        Evaluate the model on test dataset
        
        Args:
            model (tf.keras.Model): Trained model
            test_ds (tf.data.Dataset): Test dataset
        
        Returns:
            tuple: Test loss and accuracy
        """
        test_loss, test_accuracy = model.evaluate(test_ds)
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        return test_loss, test_accuracy

# Example usage function
def main():
    DATA_DIR='data'
    IMG_HEIGHT=150
    IMG_WIDTH=150
    BATCH_SIZE=20
    NUM_CLASSES=5
        #EPOCHS=30

    train_ds_dir=pathlib.Path(os.path.join(DATA_DIR,"train"))
    val_ds_dir=pathlib.Path(os.path.join(DATA_DIR,"valid"))
    test_ds_dir=pathlib.Path(os.path.join(DATA_DIR,"test"))
        
    train_ds, val_ds,test_ds=load_datasets(train_ds_dir,val_dir=val_ds_dir, test_dir=test_ds_dir,img_height=IMG_HEIGHT,img_width=IMG_WIDTH,batch_size=BATCH_SIZE)
    # Get input shape and number of classes
    # Assuming first batch to get image dimensions
    for images, labels in train_ds.take(1):
        input_shape = images.shape[1:]
        num_classes = len(set(labels.numpy()))
    
    # Tune hyperparameters
    tuning_results = CNNClassificationTuner.tune_hyperparameters(
        train_ds, 
        val_ds, 
        input_shape, 
        num_classes,
        max_trials=30,
        epochs=20
    )
    
    # Get best model
    best_model = tuning_results['best_model']
    
    # Evaluate on test dataset
    CNNClassificationTuner.evaluate_model(best_model, test_ds)

if __name__ == '__main__':
    main()