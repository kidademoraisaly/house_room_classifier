mport tensorflow as tf
from tensorflow import keras
from kerastuner import BayesianOptimization
from kerastuner.engine.hyperparameters import HyperParameters

class PretrainedModelTuner:
    @staticmethod
    def build_model(hp, base_model, input_shape, num_classes):
        """
        Dynamically build and compile a transfer learning model
        
        Args:
            hp (HyperParameters): Hyperparameters to tune
            base_model (tf.keras.Model): Pretrained base model
            input_shape (tuple): Input image shape
            num_classes (int): Number of classification classes
        
        Returns:
            tf.keras.Model: Compiled transfer learning model
        """
        # Freeze base model with tunable freezing strategy
        freezing_layers = hp.Choice('freezing_strategy', 
            ['partial', 'full', 'fine_tune']
        )
        
        if freezing_layers == 'full':
            base_model.trainable = False
        elif freezing_layers == 'partial':
            # Freeze a portion of base model layers
            for layer in base_model.layers[:hp.Int('freeze_layers', 0, len(base_model.layers)//2)]:
                layer.trainable = False
        else:  # fine_tune
            base_model.trainable = True
        
        # Build model on top of base model
        model = keras.Sequential()
        model.add(base_model)
        
        # Global Average Pooling
        model.add(keras.layers.GlobalAveragePooling2D())
        
        # Dynamically add dense layers
        num_dense_layers = hp.Int('num_dense_layers', 1, 3)
        for i in range(num_dense_layers):
            model.add(keras.layers.Dense(
                hp.Int(f'dense_units_{i}', 64, 512, step=64),
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(
                    hp.Float(f'l2_reg_{i}', 1e-4, 1e-2, sampling='log')
                )
            ))
            
            # Optional dropout
            if hp.Boolean(f'add_dropout_{i}'):
                model.add(keras.layers.Dropout(
                    hp.Float(f'dropout_rate_{i}', 0.2, 0.5, step=0.1)
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
    def tune_pretrained_model(
        train_ds, 
        val_ds, 
        input_shape, 
        num_classes, 
        base_model_func=None,
        max_trials=50, 
        epochs=20
    ):
        """
        Perform Bayesian optimization for pretrained model fine-tuning
        
        Args:
            train_ds (tf.data.Dataset): Training dataset
            val_ds (tf.data.Dataset): Validation dataset
            input_shape (tuple): Input image shape
            num_classes (int): Number of classification classes
            base_model_func (callable, optional): Function to create base model
            max_trials (int): Maximum number of hyperparameter combinations
            epochs (int): Number of epochs to train each model
        
        Returns:
            dict: Best hyperparameters and model
        """
        # Default to MobileNetV2 if no base model function provided
        if base_model_func is None:
            base_model_func = lambda: tf.keras.applications.MobileNetV2(
                input_shape=input_shape,
                include_top=False,
                weights='imagenet'
            )
        
        # Create base model
        base_model = base_model_func()
        
        # Initialize Bayesian Optimization Tuner
        tuner = BayesianOptimization(
            hyperparameters=None,
            objective='val_accuracy',
            max_trials=max_trials,
            directory='pretrained_tuning_results',
            project_name='transfer_learning_tuning'
        )
        
        # Define tuning method
        def build_model_wrapper(hp):
            return PretrainedModelTuner.build_model(
                hp, base_model, input_shape, num_classes
            )
        
        tuner.hyperparameters = HyperParameters()
        
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
            'best_hyperparameters': best_hps,
            'base_model': base_model
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
    # Load your datasets
    train_dir = 'path/to/train/directory'
    train_ds, val_ds, test_ds = load_datasets(
        train_dir, 
        img_height=224,  # Typical size for many pretrained models 
        img_width=224, 
        batch_size=32
    )
    
    # Get input shape and number of classes
    # Assuming first batch to get image dimensions
    for images, labels in train_ds.take(1):
        input_shape = images.shape[1:]
        num_classes = len(set(labels.numpy()))
    
    # Tune pretrained MobileNetV2
    tuning_results = PretrainedModelTuner.tune_pretrained_model(
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
    PretrainedModelTuner.evaluate_model(best_model, test_ds)
    
    # Optional: You can also use a different base model
    def custom_base_model():
        return tf.keras.applications.ResNet50V2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
    
    # Tune with a different base model
    alternative_results = PretrainedModelTuner.tune_pretrained_model(
        train_ds, 
        val_ds, 
        input_shape, 
        num_classes,
        base_model_func=custom_base_model,
        max_trials=30,
        epochs=20
    )

if __name__ == '__main__':
    main()