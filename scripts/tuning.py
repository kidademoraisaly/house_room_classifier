import tensorflow as tf
from tensorflow import keras
from kerastuner import RandomSearch, BayesianOptimization
from kerastuner.engine.hyperparameters import HyperParameters

# Prepare your dataset (example with MNIST)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

def build_model(hp):
    """
    Build a CNN model with hyperparameters to be tuned
    
    Args:
        hp (HyperParameters): Hyperparameters to be tuned
    
    Returns:
        tf.keras.Model: Compiled CNN model
    """
    model = keras.Sequential()
    
    # Dynamically add convolutional layers
    for i in range(hp.Int('num_conv_layers', 1, 3)):
        model.add(keras.layers.Conv2D(
            filters=hp.Int(f'conv_{i}_filters', 32, 128, step=32),
            kernel_size=hp.Choice(f'conv_{i}_kernel', [3, 5]),
            activation='relu',
            input_shape=(28, 28, 1) if i == 0 else None
        ))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(keras.layers.Flatten())
    
    # Tunable dense layers
    for i in range(hp.Int('num_dense_layers', 1, 3)):
        model.add(keras.layers.Dense(
            units=hp.Int(f'dense_{i}_units', 64, 512, step=64),
            activation='relu'
        ))
        model.add(keras.layers.Dropout(
            hp.Float(f'dropout_{i}', 0, 0.3, step=0.1)
        ))
    
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    # Tune learning rate
    learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Random Search Tuner
def random_search_tuning():
    """
    Perform hyperparameter tuning using Random Search
    
    Returns:
        Tuner results
    """
    random_tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=10,  # Number of different hyperparameter combinations to try
        executions_per_trial=3,  # Number of models that should be built and evaluated for each trial
        directory='random_search_results',
        project_name='mnist_cnn_random_search'
    )
    
    # Perform the search
    random_tuner.search(
        x_train, y_train,
        epochs=10,
        validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3)
        ]
    )
    
    # Get and print the best hyperparameters
    best_hps = random_tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\nRandom Search Best Hyperparameters:")
    print(f"Number of Conv Layers: {best_hps.get('num_conv_layers')}")
    print(f"Learning Rate: {best_hps.get('learning_rate')}")
    
    # Get the best model
    best_model = random_tuner.get_best_models(num_models=1)[0]
    return best_model, best_hps

# Bayesian Optimization Tuner
def bayesian_search_tuning():
    """
    Perform hyperparameter tuning using Bayesian Optimization
    
    Returns:
        Tuner results
    """
    bayesian_tuner = BayesianOptimization(
        build_model,
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=3,
        directory='bayesian_search_results',
        project_name='mnist_cnn_bayesian_search'
    )
    
    # Perform the search
    bayesian_tuner.search(
        x_train, y_train,
        epochs=10,
        validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3)
        ]
    )
    
    # Get and print the best hyperparameters
    best_hps = bayesian_tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\nBayesian Optimization Best Hyperparameters:")
    print(f"Number of Conv Layers: {best_hps.get('num_conv_layers')}")
    print(f"Learning Rate: {best_hps.get('learning_rate')}")
    
    # Get the best model
    best_model = bayesian_tuner.get_best_models(num_models=1)[0]
    return best_model, best_hps

# Main execution
def main():
    # Perform Random Search
    random_best_model, random_best_hps = random_search_tuning()
    
    # Perform Bayesian Optimization
    bayesian_best_model, bayesian_best_hps = bayesian_search_tuning()
    
    # Evaluate best models
    print("\nRandom Search Best Model Evaluation:")
    random_best_model.evaluate(x_test, y_test)
    
    print("\nBayesian Optimization Best Model Evaluation:")
    bayesian_best_model.evaluate(x_test, y_test)

if __name__ == '__main__':
    main()