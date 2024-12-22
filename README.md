# House Room Classifier

## Directory Structure

### Data Directory (`house_room_classifier/house_room_classifier/data`)
- **`data_exploration.py`**: Contains functions for exploring the dataset.
- **`preprocessing.py`**: Includes functions for preprocessing the data.

### Models Directory (`house_room_classifier/house_room_classifier/models`)
- **`model_architectures.py`**: Defines all the trained models and their architectures.
- **`room_classifier_model.py`**: Used for building, training, and saving all the models.
- **`training_config.py`**: Sets the configurations for hyperparameters.

### Utilities Directory (`house_room_classifier/house_room_classifier/utils`)
- **`visualization_data.py`**: Contains custom visualization functions, later used in the notebooks.

### Notebooks Directory (`house_room_classifier/notebooks`)
- **`exploration_preprocessing.ipynb`**: Notebook for data exploration and preprocessing.
- **`model_training.ipynb`**: Notebook for training the models.
- **`predictions.ipynb`**: Notebook for making predictions on test data and evaluating the final models.

### Scripts Directory (`house_room_classifier/scripts`)
- Contains various `.py` files for testing functions and methods before integrating them into the notebooks.
