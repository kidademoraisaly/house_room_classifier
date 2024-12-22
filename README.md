# House Room Classifier

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Models and Architectures](#models-and-architectures)
4. [Installation](#installation)
6. [Directory Structure](#directory-structure)
7. [Credits](#credits)
8. [License](#license)

## 1. Project Overview

This repository contains the implementation of a deep learning-based room classification system for room type images. Using both custom Convolutional Neural Networks (CNN) and pretrained models, this project explores various techniques to optimize accuracy and minimize overfitting.

## 2. Dataset

The project uses two datasets:
- **Small Dataset**: Initially used for training and evaluation (5,192 images).
- **Large Dataset**: Larger and more balanced (130,995 images).

## 3. Models and Architectures

- **Simple CNN Model**: A lightweight CNN model for quick experimentation.
- **Complex CNN Model**: A deeper custom CNN with additional layers and regularization.  
- **Pretrained Models**: MobileNetV2 and ResNet50.

## 4. Installation

### 4.1. Prerequisites
- Python 3.7+
- [Poetry](https://python-poetry.org/) for dependency management.

### 4.2. Steps

- **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/house_room_classifier.git
   cd house_room_classifier
    ```

- **Install Poetry**:
Follow the official installation guide (https://python-poetry.org/docs/)

## 5. Directory Structure

- **Data Directory** (`house_room_classifier/house_room_classifier/data`)
  - **`data_exploration.py`**: Contains functions for exploring the dataset.
  - **`preprocessing.py`**: Includes functions for preprocessing the data.

- **Models Directory** (`house_room_classifier/house_room_classifier/models`)
  - **`model_architectures.py`**: Defines all the trained models and their architectures.
  - **`room_classifier_model.py`**: Used for building, training, and saving all the models.
  - **`training_config.py`**: Sets the configurations for hyperparameters.

- **Utilities Directory** (`house_room_classifier/house_room_classifier/utils`)
  - **`visualization_data.py`**: Contains custom visualization functions, later used in the notebooks.

- **Notebooks Directory** (`house_room_classifier/notebooks`)
  - **`exploration_preprocessing.ipynb`**: Notebook for data exploration and preprocessing.
  - **`model_training.ipynb`**: Notebook for training the models.
  - **`predictions.ipynb`**: Notebook for making predictions on test data and evaluating the final models.

- **Scripts Directory** (`house_room_classifier/scripts`)
  - Contains various `.py` files for testing functions and methods before integrating them into the notebooks.

## 6. Credits

- **Group 19**

- **Contributors:**
  * Baran Çelik (20232067)
  * Kida Aly (20231491)
  * Raysa Rocha (20232051)

## 7. License
This project is licensed under the [MIT Licence](https://choosealicense.com/licenses/mit/)