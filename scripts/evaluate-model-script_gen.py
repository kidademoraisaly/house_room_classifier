# =====================================================
# 1. Importing Libraries
# =====================================================

import tensorflow as tf
from room_classifier.models.cnn_model import RoomClassificationModel
from room_classifier.data.preprocessing import prepare_data_generators
from room_classifier.utils.visualization import plot_confusion_matrix
from room_classifier.utils.metrics import detailed_classification_report
import numpy as np

# =====================================================
# 2. Model Evaluation and Testing
# =====================================================

def main():

    # Step 1: Load the Saved Model
    model = tf.keras.models.load_model('models/room_classifier_model.h5') # Loads the previously trained and saved model from the file path
    
    # Step 2: Prepare the Test Data
    _, _, test_generator = prepare_data_generators(    # It is a function that prepares the data loaders for the test set
        'data/processed', 
        img_height=224,     # The test images are resized to 224x224 pixels
        img_width=224,      
        batch_size=32       # And processed in batches of size 32
    )

    # Step 3: Get Model Predictions
    predictions = model.predict(test_generator)       # Passes the test dataset to the model to generate predictions
    true_labels = test_generator.classes              # Provides the true labels (ground truth) of all the images in the test dataset

    # Step 4: Convert Predictions to Class Indices
    predicted_classes = np.argmax(predictions, axis=1)  # Finds the index of the highest probability for each prediction
                                                        # and converts the probabilities into predicted class labels

    # Step 5: Generate a Detailed Classification Report
    classification_rep = detailed_classification_report(   # A custom function to compute and print metrics such as Precision, Recall, F1-score and Support (number of true samples per class)
        true_labels,             # true_labels: The ground truth labels
        predicted_classes,       # predicted_classes: The predicted class labels
        test_generator.class_indices   # test_generator.class_indices: A dictionary mapping class names to numeric indices
    )

    print(classification_rep)     # The output is a summary report showing model performance for each class

    # Step 6: Plot the Confusion Matrix
    plot_confusion_matrix(
        true_labels,                   # Rows represent actual class labels and columns represent predicted class labels
        predicted_classes, 
        classes=list(test_generator.class_indices.keys())
    )

# Step 7: Ensures the main() function runs when the script is executed directly
if __name__ == '__main__':
    main()
