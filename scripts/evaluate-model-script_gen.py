import tensorflow as tf
from room_classifier.models.cnn_model import RoomClassificationModel
from room_classifier.data.preprocessing import prepare_data_generators
from room_classifier.utils.visualization import plot_confusion_matrix
from room_classifier.utils.metrics import detailed_classification_report
import numpy as np

def main():
    # Load saved model
    model = tf.keras.models.load_model('models/room_classifier_model.h5')
    
    # Prepare test data
    _, _, test_generator = prepare_data_generators(
        'data/processed', 
        img_height=224, 
        img_width=224, 
        batch_size=32
    )

    # Get predictions
    predictions = model.predict(test_generator)
    true_labels = test_generator.classes

    # Convert predictions to class labels
    predicted_classes = np.argmax(predictions, axis=1)

    # Generate detailed report
    classification_rep = detailed_classification_report(
        true_labels, 
        predicted_classes, 
        test_generator.class_indices
    )
    print(classification_rep)

    # Plot confusion matrix
    plot_confusion_matrix(
        true_labels, 
        predicted_classes, 
        classes=list(test_generator.class_indices.keys())
    )

if __name__ == '__main__':
    main()
