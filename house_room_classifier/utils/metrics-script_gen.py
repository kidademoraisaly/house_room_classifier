from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def detailed_classification_report(true_labels, predicted_labels, class_indices):
    """
    Generate a detailed classification report
    """
    report = classification_report(
        true_labels, 
        predicted_labels, 
        target_names=list(class_indices.keys())
    )
    
    # Additional custom metrics calculation
    cm = confusion_matrix(true_labels, predicted_labels)
    misclassification_rates = 1 - np.diag(cm) / cm.sum(axis=1)
    
    return {
        'classification_report': report,
        'misclassification_rates': dict(zip(
            class_indices.keys(), 
            misclassification_rates
        ))
    }
