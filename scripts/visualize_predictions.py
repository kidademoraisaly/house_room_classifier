#from house_room_classifier.data.preprocessing import prepare_dataset
from house_room_classifier.models.room_classifier_model import RoomClassificationModel
from house_room_classifier.data.preprocessing import load_dataset
import pathlib
import os
from house_room_classifier.utils.visualization_data import visualize_first_images_batch
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
   
def main():
    DATA_DIR = "data"
    test_ds_dir = pathlib.Path(os.path.join(DATA_DIR, "test"))
    
    # Load the pre-trained model
    model = tf.keras.models.load_model('models/room_classifier_model_prettrained_resnet50_full_training_v3.keras')
    
    # Load the test dataset
    test_dataset = load_dataset(test_ds_dir, img_height=250, img_width=250)
    class_names = test_dataset.class_names

    # Get a batch of images and labels
    original_image_batch, label_batch = test_dataset.as_numpy_iterator().next()
    
    # Normalize the images for prediction
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    normalized_images = normalization_layer(original_image_batch)

    # Get predictions
    predictions = model.predict_on_batch(normalized_images)
    predicted_classes = np.argmax(predictions, axis=1)  # Get the class index with the highest probability

    # Determine the grid size dynamically
    num_images = len(original_image_batch)
    grid_size = math.ceil(math.sqrt(num_images))  # Get the closest square root
    
    # Visualize the images with predicted and true labels
    plt.figure(figsize=(12, 12))
    for i in range(num_images):
        ax = plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(original_image_batch[i].astype("uint8"))  # Display original image
        pred_label = class_names[predicted_classes[i]]
        true_label = class_names[label_batch[i]]
        plt.title(f"Pred: {pred_label}\nTrue: {true_label}", fontsize=8)
        plt.axis("off")
    plt.tight_layout()
    plt.show()
  
    

if __name__ == "__main__":
    main()
