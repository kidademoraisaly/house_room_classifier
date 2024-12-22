import matplotlib.pyplot as plt
import math
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import confusion_matrix

MAX_COLS=6

def visualize_first_images_batch(image_batch,class_names,labels,num_images=9):
    available_images = len(image_batch)
    if num_images > available_images:
        print(f"Requested {num_images} images, but only {available_images} are available. Displaying {available_images}.")
        num_images = available_images

    num_cols = min(MAX_COLS, math.floor(math.sqrt(num_images)))  # Limit columns to MAX_COLS
    num_rows = math.ceil(num_images / num_cols) 
   
    plt.figure(figsize=(10,10))
    for i in range(num_images):
        ax=plt.subplot(num_rows,num_cols, i+1)
        plt.imshow(image_batch[i].astype("uint8"))
        #plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
    plt.tight_layout()  # Avoid overlapping titles
    plt.show()


def visualize_first_images(ds, class_names,num_images=9):
    for images_batch , labels in ds.take(1):
        print(f"Batch shape: {images_batch.shape}, Labels shape: {labels.shape}")
        images_batch=images_batch.numpy()
        visualize_first_images_batch(images_batch,class_names,labels,num_images)
    

def visualize_first_image_augmented(ds, data_augmentation, n_times=9):
    for image, _ in ds.take(1):
        plt.figure(figsize=(10, 10))
        first_image = image[0]
        for i in range(n_times):
            ax = plt.subplot(3, 3, i + 1)
            augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
            plt.imshow(augmented_image[0] / 255)
            plt.axis('off')



def plot_training_results(history):
    #print(f"Epochs ---------------{epochs}")
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = len(history.history['accuracy'])
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def plot_confusion_matrix(true_labels, predicted_labels, classes):
    """
    Plot and save confusion matrix
    """
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=classes, 
        yticklabels=classes
    )
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png')
    plt.close()

                