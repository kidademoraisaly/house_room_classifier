#from house_room_classifier.data.preprocessing import prepare_dataset
from house_room_classifier.models.room_classifier_model import RoomClassificationModel
from house_room_classifier.data.preprocessing import load_dataset
import pathlib
import os
from house_room_classifier.utils.visualization_data import visualize_first_images_batch
import tensorflow as tf

def main():
    DATA_DIR="data"
    test_ds_dir=pathlib.Path(os.path.join(DATA_DIR,"test"))
    model = tf.keras.models.load_model('models/room_classifier_model.keras')
    test_dataset=load_dataset(test_ds_dir)
    class_names=test_dataset.class_names
    image_batch, label_batch = test_dataset.as_numpy_iterator().next()
    predictions = model.predict_on_batch(image_batch).flatten()
    predictions = tf.where(predictions < 0.5, 0, 1)

    print('Predictions:\n', predictions.numpy())
    print('Labels:\n', label_batch)

    visualize_first_images_batch(image_batch,class_names,labels=predictions,num_images=9)
if __name__=='__main__':
    main()