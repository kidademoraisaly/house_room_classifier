from house_room_classifier.data.preprocessing import prepare_dataset
from house_room_classifier.models.room_classifier_model import RoomClassificationModel
from house_room_classifier.data.preprocessing import load_dataset
import pathlib
import os
from house_room_classifier.utils.visualization_data import visualize_first_images_batch
import tensorflow as tf
import numpy as np
from house_room_classifier.utils.visualization_data import plot_training_results

def main():
    DATA_DIR="data"
    test_ds_dir=pathlib.Path(os.path.join(DATA_DIR,"test"))

    room_classifier=RoomClassificationModel(img_height=150,img_width=150,num_classes=5)
    room_classifier.load_model('models/room_classifier_model.keras')
    test_dataset=load_dataset(test_ds_dir)
    print(room_classifier.evaluate(test_dataset))
    


if __name__ == '__main__':
    main()
