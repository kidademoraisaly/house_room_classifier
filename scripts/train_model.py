from house_room_classifier.data.preprocessing import  load_datasets
from house_room_classifier.models.room_classifier_model import RoomClassificationModel
import pathlib
import os
from house_room_classifier.utils.visualization_data import plot_training_results
import tensorflow as tf

def main():
        
        DATA_DIR='data'
        IMG_HEIGHT=150
        IMG_WIDTH=150
        BATCH_SIZE=20
        NUM_CLASSES=5
        train_ds_dir=pathlib.Path(os.path.join(DATA_DIR,"train"))
        val_ds_dir=pathlib.Path(os.path.join(DATA_DIR,"valid"))
        test_ds_dir=pathlib.Path(os.path.join(DATA_DIR,"test"))
        

        train_ds, val_ds,test_ds=load_datasets(train_ds_dir,val_dir=val_ds_dir, test_dir=test_ds_dir,img_height=IMG_HEIGHT,img_width=IMG_WIDTH,batch_size=BATCH_SIZE)

        print("Shape train_ds", tf.data.experimental.cardinality(train_ds).numpy())
        print("Shape val_ds", tf.data.experimental.cardinality(val_ds).numpy())
        print("Shape test_ds", tf.data.experimental.cardinality(test_ds).numpy())
        
        image_batch,labels_batch=next(iter(train_ds))
        first_image=image_batch[0]
        print(first_image)
        
        room_classifier=RoomClassificationModel(
            img_height=IMG_HEIGHT,
            img_width=IMG_WIDTH,
            num_classes=NUM_CLASSES,
            architecture="pretrained_resnet50_fine_v1"
        )
        
        room_classifier.build_model()

        history=room_classifier.train(
             train_ds,
             val_ds,
             
         )
        plot_training_results(history)
        room_classifier.model.save('models/room_classifier_model_pretrained_resnet50_fine_v1.keras')





if __name__=='__main__':
    main()