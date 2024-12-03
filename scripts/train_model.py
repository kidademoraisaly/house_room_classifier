from house_room_classifier.data.preprocessing import prepare_dataset
from house_room_classifier.models.room_classifier_model import RoomClassificationModel

def main():
    print("Main")

        # DATA_DIR='data'
        # IMG_HEIGHT=150
        # IMG_WIDTH=150
        # BATCH_SIZE=20
        # NUM_CLASSES=5

        # train_ds, val_ds=prepare_dataset(data_dir=DATA_DIR,img_height=IMG_HEIGHT,img_width=IMG_WIDTH,batch_size=BATCH_SIZE)

        # room_classifier=RoomClassificationModel(
        #     img_height=IMG_HEIGHT,
        #     img_width=IMG_WIDTH,
        #     num_classes=NUM_CLASSES
        # )
        # room_classifier.build_model()

        # history=room_classifier.train(
        #     train_ds,
        #     val_ds,
        #     epochs=20
        # )


        # #results=room_classifier.evaluate(test_ds)
        # #print("Test results", results)

        # room_classifier.model.save('models/room_classifier_model.h5')

if __name__=='__main__':
    main()