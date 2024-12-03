import pathlib
import os
import PIL
import PIL.Image
import matplotlib.pyplot as plt


def main():
    DATA_DIR="data"
    train_ds_dir=pathlib.Path(os.path.join(DATA_DIR,"train"))
    val_ds_dir=pathlib.Path(os.path.join(DATA_DIR,"valid"))
    test_ds_dir=pathlib.Path(os.path.join(DATA_DIR,"test"))
    
    #Training set
    image_count=len(list(train_ds_dir.glob('*/*.jpg')))
    print(f"Total number of images on training: {image_count}")
    baths_count=len(list(train_ds_dir.glob("Bathroom/*")))
    print(f"Total number of Bathrooms on training : {baths_count}")


    #Validation set
    image_count=len(list(val_ds_dir.glob('*/*.jpg')))
    print(f"Total number of images on validation: {image_count}")
    baths_count=len(list(val_ds_dir.glob("Bathroom/*")))
    print(f"Total number of Bathrooms on validation : {baths_count}")

    #Test set
    image_count=len(list(test_ds_dir.glob('*/*.jpg')))
    print(f"Total number of images on test: {image_count}")
    baths_count=len(list(test_ds_dir.glob("Bathroom/*")))
    print(f"Total number of Bathrooms on test : {baths_count}")

    bath_rooms=list(val_ds_dir.glob("Bathroom/*"))
    image=PIL.Image.open(str(bath_rooms[0]))
     
    image.show()
    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()

if __name__=="__main__":
    main()