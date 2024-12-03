import matplotlib.pyplot as plt


def visualize_data(ds):
    plt.figure(figsize=(10,10))
    for images_batch , labels in ds.take(1):
        for i in range(9):
            ax=plt.subplot(3,3, i+1)
            