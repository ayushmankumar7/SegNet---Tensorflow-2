import tensorflow as tf 
import matplotlib.pyplot as plt

def display_sample(img_list):
    plt.figure(figsize=(18,18))
    title = ['Input Image', "True Mask", "Predicted Mask"]

    for i in range(len(img_list)):
        plt.subplot(1, len(img_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_image(img_list[i]))
        plt.axis('off')

    plt.show()

