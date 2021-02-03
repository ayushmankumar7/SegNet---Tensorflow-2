import tensorflow as tf 
import numpy as np 


def parse_image(img_path):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels = 3)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    mask_path = tf.strings.regex_replace(img_path, "images", "annotations")
    mask_path = tf.strings.regex_replace(mask_path, "jpg", "png")
    mask = tf.io.read_file(mask_path)

    mask = tf.image.decode_png(mask, channels=1)

    mask = tf.where(mask == 255, np.dtype('uint8').type(0), mask)
    return {'image': image, 'mask': mask}


def load_data(dataset_path, training_data, val_data):
    N_CHANNELS = 3
    N_CLASSES = 151 

    train_dataset = tf.data.Dataset.list_files(dataset_path + training_data + "*.jpg", seed=SEED)
    train_dataset = train_dataset.map(parse_image)

    val_dataset = tf.data.Dataset.list_files(dataset_path + val_data + "*.jpg", seed=SEED)
    val_dataset =val_dataset.map(parse_image)

    return train_dataset, val_dataset

