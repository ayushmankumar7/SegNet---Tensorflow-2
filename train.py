import tensorflow as tf 
import numpy as np 
from glob import glob
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard


from model import segnet
from data import load_data 
from preprocessing import load_image_train, load_image_test
from utils.visualize import *


dataset_path = "data/ADEChallengeData2016/ADEChallengeData2016/images/"
training_data = "training/"
val_data = "validation/"

TRAINSET_SIZE = len(glob(dataset_path + training_data + "*.jpg"))
print(f"The Training Dataset contains {TRAINSET_SIZE} images.")

VALSET_SIZE = len(glob(dataset_path + val_data + "*.jpg"))
print(f"The Validation Dataset contains {VALSET_SIZE} images.")

train_dataset, val_dataset = load_data(dataset_path, training_data, val_data )

LR = 1e-4 
EPOCHS = 50 
metrics = ['acc', Recall(), Precision(), iou]
BATCH_SIZE = 32
BUFFER_SIZE = 1000
N_CHANNELS = 3
N_CLASSES = 151 
SEED = 42
AUTOTUNE = tf.data.experimental.AUTOTUNE

dataset = {"train": train_dataset, "val": val_dataset}
print(dataset)

# -- Train Dataset --#
dataset['train'] = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE, seed=SEED)
dataset['train'] = dataset['train'].repeat()
dataset['train'] = dataset['train'].batch(BATCH_SIZE)
dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)

#-- Validation Dataset --#
dataset['val'] = dataset['val'].map(load_image_test)
dataset['val'] = dataset['val'].repeat()
dataset['val'] = dataset['val'].batch(BATCH_SIZE)
dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)

print(dataset['train'])
print(dataset['val'])

for image, mask in dataset['train'].take(1):
    sample_image, sample_mask = image, mask

display_sample([sample_image[0], sample_mask[0]])





m = segnet() 
# print(m.summary())



m.compile(optimizer = tf.keras.optimizers.Adam(LR), loss = "sparse_categorical_crossentropy", metrics = metrics)

callbacks = [
    ModelCheckpoint("files/model.h5"),
    ReduceLROnPlateau(monitor ="val_loss", factor =0.1, patience = 3),
    CSVLogger("files/data.csv"),
    TensorBoard(),
    EarlyStopping(monitor = "val_loss", patience = 10, restore_best_weights = False)
]



# history = m.history(
#     dataset['train'], epochs = 10, validation_data = dataset['val']
# )

