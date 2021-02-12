import tensorflow as tf 
import numpy as np 

from model import segnet
from data import load_data 
from preprocessing import load_image_train, load_image_test

dataset_path = "data/ADEChallengeData2016/ADEChallengeData2016/images/"
training_data = "training/"
val_data = "validation/"

train_dataset, val_dataset = load_data(dataset_path, training_data, val_data )

BATCH_SIZE = 32
BUFFER_SIZE = 1000
N_CHANNELS = 3
N_CLASSES = 151 
SEED = 42
AUTOTUNE = tf.data.experimental.AUTOTUNE

dataset = {"train": train_dataset, "val": val_dataset}
print(dataset)

# -- Train Dataset --#
# dataset['train'] = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE, seed=SEED)
# dataset['train'] = dataset['train'].repeat()
# dataset['train'] = dataset['train'].batch(BATCH_SIZE)
# dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)

# #-- Validation Dataset --#
# dataset['val'] = dataset['val'].map(load_image_test)
# dataset['val'] = dataset['val'].repeat()
# dataset['val'] = dataset['val'].batch(BATCH_SIZE)
# dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)

# print(dataset['train'])
# print(dataset['val'])

m = segnet() 
# print(m.summary())


m.compile(optimizer = tf.keras.optimizers.Adam(), loss = "sparse_categorical_crossentropy", metrics = ['accuracy'])

# history = m.history(
#     dataset['train'], epochs = 10, validation_data = dataset['val']
# )
