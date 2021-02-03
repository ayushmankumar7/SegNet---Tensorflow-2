import tensorflow as tf 
import numpy as np 


def encoder():
    vgg16 = tf.keras.applications.VGG16(include_top = False, input_shape = (224,224, 3))
