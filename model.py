import tensorflow as tf 
import numpy as np 
from custom_layers import *


def encoder(input_shape, kernel = 3, pool_size=(2,2)):
    inputs = tf.keras.layers.Input(shape=input_shape)

    conv_1 = tf.keras.layers.Conv2D(64, (kernel, kernel), padding="same")(inputs)
    conv_1 = tf.keras.layers.BatchNormalization()(conv_1)
    conv_1 = tf.keras.layers.Activation("relu")(conv_1)
    conv_2 = tf.keras.layers.Conv2D(64, (kernel, kernel), padding="same")(conv_1)
    conv_2 = tf.keras.layers.BatchNormalization()(conv_2)
    conv_2 = tf.keras.layers.Activation("relu")(conv_2)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

    conv_3 = tf.keras.layers.Conv2D(128, (kernel, kernel), padding="same")(pool_1)
    conv_3 = tf.keras.layers.BatchNormalization()(conv_3)
    conv_3 = tf.keras.layers.Activation("relu")(conv_3)
    conv_4 = tf.keras.layers.Conv2D(128, (kernel, kernel), padding="same")(conv_3)
    conv_4 = tf.keras.layers.BatchNormalization()(conv_4)
    conv_4 = tf.keras.layers.Activation("relu")(conv_4)

    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

    conv_5 = tf.keras.layers.Conv2D(256, (kernel, kernel), padding="same")(pool_2)
    conv_5 = tf.keras.layers.BatchNormalization()(conv_5)
    conv_5 = tf.keras.layers.Activation("relu")(conv_5)
    conv_6 = tf.keras.layers.Conv2D(256, (kernel, kernel), padding="same")(conv_5)
    conv_6 = tf.keras.layers.BatchNormalization()(conv_6)
    conv_6 = tf.keras.layers.Activation("relu")(conv_6)
    conv_7 = tf.keras.layers.Conv2D(256, (kernel, kernel), padding="same")(conv_6)
    conv_7 = tf.keras.layers.BatchNormalization()(conv_7)
    conv_7 = tf.keras.layers.Activation("relu")(conv_7)

    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)

    conv_8 = tf.keras.layers.Conv2D(512, (kernel, kernel), padding="same")(pool_3)
    conv_8 = tf.keras.layers.BatchNormalization()(conv_8)
    conv_8 = tf.keras.layers.Activation("relu")(conv_8)
    conv_9 = tf.keras.layers.Conv2D(512, (kernel, kernel), padding="same")(conv_8)
    conv_9 = tf.keras.layers.BatchNormalization()(conv_9)
    conv_9 = tf.keras.layers.Activation("relu")(conv_9)
    conv_10 = tf.keras.layers.Conv2D(512, (kernel, kernel), padding="same")(conv_9)
    conv_10 = tf.keras.layers.BatchNormalization()(conv_10)
    conv_10 = tf.keras.layers.Activation("relu")(conv_10)

    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)

    conv_11 = tf.keras.layers.Conv2D(512, (kernel, kernel), padding="same")(pool_4)
    conv_11 = tf.keras.layers.BatchNormalization()(conv_11)
    conv_11 = tf.keras.layers.Activation("relu")(conv_11)
    conv_12 = tf.keras.layers.Conv2D(512, (kernel, kernel), padding="same")(conv_11)
    conv_12 = tf.keras.layers.BatchNormalization()(conv_12)
    conv_12 = tf.keras.layers.Activation("relu")(conv_12)
    conv_13 = tf.keras.layers.Conv2D(512, (kernel, kernel), padding="same")(conv_12)
    conv_13 = tf.keras.layers.BatchNormalization()(conv_13)
    conv_13 = tf.keras.layers.Activation("relu")(conv_13)

    pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)


    return inputs, [pool_5, mask_1, mask_2, mask_3, mask_4, mask_5]

def decoder(input_shape, n_class, encoder_inputs, kernel = 3, pool_size=(2,2)):
    
    unpool_1 = MaxUnpooling2D(pool_size)([encoder_inputs[0], encoder_inputs[5]])
    print(unpool_1)
    conv_14 = tf.keras.layers.Conv2D(512, (kernel, kernel), padding="same")(unpool_1)
    conv_14 = tf.keras.layers.BatchNormalization()(conv_14)
    conv_14 = tf.keras.layers.Activation("relu")(conv_14)
    conv_15 = tf.keras.layers.Conv2D(512, (kernel, kernel), padding="same")(conv_14)
    conv_15 = tf.keras.layers.BatchNormalization()(conv_15)
    conv_15 = tf.keras.layers.Activation("relu")(conv_15)
    conv_16 = tf.keras.layers.Conv2D(512, (kernel, kernel), padding="same")(conv_15)
    conv_16 = tf.keras.layers.BatchNormalization()(conv_16)
    conv_16 = tf.keras.layers.Activation("relu")(conv_16)

    unpool_2 = MaxUnpooling2D(pool_size)([conv_16, encoder_inputs[4]])

    conv_17 = tf.keras.layers.Conv2D(512, (kernel, kernel), padding="same")(unpool_2)
    conv_17 = tf.keras.layers.BatchNormalization()(conv_17)
    conv_17 = tf.keras.layers.Activation("relu")(conv_17)
    conv_18 = tf.keras.layers.Conv2D(512, (kernel, kernel), padding="same")(conv_17)
    conv_18 = tf.keras.layers.BatchNormalization()(conv_18)
    conv_18 = tf.keras.layers.Activation("relu")(conv_18)
    conv_19 = tf.keras.layers.Conv2D(256, (kernel, kernel), padding="same")(conv_18)
    conv_19 = tf.keras.layers.BatchNormalization()(conv_19)
    conv_19 = tf.keras.layers.Activation("relu")(conv_19)

    unpool_3 = MaxUnpooling2D(pool_size)([conv_19, encoder_inputs[3]])

    conv_20 = tf.keras.layers.Conv2D(256, (kernel, kernel), padding="same")(unpool_3)
    conv_20 = tf.keras.layers.BatchNormalization()(conv_20)
    conv_20 = tf.keras.layers.Activation("relu")(conv_20)
    conv_21 = tf.keras.layers.Conv2D(256, (kernel, kernel), padding="same")(conv_20)
    conv_21 = tf.keras.layers.BatchNormalization()(conv_21)
    conv_21 = tf.keras.layers.Activation("relu")(conv_21)
    conv_22 = tf.keras.layers.Conv2D(128, (kernel, kernel), padding="same")(conv_21)
    conv_22 = tf.keras.layers.BatchNormalization()(conv_22)
    conv_22 = tf.keras.layers.Activation("relu")(conv_22)

    unpool_4 = MaxUnpooling2D(pool_size)([conv_22, encoder_inputs[2]])

    conv_23 = tf.keras.layers.Conv2D(128, (kernel, kernel), padding="same")(unpool_4)
    conv_23 = tf.keras.layers.BatchNormalization()(conv_23)
    conv_23 = tf.keras.layers.Activation("relu")(conv_23)
    conv_24 = tf.keras.layers.Conv2D(64, (kernel, kernel), padding="same")(conv_23)
    conv_24 = tf.keras.layers.BatchNormalization()(conv_24)
    conv_24 = tf.keras.layers.Activation("relu")(conv_24)

    unpool_5 = MaxUnpooling2D(pool_size)([conv_24, encoder_inputs[1]])

    conv_25 = tf.keras.layers.Conv2D(64, (kernel, kernel), padding="same")(unpool_5)
    conv_25 = tf.keras.layers.BatchNormalization()(conv_25)
    conv_25 = tf.keras.layers.Activation("relu")(conv_25)

    conv_26 = tf.keras.layers.Conv2D(n_class, (1, 1), padding="valid")(conv_25)
    conv_26 = tf.keras.layers.BatchNormalization()(conv_26)
    conv_26 = tf.keras.layers.Reshape(
        (input_shape[0] * input_shape[1], n_class),
        input_shape=(input_shape[0], input_shape[1], n_class),
    )(conv_26)
    x = tf.keras.layers.Activation("softmax")(conv_26)

    return x

def segnet(input_shape, n_class):
    
    inp, enc = encoder(input_shape)
    print("\n\n")
    for i in enc:
        print(i)
    print(len(enc))
    dec = decoder(input_shape = input_shape,n_class = 45,encoder_inputs =  enc)
    print("\n\n\n\n\n")
    

    model = tf.keras.models.Model(inputs = inp, outputs = dec, name = "SegNet")
    
    return model 
