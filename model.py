import tensorflow as tf 
import numpy as np 


def encoder():
    vgg16 = tf.keras.applications.VGG16(include_top = False, input_shape = (224,224, 3))
    b4 = vgg16.get_layer("block4_pool").output
    return vgg16, b4

def decoder(input, n_class):
    
    x = tf.keras.layers.ZeroPadding2D((1,1))(input)
    x = tf.keras.layers.Conv2D(512, (3,3), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    for i in range(1, 5):
        
        x = tf.keras.layers.UpSampling2D((2,2))(x)
        x = tf.keras.layers.ZeroPadding2D((1,1))(x)
        x = tf.keras.layers.Conv2D(int(512/2**i), (3,3), padding='valid')(x)
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(n_class, (3,3), padding = 'same')(x)

    return x 

def segnet():
    
    enc, b4 = encoder()
    dec = decoder(b4, 45)

    model = tf.keras.models.Model(inputs = enc.input, outputs = dec)
    
    return model 
