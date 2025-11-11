import tensorflow as tf
# from tensorflow.keras.layers import Input
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.layers import Conv2DTranspose
# from tensorflow.keras.layers import concatenate
# from tensorflow.keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split


def DenseBlock(input, n_neurons,dropout=None):
    A1 = tf.keras.layers.Dense(n_neurons,activation="relu")(input)
    A2 = tf.keras.layers.Dense(n_neurons,activation="relu")(A1)
    if dropout:
        A2 = tf.keras.layers.Dropout(dropout)(A2)
        
    residual = tf.keras.layers.add([input,A2])

    return residual

def FNN_Compiled(input_size=(4), n_blocks = 4, n_neurons=100, dropout=0.3,name="FNN-DSM"):
    """
    Combine both encoder and decoder blocks according to the U-Net research paper
    Return the model as output
    """
    # Input size represent the size of 1 image (the size used for pre-processing)
    inputs = tf.keras.layers.Input(input_size)
    
    #First layer (just linear)
    y = tf.keras.layers.Dense(n_neurons, activation=None)(inputs)

    #Dense layers
    for i in range(n_blocks):
        y = DenseBlock(y, n_neurons, dropout)
    
    #Output
    output = tf.keras.layers.Dense(1,activation="sigmoid")(y)

    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=output,name=name)

    return model