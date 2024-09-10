'Prepare Data for training'

import math
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import json
import os

SETTINGS_PATH = "unet_train_settings.json"

with open(SETTINGS_PATH) as f:
    settings = json.loads(f.read())

per = settings['split_percentage']                               # percentage used in training

# T1 = np.load('EIT_Data_for_CNN.npy')
with open(settings['tfrecordpath']+"/data_info.json") as f:
    data_info = json.loads(f.read())

# filename_list = os.listdir(DATAPATH)
# n_samples = len(filename_list)
# n_train = math.floor(n_samples*per)    # number samples for training
# print('Number of samples for training: ' + str(n_train))
# n_val = n_samples - n_train        # number of samples for validation

n_g = settings['n_g']
n_samples = data_info['n_samples']
n_train = data_info['n_train']
n_val = data_info['n_val']

print("Number of currents:",n_g)
print("Number of samples:", n_samples)
print('Number of samples for training: ' + str(n_train))

image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'currents': tf.io.FixedLenFeature([], tf.int64),
    'sample_raw': tf.io.FixedLenFeature([], tf.string),
    'admitivity_raw': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
  # Parse the input tf.train.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, image_feature_description)

def _parse_image_tensor(image_features):
  height = image_features['height']
  width = image_features['width']
  depth = image_features['depth']
  sample_array_raw = tf.io.decode_raw(image_features['sample_raw'],tf.float64)
  # sample_array_raw = np.frombuffer(image_features['sample_raw'].numpy())
  sample_array = tf.reshape(sample_array_raw,[height,width,depth])
  # admitivity_raw = np.frombuffer(image_features['admitivity_raw'].numpy())
  admitivity_raw = tf.io.decode_raw(image_features['admitivity_raw'],tf.float64)
  admitivity = tf.reshape(admitivity_raw,[height,width])
  return sample_array,admitivity


def create_sample_dataset(record_file,batch_size=settings['batch_size']):
  raw_image_dataset = tf.data.TFRecordDataset(record_file)
  parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
  parsed_image_dataset = parsed_image_dataset.map(_parse_image_tensor)
  parsed_image_dataset = parsed_image_dataset.batch(batch_size)
  return parsed_image_dataset.prefetch(1)

tfrecord_dirpath = settings['tfrecordpath']
dataset = create_sample_dataset(tfrecord_dirpath+"/train.tfrecords")
dataset_val = create_sample_dataset(tfrecord_dirpath+"/validation.tfrecords")

# permute the lines
# perm = np.random.permutation(n_samples)

# for i in division:
#   file_data = np.load(DATAPATH+"/"+filename_list[i])
#   n_g = file_data.shape[0]-3
#   # input_train.append(np.transpose(file_data[:n_g + 2]))
#   # label_train.append(np.transpose(file_data[n_g + 2:]))
#   input_train.append(np.transpose(file_data[:n_g + 2]))
#   label_train.append(np.transpose(file_data[n_g + 2:]))
  
# input_train = tf.convert_to_tensor(input_train)
# label_train = tf.convert_to_tensor(label_train)

# for i in division2:
#   file_data = np.load(DATAPATH+"/"+filename_list[i])
#   n_g = file_data.shape[0]-3
#   input_val.append(np.transpose(file_data[:n_g + 2]))
#   label_val.append(np.transpose(file_data[n_g + 2:]))
  
# input_val = tf.convert_to_tensor(input_val)
# label_val = tf.convert_to_tensor(label_val)

# for data load
import os

# for reading and processing images
import imageio
from PIL import Image

# for visualizations
import matplotlib.pyplot as plt

import numpy as np # for using np arrays

# for bulding and running deep learning model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split

'Unet - Encoder block'

def EncoderMiniBlock(inputs, n_filters=32, dropout_prob=settings['dropout_prob'], max_pooling=True):
    """
    This block uses multiple convolution layers, max pool, relu activation to create an architecture for learning.
    Dropout can be added for regularization to prevent overfitting.
    The block returns the activation values for next layer along with a skip connection which will be used in the decoder
    """
    # Add 2 Conv Layers with relu activation and HeNormal initialization using TensorFlow
    # Proper initialization prevents from the problem of exploding and vanishing gradients
    # 'Same' padding will pad the input to conv layer such that the output has the same height and width (hence, is not reduced in size)
    conv = Conv2D(n_filters,
                  3,   # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(inputs)
    conv = Conv2D(n_filters,
                  3,   # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(conv)

    # Batch Normalization will normalize the output of the last layer based on the batch's mean and standard deviation
    conv = BatchNormalization()(conv, training=False)

    # In case of overfitting, dropout will regularize the loss and gradient computation to shrink the influence of weights on output
    if dropout_prob > 0:
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)

    # Pooling reduces the size of the image while keeping the number of channels same
    # Pooling has been kept as optional as the last encoder layer does not use pooling (hence, makes the encoder block flexible to use)
    # Below, Max pooling considers the maximum of the input slice for output computation and uses stride of 2 to traverse across input image
    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv)
    else:
        next_layer = conv

    # skip connection (without max pooling) will be input to the decoder layer to prevent information loss during transpose convolutions
    skip_connection = conv

    return next_layer, skip_connection

'Unet - Decoder block'

def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=32):
    """
    Decoder Block first uses transpose convolution to upscale the image to a bigger size and then,
    merges the result with skip layer results from encoder block
    Adding 2 convolutions with 'same' padding helps further increase the depth of the network for better predictions
    The function returns the decoded layer output
    """
    # Start with a transpose convolution layer to first increase the size of the image
    up = Conv2DTranspose(
                 n_filters,
                 (3,3),    # Kernel size
                 strides=(2,2),
                 padding='same')(prev_layer_input)

    # Merge the skip connection from previous block to prevent information loss
    merge = concatenate([up, skip_layer_input], axis=3)

    # Add 2 Conv Layers with relu activation and HeNormal initialization for further processing
    # The parameters for the function are similar to encoder
    conv = Conv2D(n_filters,
                 3,     # Kernel size
                 activation='relu',
                 padding='same',
                 kernel_initializer='HeNormal')(merge)
    conv = Conv2D(n_filters,
                 3,   # Kernel size
                 activation='relu',
                 padding='same',
                 kernel_initializer='HeNormal')(conv)
    return conv

'Compile U-net Blocks'

def UNetCompiled(input_size=(100,100,n_g + 2), n_filters=32, n_classes=3):
   """
   Combine both encoder and decoder blocks according to the U-Net research paper
   Return the model as output
   """
   # Input size represent the size of 1 image (the size used for pre-processing)
   inputs = Input(input_size)

   # Encoder includes multiple convolutional mini blocks with different maxpooling, dropout and filter parameters
   # Observe that the filters are increasing as we go deeper into the network which will increasse the # channels of the image
   cblock1 = EncoderMiniBlock(inputs, n_filters,dropout_prob=0, max_pooling=True)
   cblock2 = EncoderMiniBlock(cblock1[0],n_filters*2,dropout_prob=0, max_pooling=True)
   cblock3 = EncoderMiniBlock(cblock2[0], n_filters*4,dropout_prob=0, max_pooling=True)
   cblock4 = EncoderMiniBlock(cblock3[0], n_filters*8,dropout_prob=0.3, max_pooling=True)
   cblock5 = EncoderMiniBlock(cblock4[0], n_filters*16, dropout_prob=0.3, max_pooling=False)

   # Decoder includes multiple mini blocks with decreasing number of filters
   # Observe the skip connections from the encoder are given as input to the decoder
   # Recall the 2nd output of encoder block was skip connection, hence cblockn[1] is used
   ublock6 = DecoderMiniBlock(cblock5[0], cblock4[1],  n_filters * 8)
   ublock7 = DecoderMiniBlock(ublock6, cblock3[1],  n_filters * 4)
   ublock8 = DecoderMiniBlock(ublock7, cblock2[1],  n_filters * 2)
   ublock9 = DecoderMiniBlock(ublock8, cblock1[1],  n_filters)

   # Complete the model with 1 3x3 convolution layer (Same as the prev Conv Layers)
   # Followed by a 1x1 Conv layer to get the image to the desired size.
   # Observe the number of channels will be equal to number of output classes
   conv9 = Conv2D(n_filters,
                3,
                activation='relu',
                padding='same',
                kernel_initializer='he_normal')(ublock9)

   conv10 = Conv2D(n_classes, 1, padding='same')(conv9)

   # Define the model
   model = tf.keras.Model(inputs=inputs, outputs=conv10)

   return model

'Build U-net architeture'

# Call the helper function for defining the layers for the model, given the input image size
unet = UNetCompiled(input_size=(128,128,n_g + 2), n_filters=32, n_classes=1)
# Check the summary to better interpret how the output dimensions change in each layer
unet.summary()

'Run model'

# There are multiple optimizers, loss functions and metrics that can be used to compile multi-class segmentation models
# Ideally, try different options to get the best accuracy
unet.compile(optimizer=tf.keras.optimizers.Adam(),
             #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             loss='MeanSquaredError'
             #metrics=['accuracy']
             )

# Setup for checkpoints
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="EIT_model/checkpoints/{epoch:02d}.keras",
    save_weights_only=False,
    save_best_only=False,
    save_freq = (n_samples//settings["batch_size"])*settings['save_period']
    )

print("Save Freq",(n_samples//settings["batch_size"])*settings['save_period'])

# Run the model in a mini-batch fashion and compute the progress for each epoch
results = unet.fit(dataset,
                   batch_size = settings["batch_size"],
                   steps_per_epoch = settings["steps_per_epoch"],
                   epochs = settings["epochs"],
                   validation_data = dataset_val,
                   verbose = 1,
                   callbacks = [checkpoint])


#-----------------------------------------------------------
# Retrieve a list of results on training and test data
# sets for each training epoch
#-----------------------------------------------------------

import matplotlib.pyplot as plt

#acc      = history.history['MeanAbsolutePercentageError' ]
#val_acc  = history.history[ 'val_accuracy' ]
loss     = results.history[    'loss' ]
val_loss = results.history['val_loss' ]

epochs   = range(len(loss)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
#plt.plot(epochs, acc, 'bo', label='Training accuracy')
#plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
#plt.title ('Training and validation accuracy')
#plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.figure(figsize=(10, 10))
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title ('Training and validation loss'   )
plt.legend()
plt.savefig("training_graph.png")

'Make Prediction and save the model'

import os, shutil
import numpy as np

# 'Save Model'
# if os.path.isdir("EIT_model/"):    # Remove directory if it exists
#     shutil.rmtree("EIT_model/")
# # Create directories
# dir = 'EIT_model'
# os.makedirs(dir, exist_ok = True)
# # Save
# # model.save('EIT_model/unet.keras')
unet.save('EIT_model/unet.keras')

'Save validation set'
# os.makedirs('Validation', exist_ok = True)
# np.save('Validation/input_val', input_val)
# np.save('Validation/label_val', label_val)

# model = tf.keras.models.load_model('unet.keras')

'Plot prediction'

# example = 0

# plt.figure(figsize=(10, 10))
# plt.subplot(4,4,1)
# plt.imshow(classes[example], interpolation='none')
# plt.subplot(4,4,2)
# plt.imshow(label_val[example], interpolation='none')
# plt.show()
