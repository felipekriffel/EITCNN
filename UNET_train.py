import math
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import sys
from unet import *

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


def create_sample_dataset(record_file,batch_size,epochs):
    raw_image_dataset = tf.data.TFRecordDataset(record_file)
    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    parsed_image_dataset = parsed_image_dataset.map(_parse_image_tensor)
    parsed_image_dataset = parsed_image_dataset.repeat(epochs).batch(batch_size)
    return parsed_image_dataset.prefetch(1)

def main(SETTINGS_JSON):
    # with open(SETTINGS_JSON) as f:
    #     settings = json.loads(f.read())
    
    settings = json.loads(SETTINGS_JSON)

    SAVEPATH =  settings['savepath']
    if not os.path.isdir(SAVEPATH):
        os.mkdir(SAVEPATH)

    with open(settings['datapath']+"/data_info.json") as f:
        data_settings =  json.loads(f.read())

    with open(SAVEPATH+'unet_train_settings.json','w') as f:
        f.write(json.dumps(settings))

    # T1 = np.load('EIT_Data_for_CNN.npy')
    with open(settings['tfrecordpath']+"/data_info.json") as f:
        data_info = json.loads(f.read())

    with open(SAVEPATH+"data_info.json","w") as f:
        f.write(json.dumps(data_settings))

    n_g = data_settings['n_g']
    n_samples = data_info['n_samples']
    n_train = data_info['n_train']
    n_val = data_info['n_val']
    if settings['steps_per_epoch'] == "full":
        steps_per_epoch = n_train // settings['batch_size']
    else:
        steps_per_epoch = settings['steps_per_epoch']
    
    print("Number of currents:",n_g)
    print("Number of samples:", n_samples)
    print('Number of samples for training: ' + str(n_train))

    tfrecord_dirpath = settings['tfrecordpath']
    dataset = create_sample_dataset(tfrecord_dirpath+"/train.tfrecords", batch_size = settings['batch_size'],epochs=settings['epochs'])
    dataset_val = create_sample_dataset(tfrecord_dirpath+"/validation.tfrecords", batch_size = settings['batch_size'],epochs=settings['epochs'])

    'Unet - Encoder block'
    'Build U-net architeture'
    # Call the helper function for defining the layers for the model, given the input image size
    unet_model = UNetCompiled(input_size=(128,128,n_g + 2), n_filters=32, n_classes=1,dropout=settings['dropout_prob'])
    # Check the summary to better interpret how the output dimensions change in each layer
    unet_model.summary()

    'Run model'

    # There are multiple optimizers, loss functions and metrics that can be used to compile multi-class segmentation models
    # Ideally, try different options to get the best accuracy
    unet_model.compile(optimizer=tf.keras.optimizers.Adam(),
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
    results = unet_model.fit(dataset,
                    batch_size = settings["batch_size"],
                    steps_per_epoch = steps_per_epoch,
                    epochs = settings["epochs"],
                    validation_data = dataset_val,
                    verbose = 1,
                    callbacks = [checkpoint])

    #-----------------------------------------------------------
    # Retrieve a list of results on training and test data
    # sets for each training epoch
    #-----------------------------------------------------------
    #acc      = history.history['MeanAbsolutePercentageError' ]
    #val_acc  = history.history[ 'val_accuracy' ]
    loss     = results.history[    'loss' ]
    val_loss = results.history['val_loss' ]

    epochs   = range(len(loss)) # Get number of epochs

    unet_model.save('EIT_model/unet.keras')
    unet_model.save(SAVEPATH+'unet.keras')

    #------------------------------------------------
    # Plot training and validation loss per epoch
    #------------------------------------------------
    plt.figure(figsize=(10, 10))
    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title ('Training and validation loss'   )
    plt.legend()
    plt.savefig(SAVEPATH+"training_graph.png")
    plt.savefig("training_graph.png")

if __name__=="__main__":
#   SETTINGS_JSON = 'unet_train_settings.json'
    SETTINGS_JSON = sys.argv[1]
    if SETTINGS_JSON.endswith('.json') and os.path.isfile(SETTINGS_JSON):
        with open(SETTINGS_JSON) as f:
            SETTINGS_JSON = f.read()

    main(SETTINGS_JSON)