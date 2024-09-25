import numpy as np
import tensorflow as tf
import os
import sys
import math
import json

def main(SETTINGS_JSON):
  # with open(SETTINGS_PATH) as f: 
  #    settings = json.loads(f.read())
  settings = json.loads(SETTINGS_JSON)

  DATAPATH = settings['datapath']
  SAVEPATH = settings['tfrecordpath']

  def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
      value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

  def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  def image_example(sample_array):
    admitivity = np.transpose(sample_array[-1])
    sample_array = np.transpose(sample_array[:-1])
    sample_shape = sample_array.shape
    n_g = sample_shape[-1] - 2

    sample_array_raw = sample_array.tobytes()
    admitivity_raw = admitivity.tobytes()

    feature = {
        'height': _int64_feature(sample_shape[0]),
        'width': _int64_feature(sample_shape[1]),
        'depth': _int64_feature(sample_shape[2]),
        'currents': _int64_feature(n_g),
        'sample_raw': _bytes_feature(sample_array_raw),
        'admitivity_raw': _bytes_feature(admitivity_raw),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

  def create_tfrecord(record_file,paths):
    with tf.io.TFRecordWriter(record_file) as writer:
      for filename in paths:
        sample_array = np.load(filename)
        tf_example = image_example(sample_array)
        writer.write(tf_example.SerializeToString())
      writer.close()

  feature_paths = [DATAPATH+'/'+x for x in os.listdir(DATAPATH) if x!="data_info.json"]

  per = settings['split_percentage']

  n_samples = len(feature_paths)
  n_train = math.floor(n_samples*per)    # number samples for training
  print('Number of samples for training: ' + str(n_train))
  n_val = n_samples - n_train        # number of samples for validation

  # permute the lines
  perm = np.random.permutation(n_samples)

  # print(perm)
  paths_division = [feature_paths[i] for i in perm[:n_train]]
  paths_division2 = [feature_paths[i] for i in perm[n_train:]]

  create_tfrecord(SAVEPATH+"/train.tfrecords",paths_division)
  create_tfrecord(SAVEPATH+"/validation.tfrecords",paths_division2)

  data_info = {
    "n_samples": n_samples,
    "n_train": n_train,
    "n_val": n_val
  }

  with open(SAVEPATH+"/data_info.json",'w') as f:
    f.write(json.dumps(data_info))

if __name__=='main':
  # SETTINGS_JSON = "unet_train_settings.json"
  SETTINGS_JSON = sys.argv[1]
  if SETTINGS_JSON.endswith('.json') and os.path.isfile(SETTINGS_JSON):
    with open(SETTINGS_JSON) as f:
      SETTINGS_JSON = f.read()

  main(SETTINGS_JSON)