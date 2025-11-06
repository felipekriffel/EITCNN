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

  def point_example(sample_array):
    is_inclusion = sample_array[-1]
    sample_array = sample_array[:-1]
    sample_shape = sample_array.shape
    
    n_g = (sample_shape[0] - 2)//2

    sample_array_raw = sample_array.tobytes()
    # is_inclusion_raw = is_inclusion.tobytes()

    feature = {
        'currents': _int64_feature(n_g),
        'sample': _bytes_feature(sample_array_raw),
        'is_inclusion': _float_feature(is_inclusion),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

  def create_tfrecord(record_file:str,paths:list,batch_size:int=32)->None:
    """
    Create tfrecord based on a list of samples with fnn data. 
    If batch_size>1, loads batchs of files, concatanate in a single matrix
    and shuffle all the rows, for randomness.
    
    :param record_file: path to created .tfrecord file
    :param paths: list with paths to each file having the samples

    """
    num_iter = len(paths)//batch_size
    
    with tf.io.TFRecordWriter(record_file) as writer:
      for j in range(num_iter):
        #load filen
        batch_files = paths[j*batch_size:(j+1)*batch_size]        
        
        sample_mat = np.concatenate([np.load(file) for file in batch_files])
        np.random.shuffle(sample_mat)

        print("Loading samples", batch_files)
        print(sample_mat.shape)
        
        if np.isnan(sample_mat).any():
          print(f"----- \n WARNING: NAN found at batch {batch_files}, skipping computation\n -----")
          continue

        for sample_array in sample_mat:
          tf_example = point_example(sample_array)
          writer.write(tf_example.SerializeToString())
      writer.close()

  feature_paths = [os.path.join(DATAPATH,x) for x in os.listdir(DATAPATH) if x.endswith("fnn.npy")]
  sample0 = np.load(feature_paths[0])

  per = settings['split_percentage']

  n_samples = len(feature_paths)
  n_train = math.floor(n_samples*per)    # number samples for training
  print('Number of samples for training: ', n_train*sample0.shape[0])
  print('Samples per file: ', sample0.shape[0])
  n_val = n_samples - n_train        # number of samples for validation
  print('Number of samples for validation: ', n_val*sample0.shape[0])

  # permute the lines
  perm = np.random.permutation(n_samples)

  # print(perm)
  paths_division = [feature_paths[i] for i in perm[:n_train]]
  paths_division2 = [feature_paths[i] for i in perm[n_train:]]

  print("Creating training tfrecords")
  create_tfrecord(os.path.join(SAVEPATH,"train.tfrecords"),paths_division)

  print("Creating validation tfrecords")
  create_tfrecord(os.path.join(SAVEPATH,"validation.tfrecords"),paths_division2)

  data_info = {
    "n_samples": n_samples,
    "n_train": n_train*sample0.shape[0],
    "n_val": n_val*sample0.shape[0],
    "n_cols": sample0.shape[1]-1
  }

  with open(SAVEPATH+"/data_info.json",'w') as f:
    f.write(json.dumps(data_info))

if __name__=='__main__':
  # SETTINGS_JSON = "unet_train_settings.json"
  
  SETTINGS_JSON = sys.argv[1]
  if SETTINGS_JSON.endswith('.json') and os.path.isfile(SETTINGS_JSON):
    with open(SETTINGS_JSON) as f:
      SETTINGS_JSON = f.read()

  main(SETTINGS_JSON)