import os
import json 
import sys
from eit_image import EIT_Image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import logging
import traceback
from unet import UNetCompiled

logging.basicConfig(
    filename='experiments.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main(TEST_PATH, RESULTS_PATH):
    SETTINGS_PATH = os.path.join(TEST_PATH, "data_info.json")

    with open(SETTINGS_PATH) as f:
        settings = json.loads(f.read())

    GAMMA_PATH = settings["samples_dir"]
    PRED_PATH = os.path.join(RESULTS_PATH, "PRED")
    
    if not os.path.isdir(PRED_PATH):
        os.mkdir(PRED_PATH)

    MODELPATH = RESULTS_PATH

    dsm_dir = [file for file in os.listdir(TEST_PATH) if file.endswith("dsm_fnn.npy")]

    model = tf.keras.models.load_model(os.path.join(MODELPATH,'fnn.keras'))

    T1 = []
    gammaimg_list = []
    pred_list = []    

    for sample in dsm_dir:
        #Load experimental data
        print(sample)
        
        sample_path = os.path.join(TEST_PATH,sample)
        sample_mat = np.load(sample_path)

        input_mat = sample_mat[:,:-1]
        true_mat = sample_mat[:,-1]

        input_val = tf.convert_to_tensor(input_mat)
        pred = model.predict(input_val)

        print(pred.shape)
        print(true_mat.shape)
        # gammaimg_list.append(T[-1])
        pred_list.append(
            np.column_stack([
                pred.flatten(),
                true_mat
            ])
        )

    sample_pred_path = os.path.join(PRED_PATH, f"TEST_{settings['n_currents']}_DELTA_{100*settings['noise_level']}")
    np.save(sample_pred_path, pred_list)
    # input_val = tf.convert_to_tensor(T1)

    # classes = model.predict(input_val)
    

if __name__=='__main__':
    if len(sys.argv)<3:
        raise Exception("Not enough arguments. Use `python3 unet_testset.py TEST_PATH RESULTS_PATH`")

    TEST_PATH = sys.argv[1]
    RESULTS_PATH = sys.argv[2]

    try:    
        main(TEST_PATH,RESULTS_PATH)
    except Exception as e:    
        msg = f"FNN test-set routine failed calling paths {sys.argv[1]} {sys.argv[2]} \n"
        print(msg)
        logging.error(msg)
        # print(e)
        print(traceback.format_exc())
        logging.error(traceback.format_exc())