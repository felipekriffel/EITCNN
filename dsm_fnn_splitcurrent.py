import numpy as np
import os
import logging
import json
import sys
from matplotlib import pyplot as plt

logging.basicConfig(
    filename='experiments.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main(SETTINGS_JSON):

    settings = json.loads(SETTINGS_JSON)

    if not os.path.isdir(settings['dsm_datapath']):
        os.mkdir(settings['dsm_datapath'])

    with open(os.path.join(settings['dsm_datapath'],"data_info.json"),"w") as f:
        f.write(json.dumps(settings))

    with open(os.path.join(settings["dsm_old_datapath"], "data_info.json")) as f:   
        old_info = json.loads(f.read())

    len_old_currents = len(old_info['currents'])
    len_old_tensor = 2*len_old_currents + 3

    col_index = [0,1,len_old_tensor-1]
    
    for k in settings['currents']:
        col_index.append(2+2*k)
        col_index.append(2+2*k+1)

    col_index.sort()
    print(col_index)

    dsm_dir_list = [sample for sample in os.listdir(settings["dsm_old_datapath"]) if sample.endswith("dsm_fnn.npy")]
    
    for sample in dsm_dir_list:
        sample_path = os.path.join(settings["dsm_old_datapath"], sample)
        sample_mat = np.load(sample_path)

        ## salva tensor na pasta nova
        new_sample_name = os.path.join(settings['dsm_datapath'], sample)        
        
        print("Saving", new_sample_name)
        
        np.save(new_sample_name, sample_mat[:,col_index])

    print(f'Data saved at {settings["dsm_datapath"]}.')


if __name__=="__main__":
    SETTINGS_JSON = sys.argv[1]
    if SETTINGS_JSON.endswith('.json') and os.path.isfile(SETTINGS_JSON):
        with open(SETTINGS_JSON) as f:
            SETTINGS_JSON = f.read()

    try:
        main(SETTINGS_JSON)
    except Exception as e:
        logging.error(f"current spliting gen failed calling {sys.argv[1]} config file")
        logging.error(e)
        print(f"current spliting gen failed calling {sys.argv[1]} config file")
        print(e)