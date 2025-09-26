import os
import sys
import numpy as np
import scipy as sp
import json

def main(settings):

    SOL_DIRPATH = settings["dbar_input_datapath"]
    MAT_DIRPATH = settings["dbar_mat_datapath"]
    IMG_DIRPATH = settings["dbar_img_datapath"]

    dbar_file_list = [file for file in os.listdir(MAT_DIRPATH) if not file.endswith(".json")]

    with open(os.path.join(IMG_DIRPATH,"data_info.json"),"w") as f:
        f.write(json.dumps(settings))

    for file in dbar_file_list:
        
        sol_img = np.load(os.path.join(SOL_DIRPATH,file.replace("_dbar.mat","_img.npy")))

        dbar_img = sp.io.loadmat(os.path.join(MAT_DIRPATH,file))

        entry = np.array([dbar_img['dbar_img'],sol_img])

        np.save(os.path.join(IMG_DIRPATH,file.replace("_dbar.mat",".npy")),entry)

    print("Data converted succesfully")

if __name__=="__main__":
  if len(sys.argv)<2:
    raise Exception("Not enough arguments. \n **Usage:** python3 dbar_data_prep.py path/to/exp_settings.json")

  SETTINGS_JSON = sys.argv[1]
  if SETTINGS_JSON.endswith('.json') and os.path.isfile(SETTINGS_JSON):
    with open(SETTINGS_JSON) as f:
      SETTINGS_JSON = f.read()

      
  
  settings = json.loads(SETTINGS_JSON)


  main(settings)