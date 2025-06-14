import os
import numpy as np
import scipy as sp

DBAR_DIRPATH = "/mnt/c/Users/Felipe/Documents/dbar_results/first_exp/dbar_img/"
SOL_DIRPATH = 'dbar_data/solutions/'
SAVEPATH = "/mnt/c/Users/Felipe/Documents/dbar_results/first_exp/unet_entries/"

dbar_file_list = os.listdir(DBAR_DIRPATH)

for file in dbar_file_list:
    print(DBAR_DIRPATH+file)
    dbar_mat = sp.io.loadmat(DBAR_DIRPATH+file)
    sol_mat = np.load(SOL_DIRPATH+file.replace("_dbar.mat","_img.npy"))
    entry = np.array([dbar_mat['dbar_img'],sol_mat])

    np.save(SAVEPATH+file.replace("_dbar.mat",".npy"),entry)