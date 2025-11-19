#!/bin/bash

python3 DSM_cnn_data_gen.py settings/dissert_cont_cnn_exp_1.json

python3 DSM_phi_datagen.py settings/dissert_cont_cnn_exp_1.json

python3 create_cnn_tfrecord.py settings/dissert_cont_unet_exp_1.json

python3 UNET_train.py settings/dissert_cont_unet_exp_1.json

python3 unet_test.py /mnt/c/Users/Felipe/Documents/DISSERTACAO/RESULTADOS/EXP_CONT_1_1