#!/bin/bash

python3 cont_data_gen.py settings/cont_datagen_1.json

python3 DSM_cnn_data_gen.py settings/cont_cnn_exp_1.json

python3 create_cnn_tfrecord.py settings/cont_unet_train_1.json

python3 UNET_train.py settings/cont_unet_train_1.json

python3 unet_test.py /home/feliperiffel/Documentos/Mestrado/Dissertação/Experimentos/Resultados/EXP_1_CONT