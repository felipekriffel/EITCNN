#!/bin/bash

# python3 CEM_data_gen.py settings/datagen_exp2.json

# python3 DSM_cnn_data_gen.py settings/cnn_exp_2.json

# python3 create_cnn_tfrecord.py settings/unet_train_exp2.json

python3 UNET_train.py settings/unet_train_exp2.json

python3 DSM_test_kit4.py /home/feliperiffel/Documentos/Mestrado/Dissertação/Experimentos/Resultados/EXP_2