import os
import sys
import json
import subprocess
import logging
import approxinv

# This code perform all the data gen, train and test procedure
# create a json file for each experiment, inform

logging.basicConfig(
    filename='experiments.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# by default, catches the name of folder in experiments_settings/ dir informed at terminal
# if not informed, catches 'experiment.json' file in experiments_settings
 
def main(experiments_list):
    # read experiments
    for experiment_name in experiments_list:
        try:
            with open(experiments_dir_path+"/"+experiment_name) as f:
                experiment = json.loads(f.read())

            print(f"\n-----\nStarting experiment {experiment_name}\n-----\n")
            
            if "data_gen" in experiment:

                print("\nStarting data generation \n")
                subprocess.run(["python3","approxinv_samples_datagen.py",json.dumps(experiment['data_gen'])],check=True)

            print("\nCreating TFRecord\n")
            subprocess.run(["python3", "create_tfrecord.py",json.dumps(experiment['unet_train'])],check=True)

            print("\nTraining UNET\n")
            subprocess.run(["python3", "approxinv_UNET_train.py", json.dumps(experiment['unet_train'])],check=True)

            print("\nTesting model samples")
            subprocess.run(['python3',"approxinv_CEM_CNN_test.py",experiment['unet_train']['savepath']],check=True)

            with open(experiment['unet_train']['save_path']+"experiment_settings.json","w"):
                f.write(json.dumps(experiment))

            print(f"\n-----Finished experiment {experiment_name} sucessfully\n-----")

        except Exception as e:
            logging.error(f"Experiment {experiment_name} resulted in error")
            logging.error(e)
            print(f"Experiment {experiment_name} resulted in error:")
            print(e)

if __name__=='__main__':
    if len(sys.argv)>1:
        experiment_battery_dir = sys.argv[1]
        experiments_dir_path = "experiments_settings/"+experiment_battery_dir
        experiments_list = os.listdir(experiments_dir_path)

    else:
        experiments_dir_path = 'experiments_settings'
        experiments_list = ['experiment.json']

    print("Experiment dir:",experiments_dir_path)
    print("Experiments list:",experiments_list)

    main(experiments_list)
