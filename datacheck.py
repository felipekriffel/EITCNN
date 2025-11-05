import os
import json
import numpy as np
import sys

data_dir = sys.argv[1]
data_list = [file for file in os.listdir(data_dir) if file.endswith('.npy')]

max_array = np.zeros(19)
min_array = np.zeros(19)

NAN_samples = []

for sample in data_list:
    print("Checking sample",sample)
    data = np.load(os.path.join(data_dir,sample))

    for i in range(max_array.shape[0]):
        max_array[i] = max(np.max(data[i]), max_array[i])
        min_array[i] = min(np.min(data[i]), min_array[i])
    if np.isnan(data).any():
        print("\n --- NAN FOUND --- \n")
        NAN_samples.append(sample)

print("Found NAN")
print(NAN_samples)

nan_samples_path = os.path.join(data_dir,"nan_samples.json")
if os.path.exists(nan_samples_path):
    with open(nan_samples_path,'r') as f:
        saved_nan_samples = json.loads(f.read())
    NAN_samples = saved_nan_samples+NAN_samples

if len(NAN_samples)>0:
    with open(nan_samples_path,'w') as f:
        f.write(json.dumps(NAN_samples))

print("Max and min")
print(max_array)
print(min_array)