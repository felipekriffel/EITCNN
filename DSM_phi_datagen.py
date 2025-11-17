import eit_cont
import scipy as sp
import numpy as np
import dolfinx
from petsc4py import PETSc
import json
import os
import sys
from eit_image import EIT_Image
import logging
import traceback

logging.basicConfig(
    filename='experiments.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main(SETTINGS_JSON):
    settings = json.loads(SETTINGS_JSON)
    
    if not os.path.isdir(settings['dsm_datapath']):
        os.mkdir(settings['dsm_datapath'])

    with open(settings['dsm_datapath']+"/data_info.json","w") as f:
        f.write(json.dumps(settings))

    "Importing modules"
    # Set the logging level to suppress most logs
    logging.getLogger('UFL_LEGACY').setLevel(logging.WARNING)
    logging.getLogger('dolfin').setLevel(logging.WARNING)
    logging.getLogger('PETSc').setLevel(logging.WARNING)
    logging.getLogger('FFC').setLevel(logging.WARNING)

    "Forward problem in background"
    L=20

    "Basic Definitions"
    radius=1               #Circle radius
    per_cober=0.5  #Percentage of area covered by electrodes
    rotate=0               #Electrodes Rotation
    
    'Return object with angular position of each electrode'
    ele_pos = eit_cont.Electrodes(L, per_cober, rotate)

    'Mesh'
    # mesh_inverse=MyMesh(radius, refine_n, n_in, n_out, ele_pos)
    mesh_object = eit_cont.MeshClass(ele_pos,0.3,0.4)

    ## Direct problem
    dir_problem = eit_cont.DirectProblem(mesh_object)
    V0 = dir_problem.V0   # Discontinuous Garlekin space function
    V = dir_problem.V     # Continuous Garlekin space function

    #"Define gamma as constant = Background"
    bg = settings['bg']
    ivhigh,ivlow = settings['ivhigh'], settings['ivlow']
    gamma0 = dolfinx.fem.Function(V0)
    gamma0.x.array[:] = bg

    current_index = settings['currents']
    max_current_index = max(current_index)+1

    print("Current index", current_index)
    print("n currents", max_current_index)

    current_list = dir_problem.get_current_list(max_current_index)
    current_list = [current_list[i] for i in current_index]
    n_currents = len(current_list)

    #Solving Forward Problem
    list_u0 = dir_problem.solve_problem_current(current_list, gamma0)

    'Retangular Mesh'
    N = settings["N"]                     # grid with N*N points (works well with 0 < N < 400)
    h = 2*radius/(N-1)                    # step size
    x = [radius - i*h for i in range(N)]  # x grid points
    y = [-radius + i*h for i in range(N)] # y grid points

    # MESH x and y
    mesh_x = np.zeros((N,N))              # x-Data (input of CNN)
    mesh_y = np.zeros((N,N))              # y-Data (input of CNN)
    for i in range(N):
        for j in range(N):
            mesh_x[i][j] = x[i]
            mesh_y[i][j] = y[j]

    eit_image = EIT_Image(dir_problem.mesh,mesh_x,mesh_y)

    gamma = dolfinx.fem.Function(V0)      # Empty function
    T1 = []                               # To save data

    samples_dir = settings['samples_dir']
    samples_names = [file for file in os.listdir(samples_dir) if file.endswith(".npy")]
    
    nan_samples_path = os.path.join(samples_dir,"nan_samples.json")
    if os.path.exists(nan_samples_path):
        with open(nan_samples_path,'r') as f:
            saved_nan = json.loads(f.read())
        nan_samples = saved_nan
    else:
        nan_samples = []

    # Loop for generating data
    noise_level = settings["noise_level"] # % of artificial noise in data
    for sample in samples_names:
        sample_path = os.path.join(settings['dsm_datapath'],sample.replace(".npy","_dsm_phi.npy"))
        if os.path.exists(sample_path) and sample_path not in nan_samples:
            print(f"{sample} dsm data already computed, skipping")
            continue
        else:
            print("Computing", sample)
            
        gamma.x.array[:]= np.load(os.path.join(samples_dir, sample))

        "Define data in a homogeneus grid for training"
        gamma_img = eit_image.genGammaImg(gamma,bg,ivhigh,ivlow,type='bin')

        "Solve Forward Problem with Background + Inclusion"
        list_u1 = dir_problem.solve_problem_current(current_list, gamma)

        "Difference of Resulting Potentials"
        differ_list = [dolfinx.fem.Function(V) for i in range(n_currents)]

        differ_noisy = dolfinx.fem.Function(V)
        for k in range(n_currents):

            differ_array = list_u1[k].x.array - list_u0[k].x.array
            differ_noisy.x.array[:] = differ_array

            noise = np.random.uniform(-1, 1, size=(len(differ_array)))
            noise = noise / np.linalg.norm(noise)
        
            differ_list[k].x.array[:] = differ_array + noise_level*noise*eit_cont.bdrNorm(differ_noisy)

        "Solve Forward Problem with Background and Difference of Potentials as Currents"
        list_ur_dif = dir_problem.solve_problem_current(differ_list, gamma0)

        "Saves data on tensor"
        save_mat = np.array([
            u.x.array for u in list_ur_dif
        ])

        if np.isnan(save_mat).any():
            print(f"NAN at sample {sample}, skipping saving")
            nan_samples.append(sample)
        else:
            np.save(sample_path,save_mat)
        
    # np.save('EIT_Data_for_CNN', T1)
    print(f'Data saved at {settings["dsm_datapath"]}.')

    if len(nan_samples)>0:
        with open(os.path.join(settings['dsm_datapath'], "nan_samples.json"),'w') as f:
            f.write(json.dumps(nan_samples))

if __name__=="__main__":
    SETTINGS_JSON = sys.argv[1]
    if SETTINGS_JSON.endswith('.json') and os.path.isfile(SETTINGS_JSON):
        with open(SETTINGS_JSON) as f:
            SETTINGS_JSON = f.read()

    try:
        main(SETTINGS_JSON)
    except Exception as e:
        msg = f"*** \n DSM phi calc routine failed calling arg {sys.argv[1]} \n***"
        print(msg)
        logging.error(msg)
        # print(e)
        print(traceback.format_exc())
        logging.error(traceback.format_exc())