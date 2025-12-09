import eit_cont
import dolfinx
import pyvista
import os
import json 
import sys
from eit_image import EIT_Image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf
import logging
import traceback
from unet import UNetCompiled

logging.basicConfig(
    filename='experiments.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main(RESULTS_PATH):
    SETTINGS_PATH = os.path.join(RESULTS_PATH, "data_info.json")

    with open(SETTINGS_PATH) as f:
        settings = json.loads(f.read())

    MODELPATH = RESULTS_PATH

    if not os.path.isdir(RESULTS_PATH):
       os.mkdir(RESULTS_PATH)

    test_path = "cont_test_samples/"

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
    mesh = mesh_object.mesh

    ## Direct problem
    dir_problem = eit_cont.DirectProblem(mesh_object)
    V0 = dir_problem.V0   # Discontinuous Garlekin space function
    V = dir_problem.V     # Continuous Garlekin space function

    #"Define gamma as constant = Background"
    bg = settings['bg']
    ivhigh,ivlow = settings['ivhigh'], settings['ivlow']
    noise_level = settings['noise_level']
    gamma0 = dolfinx.fem.Function(V0)
    gamma0.x.array[:] = bg

    current_index = settings['currents']
    max_current_index = max(current_index)+1

    current_list = dir_problem.get_current_list(max_current_index)
    current_list = [current_list[i] for i in current_index]
    n_currents = len(current_list) 

    print("Current index", current_index)
    print("n currents", max_current_index)

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

    eit_img = EIT_Image(dir_problem.mesh,mesh_x,mesh_y)

    cond_dir = [file for file in os.listdir(test_path) if file.startswith("sample_") and file.endswith(".npy") and not file.endswith("_img.npy")]
    cond_dir.sort()

    T1 = []

    gammaimg_list = []

    for sample in cond_dir:
        #Load experimental data
        gamma_array = np.load(os.path.join(test_path,sample))
        gamma.x.array[:] = gamma_array
        # gammaimg_list.append(eit_image.genGammaImg(gamma,bg,ivhigh,ivlow))
        gammaimg_list.append(eit_image.genGammaImg(gamma,bg,ivhigh,ivlow,type='seg'))

        # gammaimg_list.append(np.asarray(Image.open(os.path.join(test_path,sample.replace('.npy','.png')))))

        list_u1 = dir_problem.solve_problem_current(current_list, gamma)

        "Difference of Resulting Potentials"
        differ_list = [dolfinx.fem.Function(V) for i in range(n_currents)]
        differ_noisy = dolfinx.fem.Function(V)

        for k in range(n_currents):
            
            differ_array = list_u1[k].x.array - list_u0[k].x.array
            differ_noisy.x.array[:] = differ_array

            noise = np.random.uniform(-1, 1, size=(len(differ_array)))
            noise = noise / np.linalg.norm(noise)
            differ_noisy.x.array[:] = differ_array + noise_level*noise*eit_cont.bdrNorm(differ_noisy)

            differ_list.append(differ_noisy)
            differ_list[k].x.array[:] = differ_array

            
        "Solve Forward Problem with Background and Difference of Potentials as Currents"
        list_ur_dif = dir_problem.solve_problem_current(differ_list, gamma0)

        "Define data in a homogeneus grid for test"
        T = np.zeros((n_currents + 2,N,N))
        for k in range(n_currents):
            T[k] = eit_image.genPotentialImg(list_ur_dif[k])

        T[n_currents] = mesh_x
        T[n_currents+1] = mesh_y

        T1.append(np.transpose(T))

    np.save(os.path.join(test_path,'model_input'),T1)
    np.save(os.path.join(test_path,'gamma_img'),gammaimg_list)
    input_val = tf.convert_to_tensor(T1)

    model = tf.keras.models.load_model(os.path.join(MODELPATH,'unet.keras'))

    classes = model.predict(input_val)
    np.save(os.path.join(RESULTS_PATH,'test_result'), [pred[:,:,0].T for pred in classes])
    print(classes.shape)
    
    'Plot'
    fig, ax = plt.subplots(2,classes.shape[0],figsize=(40,10))
    img_array = []
    for k in range(classes.shape[0]):
        img_array.append(ax[0][k].imshow(classes[k,:,:,0].T))
        ax[0][k].set_axis_off()
        img_array.append(ax[1][k].imshow(gammaimg_list[k],vmin=-1.0,vmax=1.0))
        ax[1][k].set_axis_off()

    fig.colorbar(img_array[0],ax=ax[0,:],orientation='vertical')
    fig.colorbar(img_array[-1],ax=ax[1,:],orientation='vertical')

    plt.savefig(os.path.join(RESULTS_PATH,'test_result.png'))

if __name__=='__main__':
    RESULTS_PATH = sys.argv[1]

    try:    
        main(RESULTS_PATH)
    except Exception as e:    
        msg = f"Unet test failed calling {sys.argv[1]} config file"
        print(msg)
        # print(e)
        print(traceback.format_exc())
        logging.error(traceback.format_exc())