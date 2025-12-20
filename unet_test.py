import eitx
import dolfinx
import pyvista
import os
import json 
import sys
from eit_image import EIT_Image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
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
    FILEPATH = ''
    DATAMAT_PATH = "fin_data/datamat/"
    SETTINGS_PATH = os.path.join(RESULTS_PATH,"data_info.json")
    with open(SETTINGS_PATH) as f:
        settings = json.loads(f.read())

    currents = settings['currents']
    MODELPATH = RESULTS_PATH

    DATAMAT_PATH = "fin_data/datamat/"
    test_path = "cem_test_samples/"

    "Forward problem in background"
    #Load data of background
    mat = sp.io.loadmat(DATAMAT_PATH+"datamat_1_0")
    Uel=mat.get("Uel").T
    CP=mat.get("CurrentPattern").T

    if not os.path.isdir(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)

    #Selecting Potentials
    Uel_b=Uel[:16][currents] #Matrix of measuarements
    print(Uel_b.shape)

    #Selecting Potentials
    list_U0_m=np.zeros_like(Uel_b)

    #Convert type of data
    for index, potential in enumerate(Uel_b):
        list_U0_m[index]=eitx.ConvertingData(potential, method="KIT4")

    #Current
    I_all=CP[:16][currents]/np.sqrt(2)
    l, L=np.shape(I_all) #Number of experiments = 15, Number of Electrodes = 16

    "Basic Definitions"
    radius=1               #Circle radius
    per_cober=0.454728409  #Percentage of area covered by electrodes
    rotate=0               #Electrodes Rotation
    z=np.ones(L)*0.07858  

    'Return object with angular position of each electrode'
    ele_pos = eitx.Electrodes(L, per_cober, rotate)

    'Mesh'
    # mesh_inverse=MyMesh(radius, refine_n, n_in, n_out, ele_pos)
    mesh_object = eitx.MeshClass(ele_pos,0.4,0.6)
    mesh = mesh_object.mesh

    ## Direct problem
    dir_problem = eitx.DirectProblem(mesh_object,z)
    V0 = dir_problem.V0   # Discontinuous Garlekin space function
    V = dir_problem.V     # Continuous Garlekin space function

    # l=L-1                                             #Number of experiments


    "Define sigma as constant = Background"
    gamma0 = dolfinx.fem.Function(V0) #Define the function with basis DG
    ivhigh, ivlow, bg = 10, 0.1, 1.0
    gamma0.x.array[:] = bg

    import tensorflow as tf

    print("Current index", I_all)

    #Solving Forward Problem

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
        list_u1, list_U1_m = dir_problem.solve_problem_current(I_all, gamma)

        "Difference of Resulting Potentials"
        differ = np.array(list_U1_m) - np.array(list_U0_m)

        "Solve Forward Problem with Background and Difference of Potentials as Currents"
        list_ur_dif, list_U_dif = dir_problem.solve_problem_current(differ, gamma0)

        
        T = np.zeros((l + 2,N,N))
        for k in range(l):
            T[k] = eit_image.genPotentialImg(list_ur_dif[k])

        T[l] = mesh_x
        T[l+1] = mesh_y

        T1.append(np.transpose(T))

    np.save(os.path.join(test_path,'model_input'),T1)
    np.save(os.path.join(test_path,'gamma_img'),gammaimg_list)
    input_val = tf.convert_to_tensor(T1)

    model = tf.keras.models.load_model(os.path.join(MODELPATH,'unet.keras'))

    classes = model.predict(input_val)
    np.save(os.path.join(RESULTS_PATH,'samples_test_result'), [pred[:,:,0].T for pred in classes])
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

    plt.savefig(os.path.join(RESULTS_PATH,'samples_test_result.png'))

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