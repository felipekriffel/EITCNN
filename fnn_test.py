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

logging.basicConfig(
    filename='experiments.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main(RESULTS_PATH):
    SETTINGS_PATH = os.path.join(RESULTS_PATH, "data_info.json")

    with open(SETTINGS_PATH) as f:
        settings = json.loads(f.read())

    currents = settings['currents']
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
    gamma0 = dolfinx.fem.Function(V0)
    gamma0.x.array[:] = bg

    delx_phi = dolfinx.fem.Function(V0)
    dely_phi = dolfinx.fem.Function(V0)

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
    N = 128                 # grid with N*N points (works well with 0 < N < 400)
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
    input_list = []                               # To save data

    cond_dir = [file for file in os.listdir(test_path) if file.endswith(".npy")]

    gammaimg_list = []

    for sample in cond_dir:
        #Load experimental data
        gamma_array = np.load(os.path.join(test_path,sample))
        gamma.x.array[:] = gamma_array
        # gammaimg_list.append(eit_image.genGammaImg(gamma,bg,ivhigh,ivlow))
        gammaimg_list.append(eit_image.genGammaImg(gamma,bg,ivhigh,ivlow,type='seg'))


        list_u1 = dir_problem.solve_problem_current(current_list, gamma)

        "Difference of Resulting Potentials"
        differ_list = [dolfinx.fem.Function(V) for i in range(n_currents)]
        differ_noisy = dolfinx.fem.Function(V)

        for k in range(n_currents):
            
            differ_array = list_u1[k].x.array - list_u0[k].x.array
            differ_noisy.x.array[:] = differ_array

            # noise = np.random.uniform(-1, 1, size=(len(differ_array)))
            # noise = noise / np.linalg.norm(noise)
            # differ_noisy.x.array[:] = differ_array + noise_level*noise*eit_cont.bdrNorm(differ_noisy)

            # differ_list.append(differ_noisy)
            differ_list[k].x.array[:] = differ_array

            
        "Solve Forward Problem with Background and Difference of Potentials as Currents"
        list_phi = dir_problem.solve_problem_current(differ_list, gamma0)

        list_delx_phi = []
        list_dely_phi = []


        for k in range(n_currents):
            
            gradphi_array = dir_problem.compute_gradient(list_phi[k])

            delx_phi.x.array[:] = gradphi_array[:,0]
            dely_phi.x.array[:] = gradphi_array[:,1]

            delx_img = eit_image.genPotentialImg(delx_phi)
            dely_img = eit_image.genPotentialImg(dely_phi)

            list_delx_phi.append(delx_img)
            list_dely_phi.append(dely_img)


        T = np.zeros((2*n_currents + 2,N,N))
        T[0] = mesh_x
        T[1] = mesh_y
        for k in range(n_currents):
            T[2+2*k] = list_delx_phi[k]
            T[2+2*k+1] = list_dely_phi[k]        

        vec_list = []
        for i in range(N):
            for j in range(N):
                if T[0,i,j]**2 + T[1,i,j]**2 < 1:    
                    vec_list.append(T[:,i,j])

        input_list.append(vec_list)

    model = tf.keras.models.load_model(os.path.join(MODELPATH,'fnn.keras'))
    
    'Plot'
    fig, ax = plt.subplots(2,len(input_list),figsize=(40,10))
    img_array = []
    for k in range(len(input_list)):
        print('testing')
        input_val = tf.convert_to_tensor(input_list[k])

        vec_predit = model.predict(input_val)
        print("shape predict", vec_predit.shape)

        pred_img = eit_image.get_fnn_matrix(vec_predit.flatten())

        img_array.append(ax[0][k].imshow(pred_img))
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
        msg = f"Fnn test failed calling {sys.argv[1]} config file"
        print(msg)
        # print(e)
        print(traceback.format_exc())
        logging.error(traceback.format_exc())