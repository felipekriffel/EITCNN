import dolfinx
import pyvista
import eitx
import os
import json 
import sys
from eit_image import EIT_Image
import logging
import traceback
from scipy.ndimage import rotate
from tensorflow import keras

logging.basicConfig(
    filename='experiments.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main(RESULTS_PATH):
  FILEPATH = ''
  DATAMAT_PATH = "fin_data/datamat/"
  SETTINGS_PATH = RESULTS_PATH + "/data_info.json"
  with open(SETTINGS_PATH) as f:
      settings = json.loads(f.read())

  currents = settings['currents']
  MODELPATH = RESULTS_PATH

  'Load files'

  from PIL import Image

  import matplotlib.pyplot as plt
  import numpy as np
  import scipy

  #Load data of background
  mat = scipy.io.loadmat(DATAMAT_PATH+"datamat_1_0")
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
  list_U0=list_U0_m.flatten() #Matrix to vector

  #Current
  I_all=CP[:16][currents]/np.sqrt(2)
  l, L=np.shape(I_all) #Number of experiments = 15, Number of Electrodes = 16

  "Basic Definitions"
  radius=1       #Circle radius
  # L=16           #Number of Electrodes
  per_cober=0.5  #Percentage of area covered by electrodes
  rotation= 0      #Electrodes Rotation

  'Return object with angular position of each electrode'
  ele_pos = eitx.Electrodes(L, per_cober, rotation)
  refine_n = 8     #Refinement mesh
  n_in = 8         #Vertex on elec.
  n_out = 2        #Vertex on gaps (Sometimes it is important.)

  # CURRENT
  'Basic Definitions'
  # z_r=np.ones(L)*0.025E-3                         #Impedance of each electrode
  z_r=np.ones(L)*0.07858
  z = z_r

  'Mesh'
  # mesh_inverse=MyMesh(radius, refine_n, n_in, n_out, ele_pos)
  mesh_object = eitx.MeshClass(ele_pos,0.4,0.6)
  mesh = mesh_object.mesh

  ## Direct problem
  dir_problem = eitx.DirectProblem(mesh_object,z)
  V0 = dir_problem.V0   # Discontinuous Garlekin space function
  V = dir_problem.V     # Continuous Garlekin space function

  # l=L-1                                             #Number of experiments

  # HOMOGENEUS MESH
  N = 128               # grid with N*N points (works well with 0 < N < 400)
  h = 2*radius/(N-1)    # step-size
  x = [radius - i*h for i in range(N)]  # x grid points
  y = [-radius + i*h for i in range(N)] # y grid points

  # MESH x and y
  mesh_x = np.zeros((N,N))                              # x-Data (input of CNN)
  mesh_y = np.zeros((N,N))                              # y-Data (input of CNN)
  for i in range(N):
    for j in range(N):
      mesh_x[i][j] = x[i]
      mesh_y[i][j] = y[j]

  eit_img = EIT_Image(dir_problem.mesh,mesh_x,mesh_y)

  "Define sigma as constant = Background"
  gamma0 = dolfinx.fem.Function(V0) #Define the function with basis DG
  iv, bg= 10, 1.2
  gamma0.x.array[:] = bg

  delx_phi = dolfinx.fem.Function(V0)
  dely_phi = dolfinx.fem.Function(V0)

  import tensorflow as tf

  exper = ['1_1','1_2', '1_3', '1_4', '2_2','2_3','2_4','2_5','2_6','3_1','3_2','3_6','3_4','3_5','4_1' ,'4_3', '4_4','5_2']    # experiments
  n_exper = len(exper)

  T1 = []
  for sample in range(n_exper):
    #Load experimental data
    mat = scipy.io.loadmat(DATAMAT_PATH+'datamat_' + exper[sample])
    # mat = scipy.io.loadmat(exper)
    Uel=mat.get("Uel").T
    # CP=mat.get("CurrentPattern").T

    #Selecting Potentials
    Uel_f=Uel[:16][currents] #Matrix of measuarements

    #Selecting Potentials
    list_U1_m=np.zeros_like(Uel_f)

    #Convert type of data
    for index, potential in enumerate(Uel_f):
        list_U1_m[index]=eitx.ConvertingData(potential, method="KIT4")

    # Difference of potential
    differ = [list_U1_m[k] - list_U0_m[k] for k in range(len(list_U0_m))]

    "Solve Forward Problem with Background and Difference of Potentials as Currents"
    list_ur_dif, list_U_dif = dir_problem.solve_problem_current(differ, gamma0)

    "Define data in a homogeneus grid for test"
    list_delx_phi = []
    list_dely_phi = []

    for k in range(l):
        gradphi_array = eitx.compute_gradient(list_ur_dif[k])

        delx_phi.x.array[:] = gradphi_array[:,0]
        dely_phi.x.array[:] = gradphi_array[:,1]

        delx_img = eit_img.genPotentialImg(delx_phi)
        dely_img = eit_img.genPotentialImg(dely_phi)

        list_delx_phi.append(delx_img)
        list_dely_phi.append(dely_img)

    T = np.zeros((2*l + 2,N,N))
    T[1] = mesh_x
    T[0] = mesh_y
    for k in range(l):
        T[2+2*k] = list_delx_phi[k]
        T[2+2*k+1] = list_dely_phi[k]        

    vec_list = []
    for i in range(N):
        for j in range(N):
            if T[0,i,j]**2 + T[1,i,j]**2 < 1:    
                vec_list.append(T[:,i,j])


    T1.append(np.array(vec_list))


  classes = []


  'Upload model'

  'Predict and prepare images to plot'
  
  #uploaded = files.upload()

  model = keras.models.load_model(os.path.join(FILEPATH,MODELPATH,'fnn.keras'))
  model.summary()

  for input_mat in T1:     
      input_val = tf.convert_to_tensor(input_mat)
      pred = model.predict(input_val)

      classes.append(
         eit_img.get_fnn_matrix(pred.flatten())
      )

  print(np.array(input_val).shape)

  mat.keys()


  result = 0.5*np.ones((n_exper,N,N))
  for k in range(n_exper):
    result1 = classes[k]
    for i in range(N):
      for j in range(N):
        if x[i]**2 + y[j]**2 > radius**2:
          result1[i][j] = 0.0
    result[k] = result1

  'prepare target photo list'
  # plt.figure(figsize=(20, 20))
  photo_array = []
  for test in range(len(exper)):
    img = np.asarray(Image.open(os.path.join(FILEPATH,'fin_data/target_photos/fantom_' + exper[test] + '.jpg')))
    photo_array.append(img)

  'Plot'
  # plt.figure(figsize=(10, 40))
  fig, ax = plt.subplots(result.shape[0],2,figsize=(10,40))
  img_array = []
  for k in range(result.shape[0]):
    img_array.append(ax[k][0].imshow(result[k], interpolation='none',vmin=np.min([np.min(result[k]),0.0]),vmax=1.0))
    ax[k][0].set_axis_off()
    ax[k][1].imshow(photo_array[k])
    ax[k][1].set_axis_off()

  fig.colorbar(img_array[0],ax=ax,orientation='vertical')
  plt.savefig(os.path.join(RESULTS_PATH,'test_result.png'))
  np.save(os.path.join(RESULTS_PATH,'test_result'),result)


if __name__=='__main__':
  RESULTS_PATH = sys.argv[1]
   
  try:
    main(RESULTS_PATH)
  except Exception as e:
    logging.error(f"DSM fnn datagen failed calling {sys.argv[1]} config file")
    logging.error(traceback.format_exc())
    print(traceback.format_exc())