import dolfinx
import pyvista
import eitx
import os
import json 
import sys

def main(RESULTS_PATH):
  FILEPATH = ''
  DATAMAT_PATH = "fin_data/datamat/"
  SETTINGS_PATH = RESULTS_PATH + "/data_info.json"
  TRAIN_SETTINGS_PATH = "unet_train_settings.json"

  with open(SETTINGS_PATH) as f:
      settings = json.loads(f.read())

  with open(TRAIN_SETTINGS_PATH) as f:
      train_settings = json.loads(f.read())

  currents = settings['currents']
  SAVEPATH = train_settings['savepath']
  MODELPATH = SAVEPATH

  'Load files'

  from PIL import Image

  import matplotlib.pyplot as plt
  import numpy as np
  import scipy

  #Load data of background
  mat = scipy.io.loadmat(DATAMAT_PATH+"datamat_1_0")
  Uel=mat.get("Uel").T
  CP=mat.get("CurrentPattern").T

  if not os.path.isdir(SAVEPATH):
      os.mkdir(SAVEPATH)

  #Selecting Potentials
  Uel_b=Uel[-15:][currents] #Matrix of measuarements
  print(Uel_b.shape)

  #Selecting Potentials
  list_U0_m=np.zeros_like(Uel_b)

  #Convert type of data
  for index, potential in enumerate(Uel_b):
      list_U0_m[index]=eitx.ConvertingData(potential, method="KIT4")
  list_U0=list_U0_m.flatten() #Matrix to vector

  #Current
  I_all=CP[-15:][currents]/np.sqrt(2)
  l, L=np.shape(I_all) #Number of experiments = 15, Number of Electrodes = 16

  "Basic Definitions"
  radius=1       #Circle radius
  # L=16           #Number of Electrodes
  per_cober=0.5  #Percentage of area covered by electrodes
  rotate= 0      #Electrodes Rotation

  'Return object with angular position of each electrode'
  ele_pos = eitx.Electrodes(L, per_cober, rotate)
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

  "Define sigma as constant = Background"
  gamma0 = dolfinx.fem.Function(V0) #Define the function with basis DG
  iv, bg= 10, 1.2
  gamma0.x.array[:] = bg

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
    Uel_f=Uel[-15:][currents] #Matrix of measuarements

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
    T = np.zeros((l + 2,N,N))
    for k in range(l):
      T[k] = eitx.genPotentialImg(list_ur_dif[k],mesh_x,mesh_y,bg)

    T[l] = mesh_x
    T[l+1] = mesh_y
    T1.append(np.transpose(T))
  input_val = tf.convert_to_tensor(T1)
  print(np.array(input_val).shape)

  mat.keys()

  'Upload model'

  from tensorflow import keras
  #uploaded = files.upload()


  model = keras.models.load_model(FILEPATH+MODELPATH+'/unet.keras')
  model.summary()

  'Predict and prepare images to plot'
  from scipy.ndimage import rotate

  classes = model.predict(input_val)
  result = 0.5*np.ones((n_exper,N,N))
  for k in range(n_exper):
    result1 = rotate(classes[k],180)
    # result1 = classes[k]
    for i in range(N):
      for j in range(N):
        if x[i]**2 + y[j]**2 > radius**2:
          result1[i][j] = 0.5
    result[k, :, :] = result1[:,:,0]

  result.shape

  'prepare target photo list'
  # plt.figure(figsize=(20, 20))
  photo_array = []
  for test in range(len(exper)):
    img = np.asarray(Image.open(FILEPATH+'fin_data/target_photos/fantom_' + exper[test] + '.jpg'))
    photo_array.append(img)

  'Plot'
  # plt.figure(figsize=(10, 40))
  fig, ax = plt.subplots(result.shape[0],2,figsize=(10,40))
  img_array = []
  for k in range(result.shape[0]):
    img_array.append(ax[k][0].imshow(result[k], interpolation='none',vmin=-1.0,vmax=1.0))
    ax[k][0].set_axis_off()
    ax[k][1].imshow(photo_array[k])
    ax[k][1].set_axis_off()

  fig.colorbar(img_array[0],ax=ax,orientation='vertical')
  plt.savefig(SAVEPATH+'test_result.png')

if __name__=='__main__':
   RESULTS_PATH = sys.argv[1]
   main(RESULTS_PATH)