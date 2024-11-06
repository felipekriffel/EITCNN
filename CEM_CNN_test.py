import dolfinx
import pyvista
import eitx
import os
import json 
import sys

def main(RESULTS_PATH):
  FILEPATH = ''
  DATAMAT_PATH = "ifsc_data2/20240506"
  bg_estimated = 0.02 # 0.3
  rotacao = 90
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

  if not os.path.isdir(RESULTS_PATH):
    os.mkdir(RESULTS_PATH)


  # Load data of background
  mat = scipy.io.loadmat(f"{DATAMAT_PATH}/Dados/Vref.mat")
  #mat = scipy.io.loadmat(f"{DATAMAT_PATH}/datamat/Referencia.mat")
  Uel=mat.get("signal_peak")

  # mat = scipy.io.loadmat("Referencia.mat")
  # Uel=mat.get("signal_peak")

  # mat = scipy.io.loadmat("Vref.mat")
  # Uel=mat.get("signal_peak")

  #print(Uel.shape)
  # plt.plot(Uel.T, '-r')

  #Selecting Potentials
  Uel_b = Uel.reshape(16,16)    #Matrix of measuarements
  #print(Uel_b.shape)
  l, L=np.shape(Uel_b)  #Number of experiments, Number of Electrodes

  #Plot
  #fig, ax = plt.subplots(figsize=(8,5))
  #for i, U_vec in enumerate(Uel_b):
  #    x=np.linspace(1,1.8,L)+i
  #    ax.plot(x,U_vec, linewidth=1.3, marker='.', markersize=5);

  print(np.sum(Uel_b[0]))
  # Uel_b[0][0] = -2
  # Forces sum = 0 on each experiment
  for i in range(L):
    Uel_b[i][i] = -np.sum(Uel_b[i]) + Uel_b[i][i]
    #Uel_b[i][i]=-Uel_b[i][i] #Em cada primeiro emissor, troque de sinal
    # Uel_b[i] -= np.sum(Uel_b[i])/L #Force a soma ser zero em cada experimento

  #Plot
  #fig, ax = plt.subplots(figsize=(8,5))
  #for i, U_vec in enumerate(Uel_b):
  #    x=np.linspace(1,1.8,L)+i
  #    ax.plot(x,U_vec, linewidth=1.3, marker='.', markersize=5);


  #Selecting Potentials
  list_U0_m=np.zeros_like(Uel_b)

  # Convert type of data
  for index, potential in enumerate(Uel_b):
      list_U0_m[index]=eitx.ConvertingData(potential, method="KIT4")
  list_U0_m = -list_U0_m #/np.max(list_U0_m)
  #list_U0=list_U0_m.flatten() #Matrix to vector
  # list_U0_m =Uel_b

  #Plot
  #fig, ax = plt.subplots(figsize=(8,5))
  #for i, U_vec in enumerate(list_U0_m):
  #    x=np.linspace(1,1.8,L)+i
  #    ax.plot(x,U_vec, linewidth=1.3, marker='.', markersize=5)
  #plt.show()

  #Current
  # L = settings['L']
  #n_g = settings['n_g']
  #I_all= eitx.current_method( L , n_g, method=2)          #Currents
  # print(I_all)# MESH (For real data)

  "Basic Definitions"
  radius=1       #Circle radius
  # L=16           #Number of Electrodes
  per_cober=0.3543  #Percentage of area covered by electrodes
  rotate= 0      #Electrodes Rotation

  'Return object with angular position of each electrode'
  ele_pos = eitx.Electrodes(L, per_cober, rotate,anticlockwise=False)
  #refine_n = 8     #Refinement mesh
  #n_in = 8         #Vertex on elec.
  #n_out = 2        #Vertex on gaps (Sometimes it is important.)

  # CURRENT
  'Basic Definitions'
  # z_r=np.ones(L)*0.025E-3                         #Impedance of each electrode
  # z_r=np.ones(L)*1e-3
  z = np.ones(L)*1e-3

  'Mesh'
  # mesh_inverse=MyMesh(radius, refine_n, n_in, n_out, ele_pos)
  mesh_object = eitx.MeshClass(ele_pos,0.4,0.6)
  #mesh = mesh_object.mesh

  ## Direct problem
  dir_problem = eitx.DirectProblem(mesh_object,z)
  V0 = dir_problem.V0   # Discontinuous Garlekin space function
  # V = dir_problem.V     # Continuous Garlekin space function

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

  ME = []
  ME.append([0, 1, 15])
  i, j, k = 0, 1, 2
  ME.append([i, j, k])
  while k < 15:
    i, j, k = i+1, j+1, k+1
    ME.append([i, j, k])
  ME.append([0, 14, 15])
  #print(ME)


  "Define sigma as constant = Background"
  gamma0 = dolfinx.fem.Function(V0) #Define the function with basis DG
  iv, bg= 10, 1/bg_estimated
  gamma0.x.array[:] = bg

  import tensorflow as tf

  exper = [name.replace(".mat","") for name in os.listdir(DATAMAT_PATH+'/Dados')]    # experiments

  n_exper = len(exper)

  T1 = []
  for sample in range(n_exper):
    #Load experimental data
    mat = scipy.io.loadmat(DATAMAT_PATH+'/Dados/' +exper[sample]+".mat")
    # mat = scipy.io.loadmat(exper)
    Uel=mat.get("signal_peak").T
    # CP=mat.get("CurrentPattern").T

    #Selecting Potentials
    Uel_f=Uel.reshape(l,L) #Matrix of measuarements

    # Forces sum = 0 on each experiment
    for i in range(0,L):
      Uel_f[i][i] = -np.sum(Uel_f[i]) + Uel_f[i][i]
  
    #Selecting Potentials
    list_U1_m=np.zeros_like(Uel_f)
    # list_U1_m=Uel_f

    #Convert type of data
    for index, potential in enumerate(Uel_f):
        list_U1_m[index]=eitx.ConvertingData(potential, method="KIT4")
    list_U1_m = -list_U1_m #/np.max(list_U1_m)
  
    # Difference of potential
    differ = [list_U1_m[k] - list_U0_m[k] for k in range(len(list_U0_m))]
    for s in range(16):
      differ[s][ME[s]] = 0
    
    for i in range(len(differ)):
      differ[i] = differ[i] - np.sum(differ[i])/13

    for s in range(16):
        differ[s][ME[s]] = 0

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


  model = keras.models.load_model(FILEPATH+MODELPATH+'unet.keras')
  model.summary()

  'Predict and prepare images to plot'
  from scipy.ndimage import rotate

  classes = model.predict(input_val)
  result = 0.5*np.ones((n_exper,N,N))
  for k in range(n_exper):
    result1 = rotate(classes[k],rotacao)
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
    img = np.asarray(Image.open(f'{DATAMAT_PATH}/Fotos/' + exper[test] + '.jpg'))
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
  plt.savefig(RESULTS_PATH+'test_result.png')

if __name__=='__main__':
   RESULTS_PATH = sys.argv[1]
   main(RESULTS_PATH)