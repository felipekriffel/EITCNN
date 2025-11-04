import eitx
import scipy as sp
import numpy as np
import dolfinx
from petsc4py import PETSc
import json
import os
import sys
from eit_image import EIT_Image

def main(SETTINGS_JSON):

  settings = json.loads(SETTINGS_JSON)
  
  if not os.path.isdir(settings['dsm_datapath']):
      os.mkdir(settings['dsm_datapath'])

  with open(settings['dsm_datapath']+"/data_info.json","w") as f:
    f.write(json.dumps(settings))

  "Importing modules"
  import logging
  # Set the logging level to suppress most logs
  logging.getLogger('UFL_LEGACY').setLevel(logging.WARNING)
  logging.getLogger('dolfin').setLevel(logging.WARNING)
  logging.getLogger('PETSc').setLevel(logging.WARNING)
  logging.getLogger('FFC').setLevel(logging.WARNING)


  #Loading kit4 data (somente para definir a corrente de maneira correta)
  mat = sp.io.loadmat("fin_data/datamat/datamat_1_2")
  CP = mat.get("CurrentPattern").T

  #Current
  I_all=CP[0:16][settings['currents']]/np.sqrt(2)
  print('Currents:')
  print(I_all)
  l, L=np.shape(I_all) #Number of experiments = 15, Number of Electrodes = 16
  # print(I_all)# MESH (For real data)

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

  ## Direct problem
  dir_problem = eitx.DirectProblem(mesh_object,z)
  V0 = dir_problem.V0   # Discontinuous Garlekin space function
  V = dir_problem.V     # Continuous Garlekin space function

  # 'Plot'

  #"Define gamma as constant = Background"
  bg = settings['bg']
  ivhigh,ivlow = settings['ivhigh'], settings['ivlow']
  gamma0 = dolfinx.fem.Function(V0)
  gamma0.x.array[:] = bg

  #Solving Forward Problem
  list_u, list_U0_m = dir_problem.solve_problem_current(I_all, gamma0)
  list_U0 = np.array(list_U0_m).flatten()

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

  eit_img = EIT_Image(dir_problem.mesh,mesh_x,mesh_y)

  gamma = dolfinx.fem.Function(V0)      # Empty function
  T1 = []                               # To save data

  samples_dir = settings['samples_dir']
  samples_names = [file for file in os.listdir(samples_dir) if file.endswith(".npy")]

  # Loop for generating data
  noise_level = settings["noise_level"] # % of artificial noise in data
  for sample in samples_names:
    if os.path.exists(os.path.join(settings['dsm_datapath'],sample.replace(".npy","_dsm.npy"))):
      print(f"{sample} dsm data already computed, skipping")
      continue
    else:
      print("Computing", sample)
        
    
    gamma.x.array[:]= np.load(os.path.join(samples_dir, sample))

    "Define data in a homogeneus grid for training"
    A = eit_img.genGammaImg(gamma,bg,ivhigh,ivlow,settings['img_type'])

    "Solve Forward Problem with Background + Inclusion"
    list_u1, list_U1_m = dir_problem.solve_problem_current(I_all, gamma)

    "Difference of Resulting Potentials"
    differ = np.array(list_U1_m) - np.array(list_U0_m)
    noise = np.random.uniform(-1, 1, size=(len(differ),len(differ[0])))
    noise = noise / np.linalg.norm(noise)
    differ_noisy = differ + noise_level*noise*np.linalg.norm(differ)

    "Solve Forward Problem with Background and Difference of Potentials as Currents"
    list_ur_dif, list_U_dif = dir_problem.solve_problem_current(differ_noisy, gamma0)

    "Saves data on tensor"
    T = np.zeros((l + 3,N,N))
    for k in range(l):
      T[k] = eit_img.genPotentialImg(list_ur_dif[k],0)

    T[l] = mesh_x
    T[l+1] = mesh_y
    T[l+2] = A
    np.save(os.path.join(settings['dsm_datapath'],sample.replace(".npy","_dsm_cnn")),T)
    
  # np.save('EIT_Data_for_CNN', T1)
  print(f'Data saved at {settings["dsm_datapath"]}.')

if __name__=="__main__":
  SETTINGS_JSON = sys.argv[1]
  if SETTINGS_JSON.endswith('.json') and os.path.isfile(SETTINGS_JSON):
    with open(SETTINGS_JSON) as f:
      SETTINGS_JSON = f.read()
      
  main(SETTINGS_JSON)