import eitx
import scipy as sp
import numpy as np
import dolfinx
from matplotlib import pyplot as plt

"Importing modules"
import logging
# Set the logging level to suppress most logs
logging.getLogger('UFL_LEGACY').setLevel(logging.WARNING)
logging.getLogger('dolfin').setLevel(logging.WARNING)
logging.getLogger('PETSc').setLevel(logging.WARNING)
logging.getLogger('FFC').setLevel(logging.WARNING)

from petsc4py import PETSc
print(PETSc.ScalarType)

"Forward problem in background"

#Loading data (somente para definir a corrente de maneira correta)
mat = sp.io.loadmat("datamat_1_2")
CP = mat.get("CurrentPattern").T

#Current
I_all=CP[-15:]/np.sqrt(2)
l, L=np.shape(I_all) #Number of experiments = 15, Number of Electrodes = 16
# print(I_all)# MESH (For real data)

"Basic Definitions"
radius=1               #Circle radius
per_cober=0.454728409  #Percentage of area covered by electrodes
rotate=0               #Electrodes Rotation
z=np.ones(L)*0.07858  

'Return object with angular position of each electrode'
ele_pos = eitx.Electrodes(L, per_cober, rotate)
refine_n = 8     #Refinement mesh
n_in = 8         #Vertex on elec.
n_out = 2        #Vertex on gaps (Sometimes it is important.)

'Mesh'
# mesh_inverse=MyMesh(radius, refine_n, n_in, n_out, ele_pos)
mesh_object = eitx.MeshClass(ele_pos,0.4,0.6)
mesh = mesh_object.mesh


## Direct problem
dir_problem = eitx.DirectProblem(mesh_object,z)
V0 = dir_problem.V0   # Discontinuous Garlekin space function
V = dir_problem.V     # Continuous Garlekin space function

# 'Plot'
# eitx.plot_mesh(mesh)

#"Define gamma as constant = Background"
iv, bg= 10, 1
gamma0 = dolfinx.fem.Function(V0)
gamma0.x.array[:] = bg

# # "Plot"
# eitx.plot_indicator_function(gamma0)

#Solving Forward Problem
list_u, list_U0_m = dir_problem.solve_problem_current(I_all, gamma0)
list_U0 = np.array(list_U0_m).flatten()

'Plot'
#np.set_printoptions(linewidth=750, precision=3, suppress=False, threshold=1000, formatter=None)
#print(np.matrix(np.round(list_U0_m, 3)))
'Retangular Mesh'

N = 128                               # grid with N*N points (works well with 0 < N < 400)
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

T1 = []                               # To save data

# Loop for generating data

n_samples = [1200]               # number of samples in order: 1 circle, 2 circles, 3 circles, etc.
noise_level = 0.0                   # % of artificial noise in data

nn_samples = len(n_samples)
multi = 4                        # number of samples is multiplied by this number
for m in range(nn_samples):
  m_multi_m = (multi**m)*n_samples[m]
  rad_1 = np.random.uniform(0.15*radius, 0.3*radius, (m+1, m_multi_m))                # radius of the inclusions
  center_xy = np.random.uniform(-radius*0.5, radius*0.5, size=(2*(m+1), m_multi_m))   # xy-center of the inclusions
  # print(center_xy), print(rad_1), print(noise)
  vecpop = []                         # vector with indices to withdraw
  for p in range(m):
    for q in range(m - p):
      for j in range(m_multi_m):
        'withdraw indices when iclusions are overlapped'
        x0 = center_xy[2*p,j]
        y0 = center_xy[2*p+1,j]
        x1 = center_xy[2*(p+q)+2,j]
        y1 = center_xy[2*(p+q)+3,j]
        sumrad = rad_1[p,j] + rad_1[(p+q)+1,j]
        dist2 = (x0 - x1)**2 + (y0 - y1)**2
        if dist2 < sumrad**2 + 0.05*radius:               # if the circles touch each other
          vecpop.append(j)
  rad_1 = np.delete(rad_1,vecpop,1)     # remove the indices when the circles touch each other
  center_xy = np.delete(center_xy,vecpop,1)
  if rad_1.shape[1] < n_samples[m]:
    print('There are not enough samples. Correting the number of samples to: ' + str(rad_1.shape[1]))
    n_samples[m] = rad_1.shape[1]

  for sample in range(n_samples[m]):
    if m > 0:
      print('Generating sample: ' + str(np.sum(n_samples[:m]) + sample + 1))
    else:
      print('Generating sample: ' + str(sample + 1))

    "Generate Background + Inclusion"
    gamma = eitx.GammaCircle(V0,iv,bg,rad_1[0,sample],center_xy[0,sample], center_xy[1,sample])
    gamma_prov = gamma.x.array
    for p in range(m):
      ValuesCells1 = eitx.GammaCircle(V0,iv,0.0,rad_1[p+1,sample],center_xy[2*p+2,sample], center_xy[2*p+3,sample]).x.array
      gamma_prov = np.minimum(gamma_prov + ValuesCells1, iv)
    gamma.x.array[:]= gamma_prov

    #"Plot"
    # eitx.plot_indicator_function(gamma)

    "Define data in a homogeneus grid for training"
    A = eitx.genGammaImg(gamma,mesh_x,mesh_y,bg)

    "Solve Forward Problem with Background + Inclusion"
    list_u1, list_U1_m = dir_problem.solve_problem_current(I_all, gamma)

    "Difference of Resulting Potentials"
    differ = np.array(list_U1_m) - np.array(list_U0_m)
    noise = np.random.uniform(-1, 1, size=(len(differ),len(differ[0])))                               # noise to be added in data
    noise = noise / np.linalg.norm(noise)
    differ_noisy = differ + noise_level*noise*np.linalg.norm(differ)
    #print(np.linalg.norm(differ_noisy - differ)/np.linalg.norm(differ))
    
    ##Plot
    # fig, ax = plt.subplots(figsize=(8,5))
    # for w, U_vec in enumerate(differ_noisy):
    #  zx=np.linspace(1,1.8,L) + w
    #  ax.plot(zx,U_vec, linewidth=1.3, marker='.', markersize=5);

    "Solve Forward Problem with Background and Difference of Potentials as Currents"
    list_ur_dif, list_U_dif = dir_problem.solve_problem_current(differ_noisy, gamma0)

    # 'Plot'
    # plt.figure(figsize=(10, 10))
    # for i in range(0, l):
      #  plt.subplot(4,4,i+1)
      #  eitx.plot_tent_function(list_ur_dif[i])
    # plt.show()

    "Define data in a homogeneus grid for training"
    T = np.zeros((l + 3,N,N))
    for k in range(l):
      T[k] = eitx.genPotentialImg(list_ur_dif[k],mesh_x,mesh_y,bg)
    #   for i in range(N):
    #     for j in range(N):
    #       if x[i]**2 + y[j]**2 < radius**2:         # if the grid point is in the circle
    #         T[k][i][j] = list_ur_dif[k](x[i],y[j])
    #       else:
    #         T[k][i][j] = 0
    # for w in T[k]:
    #   ' '.join([str(x) for x in w] )

    #'Plot'
    # plt.figure(figsize=(10, 10))
    # for i in range(0, l):
    #    plt.subplot(4,4,i+1)
    #    plt.imshow(T[i], interpolation='none')
    # plt.show()
    T[l] = mesh_x
    T[l+1] = mesh_y
    T[l+2] = A
    T1.append(T)
  print('Generation of ' + str(m + 1) + ' circle(s) ended.')
np.save('EIT_Data_for_CNN', T1)
print('Data saved at file named: EIT_Data_for_CNN.')
print(np.array(T1).shape)