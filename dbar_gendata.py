import dolfinx
import pyvista
import eitx
import scipy as sp
pyvista.set_jupyter_backend("static")
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import scipy

import scipy.io
"Forward problem in background"

DATAPATH = "fin_data/datamat/"
mat = sp.io.loadmat(DATAPATH+"datamat_1_0")
CP = mat.get("CurrentPattern").T

#Current
I_all=CP[0:16]/np.sqrt(2)
l, L=np.shape(I_all) #Number of experiments = 15, Number of Electrodes = 16
print("Currents:\n",I_all)# MESH (For real data)

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
mesh_object = eitx.MeshClass(ele_pos,0.4,0.4)
mesh = mesh_object.mesh

## Direct problem
dir_problem = eitx.DirectProblem(mesh_object,z)
V0 = dir_problem.V0   # Discontinuous Garlekin space function
V = dir_problem.V     # Continuous Garlekin space function

#"Define gamma as constant = Background"
iv, bg= 10, 1
ivhigh,ivlow = 10, 0.1
gamma0 = dolfinx.fem.Function(V0)
gamma0.x.array[:] = bg

# "Plot"
eitx.plot_indicator_function(gamma0)

#Solving Forward Problem
list_u, list_U0_m = dir_problem.solve_problem_current(I_all, gamma0)
list_U0 = np.array(list_U0_m).flatten()

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

DBAR_PATH = "dbar_data/"
mat1 = scipy.io.loadmat("KIT4_measdata_dbar/dataMat_adj_1_1.mat")

# Loop for generating data

n_samples = [0,800,800]               # number of samples in order: 1 circle, 2 circles, 3 circles, etc.
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

    #Plot and save fig
    # eitx.plot_indicator_function(gamma,True,DBAR_PATH+f"solutions/sample_{m+1}_{sample}")
    np.save(DBAR_PATH+f"solutions/sample_{m+1}_{sample}.npy",gamma_prov)

    #Solve Forward Problem with Background + Inclusion
    list_u1, list_U1_m = dir_problem.solve_problem_current(I_all, gamma)

    # np.save(DBAR_PATH+f"ground_format_potential/sample_{m+1}_{sample}",list_U1_m)
    # np.save(DBAR_PATH+f"adjacent_format_potential/sample_{m+1}_{sample}",eitx.ConvertingData(list_U1_m,"adjacent"))
    scipy.io.savemat(DBAR_PATH+f"data_matrix/sample_{m+1}_{sample}.mat",{
      "U_ad0": np.array(list_U1_m).T,
      "U_ad10": np.array(list_U0_m).T,
      "MeasPat": mat1["MeasPat"]
    })

    "Define data in a homogeneus grid for training"
    sol_img = eitx.genGammaImg(gamma,mesh_x,mesh_y,bg,ivhigh,ivlow)
    
    np.save(DBAR_PATH+f"solutions/sample_{m+1}_{sample}_img",sol_img)
    
    # plt.imshow(sol_img,interpolation=None)
    # plt.savefig(DBAR_PATH+f"solutions/sample_{m+1}_{sample}_img.png")
    # scipy.io.savemat(DBAR_PATH+f"solutions/sample_{m+1}_{sample}_img.mat",{
    #   "gamma": sol_img
    # })
    
  print('Generation of ' + str(m + 1) + ' circle(s) ended.')
# np.save('EIT_Data_for_CNN', T1)
# print('Data saved at file named: EIT_Data_for_CNN.')
# print(np.array(T1).shape)