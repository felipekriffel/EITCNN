from approxinv import *
import eitx
import scipy as sp
import numpy as np
import dolfinx
import json
import os
import sys

def main(SETTINGS_JSON):
    from petsc4py import PETSc
    print(PETSc.ScalarType)

    settings = json.loads(SETTINGS_JSON)
    # settings['n_g'] = len(settings["currents"])
    datapath = settings['datapath']

    "Forward problem in background"

    #Loading data (somente para definir a corrente de maneira 
    mat = sp.io.loadmat("fin_data/datamat/datamat_1_2")
    CP = mat.get("CurrentPattern").T

    #Current
    I_all=CP[-15:]/np.sqrt(2)
    l, L=np.shape(I_all) #Number of experiments = 15, Number of 
    # print(I_all)# MESH (For real data)

    "Basic Definitions"
    per_cober=0.454728409  #Percentage of area covered by 
    rotate=0               #Electrodes Rotation
    z=np.ones(L)*0.07858  

    'Return object with angular position of each electrode'
    ele_pos = eitx.Electrodes(L, per_cober, rotate)

    'Mesh'
    # mesh_inverse=MyMesh(radius, refine_n, n_in, n_out, ele_pos)
    mesh_object = eitx.MeshClass(ele_pos,0.4,0.6)

    ## Direct/Inverse problem
    inv_problem = eitx.InverseProblem(mesh_object,z,I_all)
    V0 = inv_problem.V0   # Discontinuous Garlekin space function

    bg = settings['bg']
    gamma0 = dolfinx.fem.Function(V0)
    gamma0.x.array[:] = bg
    
    u0_list,U0_list = inv_problem.solve_problem_current(I_all,gamma0)
    U0_array = np.array(U0_list).flatten()

    A = inv_problem.calc_jacobian(u0_list)
    b = U0_array - A@gamma0.x.array
    
    k_list = settings['k_list']

    alpha_array = settings['alpha_list']
    for alpha in alpha_array:
        Wk_list = get_wk_matrices(A,b,alpha,k_list)
        bk_list = get_bk_vectors(Wk_list,b)
        for Wk,bk,k in zip(Wk_list,bk_list,k_list):
            np.save(f"{datapath}/matrix/W_{k}_alpha_{alpha}",Wk)
            np.save(f"{datapath}/matrix/b_{k}_alpha_{alpha}",bk)


if __name__=="__main__":
    if len(sys.argv)>1:
        SETTINGS_JSON = sys.argv[1]
        if SETTINGS_JSON.endswith('.json') and os.path.isfile (SETTINGS_JSON):
            with open(SETTINGS_JSON) as f:
                SETTINGS_JSON = f.read()
    else:
        SETTINGS_JSON = "settings/approxinv_settings.json"
        with open(SETTINGS_JSON) as f:
            SETTINGS_JSON = f.read()

    main(SETTINGS_JSON)