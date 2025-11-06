import dolfinx
import gmsh
import pyvista
import eit_cont
import scipy as sp
pyvista.set_jupyter_backend("static")
import ufl
import os
from PIL import Image
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sys
import logging

logging.basicConfig(
    filename='experiments.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main(SETTINGS_JSON):
    # with open(SETTINGS_PATH) as f:
    #     settings = json.loads(f.read())

    settings = json.loads(SETTINGS_JSON)

    if not os.path.isdir(settings['datapath']):
        os.mkdir(settings['datapath'])

    with open(os.path.join(settings['datapath'],"data_info.json"),"w") as f:
        f.write(json.dumps(settings))
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

    # 'Plot'
    eit_cont.plot_mesh(mesh)

    #"Define gamma as constant = Background"
     #"Define gamma as constant = Background"
    bg = settings['bg']
    ivhigh,ivlow = settings['ivhigh'], settings['ivlow']
    p_ivhigh, p_ivlow = settings['p_ivhigh'], settings['p_ivlow']
    gamma0 = dolfinx.fem.Function(V0)
    gamma0.x.array[:] = bg


    samples_info = []

  # Loop for generating data
    n_samples = settings["n_samples"] # number of samples in order: 1 circle, 2 circles, 3 circles, etc.

    nn_samples = len(n_samples)
    print("Generating circle data:")
    for n in range(len(n_samples)):
        print(f"{n+1} Circles:", n_samples[n])

    print("Total:",nn_samples,"\n")
    multi = 4 # number of samples is multiplied by this number
    for n_circles in range(nn_samples):
        m_multi_m = (multi**n_circles)*n_samples[n_circles]
        rad_1 = np.random.uniform(0.15*radius, 0.3*radius, (n_circles+1, m_multi_m))                # radius of the inclusions
        center_xy = np.random.uniform(-radius*0.5, radius*0.5, size=(2*(n_circles+1), m_multi_m))   # xy-center of the inclusions
        vecpop = [] # vector with indices to withdraw
        for p in range(n_circles):
            for q in range(n_circles - p):
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

        if rad_1.shape[1] < n_samples[n_circles]:
            print('There are not enough samples. Correting the number of samples to: ' + str(rad_1.shape[1]))
            n_samples[n_circles] = rad_1.shape[1]

        for sample in range(n_samples[n_circles]):
            if n_circles > 0:
                sample_id = str(np.sum(n_samples[:n_circles]) + sample + 1)
            else:
                sample_id = str(sample + 1)
            
            print('Generating sample: ' + sample_id)
            
            sample_data = {
                "n_circles": n_circles+1,
                "id": sample,
                "inclusions": []
            }

            #"Generate Background + Inclusion"
            gamma = eit_cont.GammaCircle(V0,1,bg,0,0, 0)
            gamma_prov = gamma.x.array
            for p in range(n_circles+1):
                iv = np.random.choice([ivhigh,ivlow],p=[p_ivhigh,p_ivlow])
                sample_data['inclusions'].append(
                    {
                        "center_x:": center_xy[2*p,sample],
                        "center_y": center_xy[2*p+1,sample],
                        "radius":rad_1[p,sample],
                        "iv": iv
                    }
                )
                ValuesCells1 = eit_cont.GammaCircle(V0,iv-bg,0.0,rad_1[p,sample],center_xy[2*p,sample], center_xy[2*p+1,sample]).x.array
                gamma_prov = gamma_prov + ValuesCells1

            np.save(f"{settings['datapath']}/sample_{n_circles+1}_{sample}",gamma_prov) #Salva array de coeficientes
            samples_info.append(sample_data)
        print('Generation of ' + str(n_circles + 1) + ' circle(s) ended.')

    with open(os.path.join(settings['datapath'],"samples_info.json"),"w") as f:
        f.write(json.dumps(samples_info))
    print(f'Data saved at {settings["datapath"]}.')

if __name__=="__main__":
    SETTINGS_JSON = sys.argv[1]
    if SETTINGS_JSON.endswith('.json') and os.path.isfile(SETTINGS_JSON):
        with open(SETTINGS_JSON) as f:
            SETTINGS_JSON = f.read()

    try:
        main(SETTINGS_JSON)
    except Exception as e:
        logging.error(f"Data generation failed calling {sys.argv[1]} config file")
        logging.error(e)