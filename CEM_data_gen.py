import eitx
import scipy as sp
import numpy as np
import dolfinx
import json
import os
import sys
import logging

logging.basicConfig(
    filename='experiments.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main(SETTINGS_JSON):
    # with open(SETTINGS_PATH) as f:
    #         settings = json.loads(f.read())

    settings = json.loads(SETTINGS_JSON)

    if not os.path.isdir(settings['datapath']):
        os.mkdir(settings['datapath'])

    with open(settings['datapath']+"/data_info.json","w") as f:
        f.write(json.dumps(settings))

    #"Importing modules"
    import logging
    # Set the logging level to suppress most logs
    logging.getLogger('UFL_LEGACY').setLevel(logging.WARNING)
    logging.getLogger('dolfin').setLevel(logging.WARNING)
    logging.getLogger('PETSc').setLevel(logging.WARNING)
    logging.getLogger('FFC').setLevel(logging.WARNING)

    from petsc4py import PETSc
    print(PETSc.ScalarType)

    #Current
    L = 16#Number of experiments = 15, Number of Electrodes = 16
    
    #Mesh Definitions
    radius=1                             #Circle radius
    per_cober=0.454728409    #Percentage of area covered by electrodes
    rotate=0                             #Electrodes Rotation
    ele_pos = eitx.Electrodes(L, per_cober, rotate) #'Return object with angular position of each electrode'

    # mesh_inverse=MyMesh(radius, refine_n, n_in, n_out, ele_pos)
    mesh_object = eitx.MeshClass(ele_pos,0.4,0.6)
    mesh = mesh_object.mesh

    ## Direct problem
    z = np.ones(L)
    dir_problem = eitx.DirectProblem(mesh_object,z)
    V0 = dir_problem.V0     # Discontinuous Garlekin space function
    V = dir_problem.V         # Continuous Garlekin space function

    #"Define gamma as constant = Background"
    bg = settings['bg']
    ivhigh,ivlow = settings['ivhigh'], settings['ivlow']
    p_ivhigh,p_ivlow = settings['p_ivhigh'], settings['p_ivlow']
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
        rad_1 = np.random.uniform(0.15*radius, 0.3*radius, (n_circles+1, m_multi_m))                                # radius of the inclusions
        center_xy = np.random.uniform(-radius*0.5, radius*0.5, size=(2*(n_circles+1), m_multi_m))     # xy-center of the inclusions
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
                    if dist2 < sumrad**2 + 0.05*radius:                             # if the circles touch each other
                        vecpop.append(j)
        rad_1 = np.delete(rad_1,vecpop,1)         # remove the indices when the circles touch each other
        center_xy = np.delete(center_xy,vecpop,1)

        n_circles_samples = len(
            [sample for sample in samples_info if sample['n_circles'] == n_circles+1]
        )
        print("n_circle_samples",n_circles_samples)

        for sample in range(n_samples[n_circles]):
            sample_id = sample + n_circles_samples + 1
            
            print('Generating sample: ', f'{sample+1} with {n_circles+1} circles')
                
            sample_data = {
                "id": sample_id,
                "n_circles": n_circles+1,
                "bg": bg,
                "inclusions": []
            }

            #"Generate Background + Inclusion"
            gamma = eitx.GammaCircle(V0,1,bg,0,0, 0)
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
                ValuesCells1 = eitx.GammaCircle(V0,iv-bg,0.0,rad_1[p,sample],center_xy[2*p,sample], center_xy[2*p+1,sample]).x.array
                gamma_prov = gamma_prov + ValuesCells1
            
            sample_name = os.path.join(settings['datapath'],f"sample_{n_circles+1}_{sample_id}")
            np.save(sample_name,gamma_prov) #Salva array de coeficientes
            sample_data['sample_name'] = sample_name
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
        logging.error(f"DSM fnn datagen failed calling {sys.argv[1]} config file")
        logging.error(e)