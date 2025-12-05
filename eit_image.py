import numpy as np
import dolfinx
from eit_cont import getGammaCircleLocator

class EIT_Image():
    def __init__(self, dolfin_mesh, mesh_x, mesh_y):
        self.dolfin_mesh = dolfin_mesh
        self.mesh_x = mesh_x
        self.mesh_y = mesh_y
        self.get_mesh_points_cells()
        
    
    def get_mesh_points_cells(self)->tuple[np.ndarray,np.ndarray,list]:
        """
        Computes wich cell each point belongs in a given mesh. Does not return, just sets the object attributes.
        Programmed to work with a fixed mesh at time.

        :param mesh_x: matrix with x components of the mesh
        :type mesh_x: np.ndarray
        :param mesh_y: matrix with y components of the mesh
        :type mesh_x: np.ndarray
        """
        cells = []
        N_grid = self.mesh_x.shape[0]
        
        array_points = np.stack([self.mesh_y.ravel(),self.mesh_x.ravel(),np.zeros(N_grid**2)],axis=1)
        gamma_locator = getGammaCircleLocator(1,0,0)
        circle_points_index = gamma_locator(array_points.T)
        circle_array_points = array_points[circle_points_index]
        bb_tree = dolfinx.geometry.bb_tree(self.dolfin_mesh, self.dolfin_mesh.topology.dim)
        cell_candidates = dolfinx.geometry.compute_collisions_points(
            bb_tree, circle_array_points
        )

        colliding_cells = dolfinx.geometry.compute_colliding_cells(self.dolfin_mesh, cell_candidates, circle_array_points)
        for i, point in enumerate(circle_array_points):
                if len(colliding_cells.links(i)) > 0:
                        cells.append(colliding_cells.links(i)[0])

        self.N_grid = N_grid
        self.circle_array_points = circle_array_points 
        self.circle_points_index = circle_points_index 
        self.cells = cells

    def genGammaImg(self,gamma:dolfinx.fem.Function,bg:float=None,ivhigh:float=None,ivlow:float=None,type="bin")->np.array:
        """
        Compute gamma in given square/rectangular mesh of points.

        
        :param gamma: function gamma to evaluate on the grid
        :param bg: background value
        :param ivhigh: value of conductive inclusions
        :param ivlow: value of resistive inclusions
        :param type: type of image. Options are

            - `bin`: simple indicator, 0 if background, 1 if inclusion;
            - `seg`: segmentation, 0 for bg, 1 for conductive inclusion, -1 for resistive inclusion.
            - `raw`: raw values of gamma
        """
        # points_array = np.array(points_on_proc, dtype=np.float64)

        gamma_values = gamma.eval(self.circle_array_points, self.cells)

        gamma_array = np.full(self.N_grid**2,bg,dtype=float)
        gamma_array[self.circle_points_index] = gamma_values.ravel()
        gamma_matrix = np.reshape(gamma_array,(self.N_grid,self.N_grid))

        if type=="bin":
            img_matrix = np.where(np.isclose(gamma_matrix,bg),0,1)    
        elif type=='seg':
            gamma_bg = np.where(np.isclose(gamma_matrix,bg),0,0)
            gamma_ivhigh = np.where(np.isclose(gamma_matrix,ivhigh),1,0)
            gamma_ivlow = np.where(np.isclose(gamma_matrix,ivlow),-1,0)
        
            img_matrix = gamma_bg + gamma_ivhigh + gamma_ivlow
        elif type=="raw":
             img_matrix = gamma_matrix
        else:
            raise Exception("Invalid type, options are 'bin' or 'seg")
        
        return img_matrix

    def genPotentialImg(self,u,fill_value=0):
        """
        Compute u in given square/rectangular mesh of points.
     
        :param u: function u to evaluate on the grid
        :param fill: values to fill outside the domain.
        """

        cells = []
                        
        u_values = u.eval(self.circle_array_points, self.cells)
        u_array = np.full(self.N_grid**2,fill_value,dtype=np.float64)
        u_array[self.circle_points_index] = u_values.ravel()
        u_matrix = np.reshape(u_array,(self.N_grid,self.N_grid))
        return u_matrix
    
    def get_fnn_matrix(self, u_array):
        
        full_array = np.full(self.N_grid**2,0,dtype=np.float64)
        full_array[self.circle_points_index] = u_array
        u_matrix = np.reshape(full_array,(self.N_grid,self.N_grid))

        return u_matrix