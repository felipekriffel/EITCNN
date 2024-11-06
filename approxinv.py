#Functions for approximate inverse method
import eitx
import numpy as np

def get_wk_list(A,b,alpha,i,klist):
    """
    Get the i-th row of W_k matrix given k iterations of iterated Tikohnov for solving A*w=e_i
    """
    m,n = A.shape
    e_i = np.zeros(n)
    e_i[i] = 1

    wk = iterated_tikhonov(A.T,e_i,alpha,klist)

    return wk

def iterated_tikhonov(A,b,alpha,klist,x0=None, weight = None):
    """
    Uses iterated Tikhonov for solving Ax=b, saving the x_k terms for index k given in k_list.
    """
    m,n = A.shape
    I = np.eye(n)
    if x0 is None:
        x0 = np.zeros(n)     
    xk = np.copy(x0)
    xk_list = []

    if weight is None:
        adj_A = A.T
    else:
        adj_A = weight @ A.T
    
    kmax = max(klist)

    for k in range(kmax):
        lhs = adj_A @ A + alpha*I
        rhs = adj_A @ b + alpha*xk
        xk = np.linalg.solve(lhs,rhs)
        if k+1 in klist:
            xk_list.append(np.copy(xk))
        k+=1
    
    return xk_list

def get_wk_matrices(A,b,alpha,klist):
    m,n = A.shape

    Wk_list = [np.zeros((n,m)) for k in klist]

    for i in range(n):
        wk_i_list = get_wk_list(A,b,alpha,i,klist)
        
        for Wk,wk_i in zip(Wk_list, wk_i_list):
            Wk[i,:] = wk_i

    return Wk_list

def get_bk_vectors(Wk_list,b):
    bk_list = []
    for Wk in Wk_list:
        bk = Wk@b
        bk_list.append(bk)

    return bk_list

def get_x0_vectors(A,x0,alpha,k_list):
    """
    Computes K_{alpha, n} x0 = alpha^n (A*A + alpha I)^{-n}
    """
    x0k_list = []
    T = A.T@A + alpha*np.eye(A.shape[1])
    for k in k_list:
        
        b = (alpha**k)*x0
        for j in range(k):
            b = np.linalg.solve(T,b)
        x0k_list.append(b)
    return x0k_list



    

def evaluate_method(W: np.array,b: np.array,x0k:np.array,y:np.array,method:str='default',lw_clip:float = 0.1,up_clip:float = 10) -> np.array:
    """
    Evaluates the approximate inverse with informed W, b and y matrices following, where
    Wy - b corresponds to an approximate solution to y = Ax + c. Offers some method variations
    to output the results:

    Methods:
    - `'default'`: returns simply Wy - b;
    - `'clip'`: returns Wy-b cutting off extreme values. Uses `lw_clip` and `up_clip` params as lower and upper bounds;
    - `'log'`: clip values like in `'clip'` method and applys log10 to all values.

    Args:
        W:
            Approximate inverse matrix
        b:
            Approximate inverse bias vector
        x0k:
            Vector K_{alpha,x0} obtained in get_x0_vectors
        y: 
            y data used for solving inverse problem;
        method: 
            Method for evaluation.
        lw_clip:
            Lower clip bound;
        up_clip:
            Upper clip bound.
    """

    gamma_k_array = W@y - b + x0k

    if method=='default':
        return gamma_k_array
    elif method=='clip':
        return np.clip(gamma_k_array,lw_clip,up_clip)
    elif method=='log':
        return np.log10(np.clip(gamma_k_array,lw_clip,up_clip))
    else:
        raise ValueError(f"Method {method} not implemented")


def add_noise(x,noise_level=0.01):
    noise = np.random.uniform(-1,1,x.shape)
    noise = noise/np.linalg.norm(noise)
    x_delta = x + noise_level * np.linalg.norm(x) * noise
    return x_delta

