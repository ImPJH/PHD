import os
import pickle
import json
import numpy as np
from typing import Union
from cfg import get_cfg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from datetime import datetime

cfg = get_cfg()

def generate_uniform(dim:Union[int, tuple], uniform_rng:list=None):
    assert type(dim) == int or type(dim) == tuple, "The type of 'dim' must be either int or tuple."
    
    if uniform_rng is None:
        low, high = -1., 1.
    else:
        assert len(uniform_rng) == 2, "The 'uniform_rng' must contain two elements: low and high."
        low, high = uniform_rng
        
    if type(dim) == int:
        size = dim
    else:
        dim1, dim2 = dim
        size = dim1 * dim2
    
    return np.random.uniform(low=low, high=high, size=size).reshape(dim)


def shermanMorrison(V:np.ndarray, x:np.ndarray):
    """
    ${V_t}^{-1} = V_{t-1}^{-1} - \frac{V_{t-1}^{-1}xx^T V_{t-1}^{-1}}{1 + x^T V_{t-1}^{-1} x}$
    V: inverse of old gram matrix, corresponding to $V_{t-1}$.
    x: a new observed context
    return: inverse of new gram matrix
    """
    numerator = np.einsum("ij, j, k, kl -> il", V, x, x, V)
    denominator = (1 + np.einsum("i, ij, j ->", x, V, x))
    return V - (numerator / denominator)


def l1norm(v:np.ndarray):
    v = v.flatten()
    return np.sum(np.absolute(v))


def l2norm(v:np.ndarray):
    v = v.flatten()
    return np.sqrt(np.sum(v ** 2))


def covariance_generator(d:int, distribution:str="gaussian", bound:float=None, uniform_rng:list=None):
    if distribution == "gaussian":
        rnd = np.random.randn(d, d)
    elif distribution == "uniform":
        rnd = generate_uniform(dim=(d, d), uniform_rng=uniform_rng)
    
    ## make a symmetric matrix
    sym = (rnd + rnd.T) / 2
    ## make positive semi-definite and bound its maximum singular value
    mat = sym @ sym.T
    
    if bound is not None:
        mat *= (bound / np.linalg.norm(mat, 2))
    
    return mat


def gram_schmidt(A):
    Q = np.zeros(A.shape)
    for i in range(A.shape[1]):
        # Orthogonalize the vector
        Q[:,i] = A[:,i]
        for j in range(i):
            Q[:,i] -= np.dot(Q[:,j], A[:,i]) * Q[:,j]
        
        # Normalize the vector
        Q[:,i] = Q[:,i] / np.linalg.norm(Q[:,i])
    return Q


def make_diagonal(v:np.ndarray, dim:Union[int, tuple]):
    if type(dim) == int:
        diag = np.zeros((dim, dim))
        rng = dim
    else:
        diag = np.zeros(dim)
        rng = min(dim)
        
    for i in range(rng):
        diag[i, i] = v[i]
    
    return diag


def positive_definite_generator(dimension:int, distribution:str="uniform", uniform_rng:list=None):
    d = dimension

    ## create orthogonal eigenvectors via Gram-Schmidt process
    if distribution == "uniform":
        source = generate_uniform(dim=(d, d), uniform_rng=uniform_rng)
    else:
        source = np.random.randn(d, d)        
    eigvecs = gram_schmidt(source)
    
    ## create a matrix of eigenvalues
    eigvals = generate_uniform(dim=d, uniform_rng=(0, 1))
    eigmat = make_diagonal(np.absolute(eigvals))
    
    ## make the targeted positive definite matrix
    Z = eigvecs @ eigmat @ eigvecs.T
    return Z


def minmax(v:np.ndarray, bound:float=1.):
    min = np.min(v)
    max = np.max(v)
    return ((v - min) / (max - min)) * bound


def left_pseudo_inverse(A:np.ndarray):
    d, k = A.shape
    u, A_sig, v_T = np.linalg.svd(A)
    
    B_sig = np.zeros((k, d))
    for i in range(k):
        B_sig[i, i] = 1 / A_sig[i]
    
    B = v_T.T @ B_sig @ u.T
    
    return B


def rademacher(size:int):
    return 2 * np.random.randint(0, 2, size) - 1


def subgaussian_noise(distribution:str, size:int, random_state:int=None, std:float=None):
    if random_state:
        np.random.seed(random_state)
    
    if distribution == "gaussian":
        if std is None:
            std = 1.
        noise = np.random.normal(loc=0, scale=std, size=size) 
    elif distribution == "uniform":
        if std is None:
            uniform_rng = [-1., 1.]
        else:
            low = -np.sqrt(3) * std
            high = np.sqrt(3) * std
            uniform_rng = [low, high]
        noise = generate_uniform(dim=size, uniform_rng=uniform_rng)
    else:
        std = 1.
        noise = rademacher(size=size)
    return noise


def feature_sampler(dimension:int, feat_dist:str, size:int, disjoint:bool, cov_dist:str=None, bound:float=None, 
                    bound_method:str=None, uniform_rng:list=None, random_state:int=None):
    if random_state:
        np.random.seed(random_state)

    assert feat_dist.lower() in ["gaussian", "uniform"], "Feature distribution must be either 'gaussian' or 'uniform'."
    
    if disjoint:
        if feat_dist.lower() == "gaussian":
            assert uniform_rng is None, f"If the distribution is {feat_dist}, variable range is not required."
            ## gaussian
            feat = np.random.multivariate_normal(mean=np.zeros(dimension), cov=np.identity(dimension), size=size)
        else:
            ## uniform
            feat = generate_uniform(dim=(size, dimension), uniform_rng=uniform_rng)
    else:
        assert cov_dist is not None, f"If 'disjoint' is set to {disjoint}, it is required to specify the distribution to sample the covariance matrix."
        if feat_dist.lower() == "gaussian":
            assert uniform_rng is None, f"If the distribution is {feat_dist}, variable range is not required."
            ## gaussian
            cov = covariance_generator(dimension, distribution=cov_dist)
            feat = np.random.multivariate_normal(mean=np.zeros(dimension), cov=cov, size=size)
        else:
            ## uniform
            feat = generate_uniform(dim=(size, dimension), uniform_rng=uniform_rng)
            
            # Cholesky decomposition
            pd = positive_definite_generator(dimension, distribution=cov_dist)
            L = np.linalg.cholesky(pd)
            for i in range(size):
                feat[i, :] = L @ feat[i, :]
            
    if bound is not None:
        assert bound_method in ["scaling", "clipping"], "Bounding method should either be 'scaling' or 'clipping'."
        ## bound the L2 norm of each row vector
        if bound_method == "scaling":
            norms = [l2norm(feat[i, :]) for i in range(size)]
            max_norm = np.max(norms)
            for i in range(size):
                feat[i, :] *= (bound / max_norm)
        elif bound_method == "clipping":
            for i in range(size):
                norm = l2norm(feat[i, :])
                if norm > bound: 
                    feat[i, :] *= (bound / norm)                
    
    return feat


def mapping_generator(latent_dim:int, obs_dim:int, distribution:str, lower_bound:float=None, upper_bound:float=None, uniform_rng:list=None, random_state:int=None):
    if random_state:
        np.random.seed(random_state)
        
    assert distribution.lower() in ["gaussian", "uniform"], "Feature distribution must be either 'gaussian' or 'uniform'."
    
    if distribution.lower() == "gaussian":
        assert uniform_rng is None, f"If the distribution is {distribution}, variable range is not required."
        mat =  np.random.randn(obs_dim, latent_dim)
    else:
        if uniform_rng is None:
            mat = generate_uniform(dim=(obs_dim, latent_dim), uniform_rng=(-np.sqrt(6/(obs_dim+latent_dim)), np.sqrt(6/(obs_dim+latent_dim))))
        else:
            mat = generate_uniform(dim=(obs_dim, latent_dim), uniform_rng=uniform_rng)
        
    if lower_bound:
        ## constrain the lower bound of the minimum singular value
        u, sig, v_T = np.linalg.svd(mat)
        sig = sig - np.min(sig) + lower_bound
        sig_mat = make_diagonal(sig, dim=mat.shape)
        mat = u @ sig_mat @ v_T
    
    if upper_bound:
        ## constrain the upper bound of the spectral norm
        max_singular = np.linalg.norm(mat, 2)
        mat *= (upper_bound / max_singular)

    return mat


def param_generator(dimension:int, distribution:str, disjoint:bool, bound:float=None, uniform_rng:list=None, random_state:int=None):
    if random_state:
        np.random.seed(random_state)
    assert distribution.lower() in ["gaussian", "uniform"], "Parameter distribution must be either 'gaussian' or 'uniform'."
    
    if disjoint:
        if distribution == "gaussian":
            assert uniform_rng is None, f"If the distribution is {distribution}, variable range is not required."
            param = np.random.randn(dimension)
        else:
            param = generate_uniform(dim=dimension, uniform_rng=uniform_rng)
    else:
        if distribution == "gaussian":
            assert uniform_rng is None, f"If the distribution is {distribution}, variable range is not required."
            cov = covariance_generator(dimension, distribution=distribution)
            param = np.random.multivariate_normal(mean=np.zeros(dimension), cov=cov)
        else:
            # uniform
            param = generate_uniform(dim=dimension, uniform_rng=uniform_rng)
            pd = positive_definite_generator(dimension, distribution=distribution)
            L = np.linalg.cholesky(pd)
            param = L @ param
        
    if (bound is not None) and (l2norm(param) > bound): 
        param *= (bound / l2norm(param))
    
    return param


def save_plot(fig:Figure, path:str, fname:str):
    os.makedirs(path, exist_ok=True)
    fig.savefig(f"{path}/{fname}.png")
    print("Plot is Saved Completely!")
    

def save_result(result:dict, path:str, fname:str, filetype:str):
    assert filetype in ["pickle", "json"]
    os.makedirs(path, exist_ok=True)
    
    if filetype == "pickle":
        with open(f"{path}/{fname}.pkl", "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif filetype == "json":
        with open(f"{path}/{fname}.json", "w") as f:
            json.dump(result, f)
    
    print("Result is Saved Completely!")
