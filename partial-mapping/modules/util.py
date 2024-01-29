import os
import pickle
import json
import numpy as np
from typing import Union, List, Tuple, Dict
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from datetime import datetime


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


def covariance_generator(d:int, independent:bool, distribution:str=None, uniform_rng:list=None, 
                         variances:Union[list, np.ndarray]=None):
    if independent:
        if variances is None:
            assert distribution is not None and distribution.lower() in ["gaussian", "uniform"], "If the variances are not given, you need to pass the distribution to sample them."
            ## then variances are sampled randomly
            if distribution == "gaussian":
                variances = (np.random.randn(d)) ** 2
            else:
                variances = (generate_uniform(dim=d, uniform_rng=uniform_rng)) ** 2

        mat = np.zeros(shape=(d, d))
        for i in range(d):
            mat[i, i] = variances[i]
    
    else:
        assert distribution is not None and distribution.lower() in ["gaussian", "uniform"], f"If independent is {independent}, you need to pass the distribution to sample them."
        if distribution == "gaussian":
            rnd = np.random.randn(d, d)
        elif distribution == "uniform":
            rnd = generate_uniform(dim=(d, d), uniform_rng=uniform_rng)
        
        ## make a symmetric matrix
        sym = (rnd + rnd.T) / 2
        ## make positive semi-definite and bound its maximum singular value
        mat = sym @ sym.T
        if variances is not None:
            for i in range(d):
                mat[i, i] = variances[i]
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
    for i in range(min(d, k)):
        B_sig[i, i] = 1 / A_sig[i]
    B = v_T.T @ B_sig @ u.T
    return B


def rademacher(size:int):
    return 2 * np.random.randint(0, 2, size) - 1


def subgaussian_noise(distribution:str, size:int, std:float=None, random_state:int=None):
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


def bounding(type:str, v:np.ndarray, bound:float, method:str=None):
    if type == "param":
        if l2norm(v) > bound:
            v *= (bound / l2norm(v))
    elif type == "feature":
        assert method in ["scaling", "clipping"], f"If you're trying to bound {type}, the method should not be None."
        if method == "scaling":
            maxnorm = np.max([l2norm(item) for item in v])
            v *= (bound / maxnorm)
        else:
            for i in range(v.shape[0]):
                if l2norm(v[i]) > bound:
                    v[i] *= (bound / l2norm(v[i]))
    elif type == "mapping":
        assert method in ["lower", "upper"], f"If you're trying to bound {type}, you need to specify the lower or the upper bound."
        if method == "lower":
            ## constrain the lower bound of the minimum singular value
            u, sig, v_T = np.linalg.svd(v)
            sig = sig - np.min(sig) + bound
            sig_v = make_diagonal(sig, dim=v.shape)
            v = u @ sig_v @ v_T
        
        if method == "upper":
            ## constrain the upper bound of the spectral norm
            v *= (bound / np.linalg.norm(v, 2))
    return v


def feature_sampler(dimension:int, feat_dist:str, size:int, disjoint:bool, cov_dist:str=None, bound:float=None, 
                    bound_method:str=None, uniform_rng:list=None, random_state:int=None):
    assert feat_dist.lower() in ["gaussian", "uniform"], "Feature distribution must be either 'gaussian' or 'uniform'."
    if random_state:
        np.random.seed(random_state)
    
    if disjoint:
        if feat_dist.lower() == "gaussian":
            assert uniform_rng is None, f"If the distribution is {feat_dist}, variable range is not required."
            ## gaussian
            variances = np.ones(dimension)
            cov = covariance_generator(d=dimension, independent=True, variances=variances)
            feat = np.random.multivariate_normal(mean=np.zeros(dimension), cov=cov, size=size)
        else:
            ## uniform
            feat = generate_uniform(dim=(size, dimension), uniform_rng=uniform_rng)
    else:
        assert cov_dist is not None, f"If 'disjoint' is set to {disjoint}, it is required to specify the distribution to sample the covariance matrix."
        if feat_dist.lower() == "gaussian":
            ## gaussian
            cov = covariance_generator(d=dimension, independent=False, distribution=cov_dist)
            feat = np.random.multivariate_normal(mean=np.zeros(dimension), cov=cov, size=size)
        else:
            ## uniform
            feat = generate_uniform(dim=(size, dimension), uniform_rng=uniform_rng)
            
            # Cholesky decomposition
            pd = positive_definite_generator(dimension=dimension, distribution=cov_dist)
            L = np.linalg.cholesky(pd)
            for i in range(size):
                feat[i, :] = L @ feat[i, :]
            
    if bound is not None:
        assert bound_method in ["scaling", "clipping"], "Bounding method should either be 'scaling' or 'clipping'."
        feat = bounding(type="feature", v=feat, bound=bound, method=bound_method)
    return feat


def mapping_generator(latent_dim:int, obs_dim:int, distribution:str, lower_bound:float=None, upper_bound:float=None, uniform_rng:list=None, random_state:int=None):
    assert distribution.lower() in ["gaussian", "uniform"], "Feature distribution must be either 'gaussian' or 'uniform'."
    if random_state:
        np.random.seed(random_state)
    
    if distribution.lower() == "gaussian":
        assert uniform_rng is None, f"If the distribution is {distribution}, variable range is not required."
        mat = np.random.randn(obs_dim, latent_dim)
    else:
        if uniform_rng is None:
            mat = generate_uniform(dim=(obs_dim, latent_dim), uniform_rng=[-np.sqrt(2/latent_dim), np.sqrt(2/latent_dim)])
        else:
            mat = generate_uniform(dim=(obs_dim, latent_dim), uniform_rng=uniform_rng)
        
    if lower_bound is not None:
        ## constrain the lower bound of the spectral norm
        mat = bounding(type="mapping", v=mat, bound=lower_bound, method="lower")
    
    if upper_bound is not None:
        ## constrain the upper bound of the spectral norm
        mat = bounding(type="mapping", v=mat, bound=upper_bound, method="upper")
    return mat


def param_generator(dimension:int, distribution:str, disjoint:bool, bound:float=None, uniform_rng:list=None, random_state:int=None):
    assert distribution.lower() in ["gaussian", "uniform"], "Parameter distribution must be either 'gaussian' or 'uniform'."
    if random_state:
        np.random.seed(random_state)
    
    if disjoint:
        if distribution == "gaussian":
            assert uniform_rng is None, f"If the distribution is {distribution}, variable range is not required."
            param = np.random.randn(dimension)
        else:
            param = generate_uniform(dim=dimension, uniform_rng=uniform_rng)
    else:
        if distribution == "gaussian":
            cov = covariance_generator(dimension, distribution=distribution)
            param = np.random.multivariate_normal(mean=np.zeros(dimension), cov=cov)
        else:
            # uniform
            param = generate_uniform(dim=dimension, uniform_rng=uniform_rng)
            pd = positive_definite_generator(dimension, distribution=distribution)
            L = np.linalg.cholesky(pd)
            param = L @ param
        
    if bound is not None:
        param = bounding(type="param", v=param, bound=bound)
    return param


def save_plot(fig:Figure, path:str, fname:str, extension:str="pdf"):
    os.makedirs(path, exist_ok=True)
    fig.savefig(f"{path}/{fname}.{extension}")
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


def tau_min(delta:float, dimension:int, horizon:int, rho_min:float):
    first = (16 / (rho_min ** 2)) + (8 / (3*rho_min))
    second = np.log(2*dimension*horizon / delta)
    return first * second


def gamma_delta(num_arms:int, horizon:int, round:int, dimension:int, sigma:float, delta:float):
    first = (10/3) * (2 + (sigma*np.sqrt(1 + (2*np.log(2*num_arms*horizon / delta)))))
    second = np.log(2*dimension*horizon/delta)
    third = np.sqrt((round*np.log(2*dimension*horizon/delta)) + (np.log(2*dimension*horizon/delta)**2))
    return first * (second + third)


def M_delta(round:int, dimension:int, delta:float, sigma:float):
    inside_first = (dimension/2) * np.log(1 + (round/dimension))
    inside_second = np.log(1/delta)
    inside = 2 * (sigma**2) * (inside_first + inside_second)
    return np.sqrt(inside) + 1


def kappa_delta(num_arms:int, horizon:int, round:int, dimension:int, sigma:float, delta:float, rho_min:float):
    if round > num_arms and round <= num_arms + tau_min(delta, dimension, horizon, rho_min):
        return M_delta(round, dimension, delta, sigma) + gamma_delta(num_arms, horizon, round, dimension, sigma, delta)
    first_denominator = np.sqrt(1 + (rho_min*(round-num_arms)/2))
    second_denominator = 1 + (rho_min*(round-num_arms)/2)
    return (M_delta(round, dimension, delta, sigma) / first_denominator) + (gamma_delta(num_arms, horizon, round, dimension, sigma, delta) / second_denominator)


def Q_delta(num_arms:int, horizon:int, round:int, dimension:int, sigma:float, delta:float, rho_min:float, rho_max:float):
    outside = 16 * np.sqrt(np.log(num_arms) * rho_max)
    
    kappa_val = 0
    kappa_val_sq = 0
    for s in range(num_arms, round):
        kappa_val += kappa_delta(num_arms, horizon, s, dimension, sigma, delta, rho_min)
        kappa_val_sq += (kappa_delta(num_arms, horizon, s, dimension, sigma, delta, rho_min) ** 2)
        
    inside_first = np.sqrt(kappa_val_sq * np.log(1/delta))
    return (outside * (inside_first + kappa_val)) + (3*np.log(1/delta))


def W_delta(num_arms:int, horizon:int, round:int, dimension:int, sigma:float, delta:float, rho_min:float, rho_max:float):
    first = 2 * Q_delta(num_arms, horizon, round, dimension, sigma, delta, rho_min, rho_max)
    second = sigma * np.sqrt(((1+round)/2)*np.log(1/delta))
    third = (2*sigma + 3) * np.sqrt(1 + 2*np.log(num_arms * np.sqrt(round) / delta))
    fourth = np.sqrt(num_arms * round)
    return first + second + (third * fourth)

