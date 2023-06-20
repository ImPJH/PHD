import numpy as np

def l2norm(v:np.ndarray):
    return np.sqrt(np.sum(v**2))

# random vector generator - can be used for both features and (true) parameters
def get_random_vector(dimension:int, distribution:str, params:dict, 
                      is_element_bounded:bool=False, is_norm_bounded:bool=True, 
                      bound:float=None, is_weighted_norm:bool=False, **kwargs):
    if distribution == "gaussian":
        vector = np.random.normal(loc=params['loc'], scale=params['scale'], size=dimension)
    elif distribution == "uniform":
        vector = np.random.uniform(low=params['low'], high=params['high'], size=dimension)
    elif distribution == "bernoulli":
        vector = np.random.binomial(n=1, p=params['p'], size=dimension)
    elif distribution == "logistic":
        vector = np.random.logistic(loc=params['loc'], scale=params['scale'], size=dimension)

    # bound
    if is_norm_bounded:
        assert bound is not None
        assert not is_element_bounded
        if is_weighted_norm:
            weight = kwargs['weight']
            norm = vector.T @ weight @ vector
        else:
            norm = l2norm(vector)
        vector *= (bound / norm)
    elif is_element_bounded:
        assert bound is not None
        assert not is_norm_bounded
        maximum = np.max(np.absolute(vector))
        vector *= (bound / maximum)
    
    return vector

# noise generator
def rademacher(size:int):
    """
    Generate Rademacher random variables.

    Args:
    size (int): Number of random variables to generate.

    Returns:
    numpy.ndarray: An array of Rademacher random variables.
    """
    return 2 * np.random.randint(0, 2, size) - 1

def subgaussian_noise(distribution:str, size:int):
    """
    distribution (str): the distribution to sample a sub-Gaussian noise
    size (int): The number of total rounds (T)
    """
    if distribution == "gaussian":
        return np.random.normal(loc=0, scale=1, size=size)
    elif distribution == "uniform":
        return np.random.uniform(low=-1, high=1, size=size)
    return rademacher(size=size)

# reward function
def get_reward(x:np.ndarray, param:np.ndarray, noise:float, bound:float=None):
    """
    x: feature vector
    param: aprameter vector
    noise: sub-Gaussian random variable
    """
    expected_reward = (x @ param)
    if bound:
        if expected_reward > bound:
            expected_reward = bound
    return expected_reward + noise

# generate arms
def generate_arms(distribution:str, params:dict, dimension:int, num_arms:int, 
                  is_norm_bounded:bool=True, bound:float=1.):
    """
    distribution: distribution to sample arms
    d: the dimension of each arm
    k: the number of arms
    """
    arms = np.zeros(shape=(dimension, num_arms))
    for i in range(num_arms):
        ## for each arm
        arm_feature = get_random_vector(dimension=dimension, 
                                        distribution=distribution, 
                                        params=params,
                                        is_norm_bounded=is_norm_bounded, 
                                        bound=bound)
        arms[:, i] = arm_feature
    return arms
