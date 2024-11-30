import numpy as np
import pandas as pd
import random as rd
from util import *

# class IRIS():
#     def __init__(self):
#         self.arm = 3
#         self.dim = 12
#         self.data = pd.read_csv('Iris.csv')

#     def step(self):
#         r = rd.randint( 0, 149)
#         if  0 <= r <= 49:
#             target = 0
#         elif 50 <= r <= 99:
#             target = 1
#         else:
#             target = 2
#         random = self.data.loc[r]
#         x = np.zeros(4)
#         for i in range(1,5):
#             x[i-1] = random[i]
#         X_n = []
#         for i in range(3):
#             front = np.zeros((4 * i))
#             back = np.zeros((4 * (2 - i)))
#             new_d = np.concatenate((front, x, back), axis=0)
#             X_n.append(new_d)
#         X_n = np.array(X_n)
#         # print(X_n.shape)
#         reward = np.zeros(self.arm)
#         # print(target)
#         reward[target] = 1
#         return X_n, reward

def sigmoid(x):
    return 1 / 1 + np.exp(-x)

class Env:
    def __init__(self, k, d, arms, action_space, seed, num_visibles=0, func_type="linear", param_dist="gaussian", bias_dist=None, check_specs=False):
        self.k = k
        self.d = d
        self.num_visibles = num_visibles
        self.arms = arms
        self.func_type = func_type
        self.bias_dist = bias_dist
        self.param_dist = param_dist
        
        Z = feature_sampler(dimension=k, feat_dist="gaussian", size=action_space, 
                            disjoint=True, cov_dist=None, 
                            bound=1, bound_method="scaling", 
                            uniform_rng=None, random_state=seed)
        A = mapping_generator(latent_dim=k, obs_dim=d, distribution="uniform", 
                              lower_bound=None, upper_bound=1, 
                              uniform_rng=None, random_state=seed+1)
        
        if self.num_visibles != 0:
            assert self.bias_dist is not None
            Z = Z[:, :self.num_visibles]  # (M, k) -> (M, m)
            A = A[:, :self.num_visibles]  # (d, k) -> (d, m)
            self.inherent_rewards = param_generator(dimension=arms, distribution=self.bias_dist, 
                                                    disjoint=True, bound=1, uniform_rng=None, random_state=seed)
            self.true_mu = param_generator(dimension=self.num_visibles, distribution=self.param_dist, 
                                           disjoint=True, bound=1, uniform_rng=None, random_state=seed-1)
        else:
            self.inherent_rewards = 0
            self.true_mu = param_generator(dimension=k, distribution=self.param_dist, 
                                           disjoint=True, bound=1, uniform_rng=None, random_state=seed-1)

        np.random.seed(seed)
        self.idx = np.random.choice(np.arange(action_space), size=arms, replace=False)
        self.latent = Z[self.idx, :].copy()
        self.observe = self.latent @ A.T

        if check_specs:
            print(f"Arms : {arms}\tBias : {self.bias_dist}\tParameter : {self.param_dist}\tOriginal seed : {seed}")
            print(f"Number of influential variables : {num_visibles}\tNumber of reward parameters : {self.true_mu.shape[0]}")
            print(f"The maximum norm of the latent features : {np.amax([l2norm(feat) for feat in self.latent]):.4f}")
            print(f"Shape of - Z : {self.latent.shape}\tdecoder : {A.shape}\tX : {self.observe.shape}")
            print(f"Type of inherent_rewards : {type(self.inherent_rewards)}\tReward function : {self.func_type}")


    def step(self):
        inner = self.latent @ self.true_mu + self.inherent_rewards
        if self.func_type == "linear":
            expected_rewards = inner
        elif self.func_type == "glm":
            expected_rewards = sigmoid(inner)
        elif self.func_type == "quadratic":
            expected_rewards = (inner) ** 2
        elif self.func_type == "sin":
            expected_rewards = np.sin(inner)

        optimal_arm = np.argmax(expected_rewards)
        # print(self.observe.shape)
        return self.observe, optimal_arm, expected_rewards
