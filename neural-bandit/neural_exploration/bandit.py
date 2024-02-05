import numpy as np
import itertools
import random
import torch
from .utils import *


class ContextualBandit():
    def __init__(self,
                 T,
                 n_arms,
                 n_latent_features,
                 n_obs_features,
                 h,
                 num_visibles,
                 noise_std=0.1,
                 seed=None,
                 is_partial=False
                 ):
        # if not None, freeze seed for reproducibility
        self.random_state = seed
        self._seed(seed)

        # number of rounds
        self.T = T
        # number of arms
        self.n_arms = n_arms
        # number of features for each arm
        self.num_visibles = num_visibles
        self.n_latent_features = n_latent_features
        self.n_obs_features = n_obs_features
        # average reward function
        # h : R^d -> R
        self.h = h
        
        self.is_partial = is_partial

        # standard deviation of Gaussian reward noise
        self.noise_std = noise_std

        # generate random features
        self.reset()

    @property
    def arms(self):
        """Return [0, …,n_arms-1]
        """
        return range(self.n_arms)

    def reset(self):
        """Generate new features and new rewards.
        """
        self.reset_features()
        self.reset_rewards()

    def reset_features(self):
        """Generate normalized random N(0,1) features.
        """
        # x = np.random.randn(self.T, self.n_arms, self.n_features)
        # x /= np.repeat(np.linalg.norm(x, axis=-1, ord=2), self.n_features).reshape(self.T, self.n_arms, self.n_features)
        
        Z = feature_sampler(dimension=self.n_latent_features, feat_dist="gaussian", size=self.n_arms, disjoint=True, cov_dist=None,
                            bound=1, bound_method="scaling", uniform_rng=None, random_state=self.random_state)
        A = mapping_generator(latent_dim=self.n_latent_features, obs_dim=self.n_obs_features, distribution="uniform", lower_bound=None, 
                              upper_bound=1, uniform_rng=None, random_state=self.random_state)
        
        if self.num_visibles > 0:
            Z = Z[:, :self.num_visibles]
            A = A[:, :self.num_visibles]
        
        X = Z @ A.T
        
        self.latent = np.array([Z for _ in range(self.T)]).reshape(self.T, self.n_arms, self.num_visibles) # (T, K, m)
        self.features = np.array([X for _ in range(self.T)]).reshape(self.T, self.n_arms, self.n_obs_features) # (T, K, d)
        if self.is_partial:
            self.features = np.array([
                    np.concatenate([self.features[t], np.identity(self.n_arms)], axis=1) for t in range(self.T)
                ]).reshape(self.T, self.n_arms, (self.n_obs_features+self.n_arms))
            print(self.features.shape)
            
        
    def reset_rewards(self):
        """Generate rewards for each arm and each round,
        following the reward function h + Gaussian noise.
        """
        inherent_rewards_ = param_generator(dimension=self.n_arms, 
                                           distribution="gaussian",
                                           disjoint=True, bound=1, 
                                           random_state=self.random_state)
        inherent_rewards = np.array(
            [inherent_rewards_ for _ in range(self.T)]
        ).reshape(self.T, self.n_arms)
        self.exp_rewards = np.array(
            [
                self.h(self.latent[t, k]) + inherent_rewards[t, k]
                for t, k in itertools.product(range(self.T), range(self.n_arms))
            ]
        ).reshape(self.T, self.n_arms)
        
        reward_noise = subgaussian_noise(distribution="gaussian", 
                                         size=(self.T*self.n_arms), 
                                         std=self.noise_std, 
                                         random_state=self.random_state).reshape(self.T, self.n_arms)
        
        self.rewards = self.exp_rewards + reward_noise

        # to be used only to compute regret, NOT by the algorithm itself
        self.best_rewards_oracle = np.max(self.exp_rewards, axis=1)
        self.best_actions_oracle = np.argmax(self.exp_rewards, axis=1)

    def _seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)