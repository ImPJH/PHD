import numpy as np
from abc import ABC, abstractmethod
from cfg import get_cfg
from utils import *

cfg = get_cfg()

class ContextualBandit(ABC):  
    @abstractmethod
    def choose(self, x): pass
    
    @abstractmethod
    def update(self, x, a, r): pass


## Disjoint LinUCB
class LinUCB(ContextualBandit):
    def __init__(self, arms, alpha=1.):
        """
        arms: (d+k) x k matrix, k - the number of actions, d - features of each action
        alpha: hyper-parameter that determines degree of exploration <- given in the main file
        """
        self.arms = arms
        self.dim, self.n_arms = arms.shape
        self.alpha = alpha
        self.As = [np.identity(self.dim) for _ in range(self.n_arms)]         # A = d x d
        self.bs = [np.zeros(shape=(self.dim, 1)) for _ in range(self.n_arms)] # b = (d, 1) array
        self.ps = np.zeros(shape=self.n_arms)
           
    def choose(self):
        """
        return: index of arm which yields the highest payoff
        """
        for i in range(self.n_arms):
            arm_feat = self.arms[:, i].reshape(-1, 1)
            A_a, b_a = self.As[i], self.bs[i]
            A_a_inv = np.linalg.inv(A_a)
            theta_a = A_a_inv @ b_a
            p = (theta_a.T @ arm_feat) + (self.alpha * np.sqrt(arm_feat.T@A_a_inv@arm_feat))
            self.ps[i] = p.item()

        max_p = np.max(self.ps)
        tie = np.where(self.ps == max_p)[0]
        return np.random.choice(tie)

    def update(self, a, r):
        """
        x: (d-l) shaped 1d-array - user feature
        a: index of the chosen arm - return value of choose function
        r: reward yielded from the chosen arm
        """
        chosen_arm_feat = self.arms[:, a].reshape(-1, 1)
        oldA, oldb = self.As[a], self.bs[a]
        newA = oldA + np.outer(chosen_arm_feat, chosen_arm_feat)
        newb = oldb + (r * chosen_arm_feat)
        self.As[a], self.bs[a] = newA, newb

## OFUL
class OFUL(ContextualBandit):
    def __init__(self, arms, delta, lam, sigma, bound, distribution, candidates=10):
        """
        arms: (d+k) x k matrix, k - the number of actions, d+k - features of each action
        delta: probability bound of the confidence interval
        lam: regularization coefficient for ridge regression
        sigma: variance proxy of the noise term
        bound: bound of the expected reward
        distribution: distribution dictionary
        """
        self.arms = arms
        self.dim, self.n_arms = arms.shape
        self.X = []
        self.y = [] # b = (d, 1) array
        self.ps = np.zeros(shape=self.n_arms)
        
        self.delta = delta
        self.lam = lam
        self.sigma = sigma
        self.bound = bound
        self.V = self.lam * np.identity(n=(self.dim))
        self.theta_hat = np.zeros(shape=(self.dim, 1))
        self.C = (self.sigma * np.sqrt((self.dim) * np.log(1/self.delta))) + (np.sqrt(self.lam * self.n_arms) * self.bound)
        self.distribution = distribution
        self.candidates = candidates
        
        self.t = 0
        
    def choose(self):
        V_inv = np.linalg.inv(self.V)
        for i in range(self.n_arms):
            arm_feat = self.arms[:, i].reshape(-1, 1)
            ucb = (np.sqrt(self.theta_hat.T @ V_inv @ self.theta_hat) + self.C).item()
            params = [get_random_vector(dimension=self.dim,
                                        distribution=self.distribution['distribution'],
                                        params=self.distribution['params'],
                                        is_norm_bounded=True,
                                        bound=ucb,
                                        is_weighted_norm=True, weight=V_inv).reshape(-1, 1) for _ in range(self.candidates)]
            itm_max = (-1) * np.inf
            for param in params:
                itm_reward = arm_feat.T @ param
                if itm_max < itm_reward:
                    itm_max = itm_reward
            self.ps[i] = itm_max

        max_p = np.max(self.ps)
        tie = np.where(self.ps == max_p)[0]
        self.t += 1
        return np.random.choice(tie)
    
    def update(self, a, r):
        chosen_arm_feat = self.arms[:, a].reshape(-1, 1)
        self.X.append(chosen_arm_feat.T)
        self.y.append(r)
        
        X_compute = np.array(self.X).squeeze(axis=1)
        y_compute = np.array(self.y)
        self.V += (X_compute.T @ X_compute)
        self.theta_hat = np.linalg.inv(self.V) @ X_compute.T @ y_compute
        self.C = (self.sigma * np.sqrt((self.dim) * np.log((1+(4*self.t/self.lam))/self.delta))) + (np.sqrt(self.lam * self.n_arms) * self.bound)
        