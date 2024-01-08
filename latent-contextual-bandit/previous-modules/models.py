import numpy as np
from util import *
from abc import ABC, abstractmethod

## abstract class
class ContextualBandit(ABC):
    @abstractmethod
    def choose(self, x): pass
    
    @abstractmethod
    def update(self, x, r): pass
    
    
class LinUCB(ContextualBandit):
    def __init__(self, d, alpha, lbda):
        self.alpha = alpha
        self.t = 0
        self.xty = np.zeros(d)
        self.Vinv = (1 / lbda) * np.identity(d)
        self.theta_hat = np.zeros(d)
        
    def choose(self, x):
        # x: action set at each round (N, d)
        self.t += 1
        
        ## compute the ridge estimator
        self.theta_hat = self.Vinv @ self.xty
        
        ## compute the ucb scores for each arm
        self.alpha *= np.sqrt(np.log(self.t))
        expected = x @ self.theta_hat # (N, ) theta_T @ x_t
        width = np.sqrt(np.einsum("Ni, ij, Nj -> N", x, self.Vinv, x)) # (N, ) widths
        ucb_scores = expected + (self.alpha * width) # (N, ) ucb score
        
        ## choose the argmax the ucb score
        maximum = np.max(ucb_scores)
        argmax, = np.where(ucb_scores == maximum)
        return np.random.choice(argmax)
        # return np.argmax(ucb_scores)
    
    def update(self, x, r):
        # x: context of the chosen action (d, )
        self.Vinv = shermanMorrison(self.Vinv, x)
        self.xty += (r * x)
        

class LineGreedy(LinUCB):
    ## epsilon greedy linear bandit
    def __init__(self, d, alpha, lbda):
        super().__init__(d, alpha, lbda)
        
    def choose(self, x):
        self.t += 1
        epsilon = (1./self.t) * self.alpha
        
        arms, _ = x.shape
        theta_hat = self.Vinv @ self.xty
        expected = x @ theta_hat
        
        ## epsilon greedy
        p = np.random.random()
        if epsilon < p:
            ## with 1-epsilon probability, choose the greedy action
            maximum = np.max(expected)
            argmax, = np.where(expected == maximum)
            best_arm = np.random.choice(argmax)
        else:
            ## with epsilon probability, choose a random action
            best_arm = np.random.randint(arms)    
        return best_arm


class PALO(LinUCB):
    def __init__(self, d, arms, alpha, lbda):
        super().__init__(d, alpha, lbda)
        self.arms = arms
        self.xty = np.zeros(d+arms)
        self.Vinv = (1 / lbda) * np.identity(d+arms)
        self.theta_hat = np.zeros(d+arms)
        
    def choose(self, x):
        # x: action set at each round (N, d)
        self.t += 1

        ## compute the ridge estimator
        self.theta_hat = self.Vinv @ self.xty
        
        ## compute the ucb scores for each arm
        expected = x @ self.theta_hat # (N, ) theta_T @ x_t
        width = np.sqrt(np.einsum("Ni, ij, Nj -> N", x, self.Vinv, x)) # (N, ) widths
        ucb_scores = expected + (self.alpha * width) # (N, ) ucb score
        
        ## choose the argmax the ucb score
        maximum = np.max(ucb_scores)
        argmax, = np.where(ucb_scores == maximum)
        return np.random.choice(argmax)
        

class LinTS(ContextualBandit):
    def __init__(self, d, alpha, lbda):
        self.alpha = alpha
        self.Binv = (1 / lbda) * np.identity(d)
        self.xty = np.zeros(d)
        self.theta_hat = np.zeros(d)
    
    def choose(self, x):
        # x: action set at each round (N, d)
        ## compute the ridge estimator
        self.theta_hat = self.Binv @ self.xty
        
        ## parameter sampling
        tilde_theta = np.random.multivariate_normal(mean=self.theta_hat, cov=(self.alpha**2) * self.Binv)  # (d, ) random matrix
        
        ## compute estimates and choose the argmax
        expected = x @ tilde_theta  # (N, ) vector
        maximum = np.max(expected)
        argmax, = np.where(expected == maximum)
        return np.random.choice(argmax)
        # return np.argmax(expected)
    
    def update(self, x, r):
        # x: context of the chosen action (d, )
        # r: reward seen (scalar)
        self.Binv = shermanMorrison(self.Binv, x)
        self.xty += (r * x)
        