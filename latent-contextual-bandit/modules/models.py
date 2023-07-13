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
    def __init__(self, d, alpha, lbda=1.):
        self.alpha = alpha
        self.t = 0
        self.xty = np.zeros(d)
        self.Vinv = (1 / lbda) * np.identity(d)
        
    def choose(self, x):
        # x: action set at each round (N, d)
        self.t += 1
        
        ## compute the ridge estimator
        theta_hat = self.Vinv @ self.xty
        
        ## compute the ucb scores for each arm
        expected = x @ theta_hat # (N, ) theta_T @ x_t
        width = np.sqrt(np.einsum("Ni, ij, Nj -> N", x, self.Vinv, x) * np.log(self.t)) # (N, ) widths
        ucb_scores = expected + (self.alpha * width) # (N, ) ucb score
        
        ## chose the argmax the ucb score
        maximum = np.max(ucb_scores)
        argmax, = np.where(ucb_scores == maximum)
        
        return np.random.choice(argmax)
    
    def update(self, x, r):
        # x: context of the chosen action (d, )
        self.Vinv = shermanMorrison(self.Vinv, x)
        self.xty += (r * x)
        

class LineGreedy(LinUCB):
    ## epsilon greedy linear bandit
    def __init__(self, d, alpha, lbda=1, epsilon=0.1):
        super().__init__(d, alpha, lbda)
        self.epsilon = epsilon        
        
    def choose(self, x):
        num_actions, _ = x.shape
        theta_hat = self.Vinv @ self.xty
        expected = x @ theta_hat
        
        ## epsilon greedy
        p = np.random.random()
        if self.epsilon < p:
            ## with 1-epsilon probability, choose the greedy action
            maximum = np.max(expected)
            argmax, = np.where(expected == maximum)
            best_arm = np.random.choice(argmax)
        else:
            ## with epsilon probability, choose a random action
            best_arm = np.random.randint(num_actions)
            
        return best_arm
    
    
class PartialLinUCB(LinUCB):
    def __init__(self, d, num_actions, alpha, lbda=1):
        super().__init__(d, alpha, lbda)
        self.num_actions = num_actions
        self.xty = np.zeros(d+num_actions)
        self.Vinv = (1 / lbda) * np.identity(d+num_actions)
