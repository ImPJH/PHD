import numpy as np
from util import *
from abc import ABC, abstractmethod
from calculate_alpha_v2 import oful_alpha

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
        self.alpha_ = self.alpha * np.sqrt(np.log(self.t))
        expected = x @ self.theta_hat # (N, ) theta_T @ x_t
        width = np.sqrt(np.einsum("Ni, ij, Nj -> N", x, self.Vinv, x)) # (N, ) widths
        ucb_scores = expected + (self.alpha_ * width) # (N, ) ucb score
        
        ## chose the argmax the ucb score
        maximum = np.max(ucb_scores)
        argmax, = np.where(ucb_scores == maximum)        
        return np.random.choice(argmax)
    
    def update(self, x, r):
        # x: context of the chosen action (d, )
        self.Vinv = shermanMorrison(self.Vinv, x)
        self.xty += (r * x)


class LinTS(ContextualBandit):
    def __init__(self, d, alpha, lbda):
        self.alpha = alpha
        self.Binv = (1 / lbda) * np.identity(d)
        self.xty = np.zeros(d)
        self.theta_hat = np.zeros(d)
        self.t = 0
    
    def choose(self, x):
        # x: action set at each round (N, d)
        self.t += 1
        
        ## compute the ridge estimator
        self.theta_hat = self.Binv @ self.xty
        
        ## parameter sampling
        self.alpha_ = self.alpha * np.sqrt(np.log(self.t))
        tilde_theta = np.random.multivariate_normal(mean=self.theta_hat, cov=(self.alpha_**2) * self.Binv)  # (d, ) random matrix
        
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


class OFUL(ContextualBandit):
    def __init__(self, d, lbda, reward_std, context_std, horizon):
        self.d = d
        self.xty = np.zeros(d)
        self.lbda = lbda
        self.Vinv = (1 / self.lbda) * np.identity(d)
        self.theta_hat = np.zeros(d)
        self.reward_std = reward_std
        self.context_std = context_std
        self.horizon = horizon
        
    def choose(self, x):
        # x: action set at each round (N, d)
        ## compute alpha
        maxnorm = np.amax([l2norm(x[i]) for i in range(x.shape[0])])
        if isinstance(self.context_std, float):
            alpha = oful_alpha(maxnorm=maxnorm, horizon=self.horizon, d=self.d, arms=0, lbda=self.lbda, 
                               reward_std=self.reward_std, context_std=self.context_std)
        else:
            alpha = oful_alpha(maxnorm=maxnorm, horizon=self.horizon, d=self.d, arms=0, lbda=self.lbda, 
                               reward_std=self.reward_std, context_std=self.context_std[self.t-1])

        ## compute the ridge estimator
        self.theta_hat = self.Vinv @ self.xty
        
        ## compute the ucb scores for each arm
        expected = x @ self.theta_hat # (N, ) theta_T @ x_t
        width = np.sqrt(np.einsum("Ni, ij, Nj -> N", x, self.Vinv, x)) # (N, ) widths
        ucb_scores = expected + (alpha * width) # (N, ) ucb score
        
        ## choose the argmax the ucb score
        maximum = np.max(ucb_scores)
        argmax, = np.where(ucb_scores == maximum)
        return np.random.choice(argmax)

    def update(self, x, r):
        # x: context of the chosen action (d, )
        self.Vinv = shermanMorrison(self.Vinv, x)
        self.xty += (r * x)


class POLO(ContextualBandit):
    def __init__(self, d, arms, lbda, reward_std, context_std, horizon):
        self.d = d
        self.arms = arms
        self.xty = np.zeros(d+arms)
        self.lbda = lbda
        self.Vinv = (1 / self.lbda) * np.identity(d+arms)
        self.theta_hat = np.zeros(d+arms)
        self.reward_std = reward_std
        self.context_std = context_std
        self.horizon = horizon
        self.t = 0
        
    def choose(self, x):
        # x: action set at each round (N, d)
        self.t += 1
        ## compute alpha
        maxnorm = np.amax([l2norm(x[i]) for i in range(self.arms)])
        if isinstance(self.context_std, float):
            alpha = oful_alpha(maxnorm=maxnorm, horizon=self.horizon, d=self.d, arms=self.arms, lbda=self.lbda, 
                               reward_std=self.reward_std, context_std=self.context_std)
        else:
            alpha = oful_alpha(maxnorm=maxnorm, horizon=self.horizon, d=self.d, arms=self.arms, lbda=self.lbda, 
                               reward_std=self.reward_std, context_std=self.context_std[self.t-1])

        ## compute the ridge estimator
        self.theta_hat = self.Vinv @ self.xty
        
        ## compute the ucb scores for each arm
        expected = x @ self.theta_hat # (N, ) theta_T @ x_t
        width = np.sqrt(np.einsum("Ni, ij, Nj -> N", x, self.Vinv, x)) # (N, ) widths
        ucb_scores = expected + (alpha * width) # (N, ) ucb score
        
        ## choose the argmax the ucb score
        maximum = np.max(ucb_scores)
        argmax, = np.where(ucb_scores == maximum)
        return np.random.choice(argmax)

    def update(self, x, r):
        # x: context of the chosen action (d, )
        self.Vinv = shermanMorrison(self.Vinv, x)
        self.xty += (r * x)
        

class LineGreedy(LinUCB):
    ## epsilon greedy linear bandit
    def __init__(self, d, alpha, lbda, epsilon):
        super().__init__(d, alpha, lbda)
        self.epsilon = epsilon
        
    def choose(self, x):
        arms, _ = x.shape
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
            best_arm = np.random.randint(arms)    
        return best_arm
