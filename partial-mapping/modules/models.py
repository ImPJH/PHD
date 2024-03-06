import numpy as np
from util import *
from abc import ABC, abstractmethod
from calculate_alpha import *
import scipy
from typing import Callable

## abstract class
class ContextualBandit(ABC):
    @abstractmethod
    def choose(self, x): pass
    
    @abstractmethod
    def update(self, x, r): pass


class LinUCB(ContextualBandit):
    def __init__(self, d:int, lbda:float, delta:float) -> None:
        self.d = d
        self.xty = np.zeros(d)
        self.Vinv = (1 / lbda) * np.identity(d)
        self.theta_hat = np.zeros(d)
        self.delta = delta
        self.t = 0
        
    def choose(self, x:np.ndarray) -> int:
        # x: action set at each round (N, d)
        self.t += 1
        
        ## compute the ridge estimator
        self.theta_hat = self.Vinv @ self.xty
        
        ## compute the ucb scores for each arm
        alpha = linucb_alpha(delta=self.delta) * np.sqrt(np.log(self.t))
        expected = x @ self.theta_hat # (N, ) theta_T @ x_t
        width = np.sqrt(np.einsum("Ni, ij, Nj -> N", x, self.Vinv, x)) # (N, ) widths
        ucb_scores = expected + (alpha * width) # (N, ) ucb score
        
        ## chose the argmax the ucb score
        maximum = np.max(ucb_scores)
        argmax, = np.where(ucb_scores == maximum)        
        return np.random.choice(argmax)
    
    def update(self, x:np.ndarray, r:float) -> None:
        # x: context of the chosen action (d, )
        self.Vinv = shermanMorrison(self.Vinv, x)
        self.xty += (r * x)


class LinTS(ContextualBandit):
    def __init__(self, d:int, lbda:float, horizon:int, reward_std:float, delta:float) -> None:
        self.d = d
        self.Binv = (1 / lbda) * np.identity(d)
        self.xty = np.zeros(d)
        self.theta_hat = np.zeros(d)
        self.horizon = horizon
        self.reward_std = reward_std
        self.delta = delta
        self.t = 0
    
    def choose(self, x:np.ndarray) -> int:
        # x: action set at each round (N, d)
        self.t += 1
        
        ## compute the ridge estimator
        self.theta_hat = self.Binv @ self.xty
        
        ## parameter sampling
        # self.alpha_ = self.alpha * np.sqrt(np.log(self.t))
        # alpha = lints_alpha(d=self.d, horizon=self.horizon, reward_std=self.reward_std, delta=self.delta) * np.sqrt(np.log(self.t))
        alpha = lints_alpha(d=self.d, reward_std=self.reward_std, delta=self.delta)
        tilde_theta = np.random.multivariate_normal(mean=self.theta_hat, cov=(alpha**2) * self.Binv)  # (d, ) random matrix
        
        ## compute estimates and choose the argmax
        expected = x @ tilde_theta  # (N, ) vector
        maximum = np.max(expected)
        argmax, = np.where(expected == maximum)
        return np.random.choice(argmax)
        # return np.argmax(expected)
    
    def update(self, x:np.ndarray, r:float) -> None:
        # x: context of the chosen action (d, )
        # r: reward seen (scalar)
        self.Binv = shermanMorrison(self.Binv, x)
        self.xty += (r * x)    


class OFUL(ContextualBandit):
    def __init__(self, d:int, lbda:float, reward_std:float, context_std:float, horizon:int):
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
        arms = x.shape[0]
        maxnorm = np.amax([l2norm(x[i]) for i in range(arms)])
        if isinstance(self.context_std, float):
            alpha = oful_alpha(maxnorm=maxnorm, horizon=self.horizon, d=self.d, arms=arms, lbda=self.lbda, 
                               reward_std=self.reward_std, context_std=self.context_std)
        else:
            alpha = oful_alpha(maxnorm=maxnorm, horizon=self.horizon, d=self.d, arms=arms, lbda=self.lbda, 
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


class LinUCBPO(ContextualBandit):
    def __init__(self, d:int, arms:int, lbda:float, reward_std:float, delta:float, horizon:int) -> None:
        self.d = d
        self.arms = arms
        self.xty = np.zeros(d+arms)
        self.lbda = lbda
        self.Vinv = (1 / self.lbda) * np.identity(d+arms)
        self.theta_hat = np.zeros(d+arms)
        self.reward_std = reward_std
        self.delta = delta
        self.horizon = horizon
        # self.delta = 1
        self.t = 0
        
    def choose(self, x:np.ndarray) -> int:
        # x: action set at each round (N, d)
        self.t += 1
        ## compute alpha
        # alpha = hop_alpha(lbda=self.lbda, reward_std=self.reward_std, arms=self.arms, delta=self.delta) * np.sqrt(np.log(self.t))
        alpha = hop_alpha(lbda=self.lbda, horizon=self.horizon, reward_std=self.reward_std, arms=self.arms, delta=self.delta)
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

    def update(self, x:np.ndarray, r:float) -> None:
        # x: context of the chosen action (d, )
        self.Vinv = shermanMorrison(self.Vinv, x)
        self.xty += (r * x)
        

class LineGreedy(ContextualBandit):
    ## epsilon greedy linear bandit
    def __init__(self, d, alpha, lbda):
        self.d = d
        self.alpha = alpha
        self.xty = np.zeros(d)
        self.Vinv = (1 / lbda) * np.identity(d)
        self.theta_hat = np.zeros(d)
        self.t = 0

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

    def update(self, x, r):
        # x: context of the chosen action (d, )
        self.Vinv = shermanMorrison(self.Vinv, x)
        self.xty += (r * x)


class GLMUCB(ContextualBandit):
    def __init__(self, d:int, alpha:float, link_function:Callable):
        self.alpha = alpha
        self.t = 0
        self.prev = np.zeros(d)
        self.link = link_function
        self.gram = np.zeros(shape=(d, d))
        self.history = []
        self.rewards = []

    def choose(self, x:np.ndarray):
        self.t += 1

        ## x : (K, d)
        K, d = x.shape
        if scipy.linalg.det(self.gram) < 0.01:
            idx = np.random.choice(np.arange(d))
            return idx

        ## estimate theta_hat according to MLE equation
        theta_hat = scipy.optimize.root(self.__estimate, self.prev).x

        ## calculate the reward
        expected = self.link(x @ theta_hat)
        tuning = self.alpha * np.sqrt(np.log(self.t))
        width = np.sqrt(np.einsum("Ni, ij, Nj -> N", x, scipy.linalg.inv(self.gram), x)) # (N, ) widths
        ucb_scores = expected + (tuning * width)

        ## choose the argmax
        maximum = np.max(ucb_scores)
        argmax, = np.where(ucb_scores == maximum)
        return np.random.choice(argmax)

    def update(self, x:np.ndarray, r:float):
        ## x: (d, ), r: scalar
        self.rewards.append(r)
        self.history.append(x)
        self.gram += np.outer(x, x)

    def __estimate(self, theta:np.ndarray):
        mle_sum = []
        for i in range(len(self.history)):
            context = self.history[i]
            reward = self.rewards[i]
            estimate = self.link(context @ theta)
            mle_sum.append((reward - estimate) * context)
        return np.sum(mle_sum, axis=0)