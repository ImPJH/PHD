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
        self.alpha_ = self.alpha * np.sqrt(np.log(self.t))
        expected = x @ self.theta_hat # (N, ) theta_T @ x_t
        width = np.sqrt(np.einsum("Ni, ij, Nj -> N", x, self.Vinv, x)) # (N, ) widths
        ucb_scores = expected + (self.alpha_ * width) # (N, ) ucb score
        
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


class OFUL(ContextualBandit):
    def __init__(self, d, lbda, reward_std, context_std, delta):
        self.d = d
        self.xty = np.zeros(d)
        self.lbda = lbda
        self.Vinv = (1 / self.lbda) * np.identity(d)
        self.theta_hat = np.zeros(d)
        self.reward_std = reward_std
        self.context_std = context_std
        self.delta = delta
        self.t = 0
        
    def choose(self, x):
        # x: action set at each round (N, d)
        self.t += 1
        
        ## compute alpha
        maxnorm = np.amax([l2norm(x[i]) for i in range(x.shape[0])])
        log_numerator = (maxnorm ** 2) * self.t
        log_denominator = self.d * self.lbda
        log_term = 1 + (log_numerator / log_denominator)
        sqrt_inside1 = self.d * np.log(log_term)
        sqrt_inside2 = 2 * np.log(1 / self.delta)
        alpha_1 = self.reward_std * np.sqrt(sqrt_inside1 + sqrt_inside2)
        alpha_2 = np.sqrt(self.lbda)
        
        sqrt_numerator1 = 2 * self.d * self.t * self.d
        sqrt_numerator2 = np.log(2 * self.d * self.t / self.delta)
        sqrt_inside = (sqrt_numerator1 + sqrt_numerator2) / self.lbda
        alpha_3 = self.context_std * np.sqrt(sqrt_inside)
        # self.alpha = (alpha_1 + alpha_2 + alpha_3) / np.sqrt(self.d)
        self.alpha = (alpha_1 + alpha_2 + alpha_3)
        # print(self.alpha)

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

    def update(self, x, r):
        # x: context of the chosen action (d, )
        self.Vinv = shermanMorrison(self.Vinv, x)
        self.xty += (r * x)


class PALO(ContextualBandit):
    def __init__(self, d, arms, lbda, reward_std, context_std, delta):
        self.d = d
        self.arms = arms
        self.xty = np.zeros(d+arms)
        self.lbda = lbda
        self.Vinv = (1 / self.lbda) * np.identity(d+arms)
        self.theta_hat = np.zeros(d+arms)
        self.reward_std = reward_std
        self.context_std = context_std
        self.delta = delta
        self.t = 0
        
    def choose(self, x):
        # x: action set at each round (N, d)
        self.t += 1
        
        ## compute alpha
        maxnorm = np.amax([l2norm(x[i]) for i in range(x.shape[0])])
        log_numerator = (maxnorm ** 2) * self.t
        log_denominator = (self.d + self.arms) * self.lbda
        log_term = 1 + (log_numerator / log_denominator)
        sqrt_inside1 = (self.d+self.arms) * np.log(log_term)
        sqrt_inside2 = 2 * np.log(1 / self.delta)
        alpha_1 = self.reward_std * np.sqrt(sqrt_inside1 + sqrt_inside2)
        alpha_2 = np.sqrt(self.lbda)
        
        sqrt_numerator1 = 2 * self.d * self.t * (self.d + self.arms)
        sqrt_numerator2 = np.log(2 * self.d * self.t / self.delta)
        sqrt_inside = (sqrt_numerator1 + sqrt_numerator2) / self.lbda
        alpha_3 = self.context_std * np.sqrt(sqrt_inside)
        # self.alpha = (alpha_1 + alpha_2 + alpha_3) / np.sqrt(self.d + self.arms)
        self.alpha = (alpha_1 + alpha_2 + alpha_3)
        # print(self.alpha)

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
        