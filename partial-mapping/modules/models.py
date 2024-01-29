import numpy as np
from util import *
from abc import ABC, abstractmethod
from calculate_alpha import *
import scipy.optimize

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


class HOPlinear(ContextualBandit):
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


# class UCBGLM(ContextualBandit):
#     def __init__(self, d, horizon, reward_std, lbda, link_func, delta):
#         self.d = d
#         self.link = link_func
#         self.reward_std = reward_std
#         self.t = 0
#         self.horizon = horizon
#         self.delta = delta
#         self.prev_theta = np.zeros(d)
#         self.Vinv = (1 / lbda) * np.identity(d)
#         self.rewards = []  # contains the chosen reward
#         self.contexts = [] # contains the chosen context
    
#     def _calc_mle(self, theta):
#         to_sum = []
#         for tau in range(self.t):
#             diff = self.rewards[tau] - self.link(self.contexts[tau] @ theta)
#             to_sum.append(diff * self.contexts[tau])
#         return np.sum(to_sum, 0)
    
#     def _numerical_diff(self, f, x, theta):
#         # f: function
#         # x: K, d
#         K, d = x.shape
#         epsilon = 1e-4
#         kappas = np.zeros(K)
#         for k in range(K):
#             arg = x[k, :] @ theta
#             quotient = (f(arg+epsilon) - f(arg-epsilon)) / (2*epsilon)
#             kappas[k] = quotient
#         return np.amin(kappas)
    
#     def choose(self, x):
#         K, d = x.shape  # K: number of arms, d: dimension
#         self.t += 1
#         if self.t <= K:
#             return self.t-1
        
#         ## compute the estimator
#         theta_hat = scipy.optimize.root(self._calc_mle, self.prev_theta)
        
#         ## compute alpha
#         kappa = self._numerical_diff(self.link, x, theta_hat)
#         alpha = (3*self.reward_std / kappa) * np.sqrt(2 * np.log(self.horizon*K / self.delta))
#         expected = x @ theta_hat
#         width = np.sqrt(np.einsum("Ni, ij, Nj -> N", x, self.Vinv, x))
#         ucb_scores = expected + (alpha * width)
        
#         ## chose the argmax the ucb score
#         maximum = np.max(ucb_scores)
#         argmax, = np.where(ucb_scores == maximum)
        
#         ## update theta hat
#         self.prev_theta = theta_hat
#         return np.random.choice(argmax)
    
#     def update(self, x, r):
#         # x: context of the chosen action (d, )
#         self.Vinv = shermanMorrison(self.Vinv, x)
#         self.rewards.append(r)
#         self.contexts.append(x) 