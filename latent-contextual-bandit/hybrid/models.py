import numpy as np
from util import *
from abc import ABC, abstractmethod

############################################# ARMS #############################################
class BernoulliArm:
    def __init__(self, p):
        self.mu = p
    
    def draw(self):
        if np.random.random() > self.mu:
            return 0.0
        else:
            return 1.0
        

class GaussianArm:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        
    def draw(self):
        return np.random.normal(loc=self.mu, scale=self.sigma)
################################################################################################    

###################################### Multi-Armed Bandit ######################################
class UCB:
    def __init__(self, arms, sigma, delta=0.1, alpha=1.0):
        self.arms = arms  # number of arms
        self.t = 0  # total play count
        self.sigma = sigma
        self.n = np.zeros(arms, dtype=int)  # play count for each arm
        self.mean_reward = np.zeros(arms, dtype=float)  # mean reward for each arm
        self.delta = delta
        self.alpha = alpha
        self.ucb_values = np.zeros(self.arms, dtype=float)

    def choose(self):
        # Increment the playing round
        self.t += 1
        
        # Play each arm once first
        for i in range(self.arms):
            if self.n[i] == 0:
                return i

        # ucb_values = np.zeros(self.arms, dtype=float)
        for arm in range(self.arms):
            bonus_first = (1 + self.n[arm]) / (self.n[arm] ** 2)
            bonus_second_numerator = self.arms * np.sqrt(1 + self.n[arm])
            bonus_second = 1 + (2 * np.log(bonus_second_numerator / self.delta))
            bonus = self.sigma * np.sqrt(bonus_first * bonus_second)
            self.ucb_values[arm] = self.mean_reward[arm] + (bonus * self.alpha)
            
        max_value = np.amax(self.ucb_values)
        argmaxes, = np.where(self.ucb_values == max_value)

        return np.random.choice(argmaxes)

    def update(self, chosen_arm, reward):
        self.n[chosen_arm] += 1
        n = self.n[chosen_arm]

        # Update the mean reward for the chosen arm
        value = self.mean_reward[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.mean_reward[chosen_arm] = new_value
################################################################################################
    
####################################### Contextual Bandit ######################################
class LinUCB:
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
        expected = x @ self.theta_hat # (N, ) theta_T @ x_t
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
    
    
class PartialLinUCB(LinUCB):
    def __init__(self, d, arms, alpha, lbda):
        super().__init__(d, alpha, lbda)
        self.arms = arms
        self.xty = np.zeros(d+arms)
        self.Vinv = (1 / lbda) * np.identity(d+arms)
        self.theta_hat = np.zeros(d+arms)
################################################################################################