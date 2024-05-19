import numpy as np
from util import *
from abc import ABC, abstractmethod
from calculate_alpha import *
import scipy
from sklearn.linear_model import Lasso, LinearRegression
import statsmodels.api as sm
from typing import Callable

#############################################################################
############################ Multi-Armed Bandits ############################
#############################################################################
class MAB(ABC):
    @abstractmethod
    def choose(self): pass
    
    @abstractmethod
    def update(self, a, r): pass

class eGreedyMAB(MAB):
    def __init__(self, n_arms:int, epsilon:float, alpha:float=0.9, initial:float=0):
        self.n_arms = n_arms
        self.alpha = alpha      # diminishing rate of epsilon
        self.initial = initial  # set to 0 by default
        self.epsilon = epsilon
        self.counts = np.zeros(self.n_arms)
        self.qs = np.zeros(self.n_arms) + self.initial
        self.epsilon_ = self.epsilon
            
    def choose(self):
        p = np.random.uniform(low=0., high=1.)
        if p > self.epsilon_:
            argmaxes = np.where(self.qs == np.max(self.qs))[0]
            idx = np.random.choice(argmaxes)
        else:
            idx = np.random.choice(self.n_arms)
        return idx
    
    def update(self, a:int, r:float):
        """
        action: index of the chosen arm
        reward: reward of the chosen arm
        """
        ## count update
        self.counts[a] += 1
        
        ## q update
        value = self.qs[a]
        n = self.counts[a]
        new_value = (((n-1)/n)*value) + ((1/n)*r)
        self.qs[a] = new_value
        
        ## epsilon update
        self.epsilon_ *= self.alpha
        

class ETC(MAB):
    ## Explore-then-commit Bandit
    def __init__(self, n_arms:int, explore:int, horizon:int, initial:float=0):
        assert explore * n_arms <= horizon
        self.n_arms = n_arms
        self.initial = initial  # set to 0 by default
        self.explore = explore  # total steps for exploration
        self.counts = np.zeros(self.n_arms)
        self.qs = np.zeros(self.n_arms) + self.initial
        self.step = 0
    
    def choose(self):
        ## exploration step
        if self.step < self.explore * self.n_arms:
            idx = self.step % self.n_arms
        ## exploitation step
        else:
            argmaxes = np.where(self.qs == np.max(self.qs))[0]
            idx = np.random.choice(argmaxes)
        self.step += 1
        return idx
    
    def update(self, a:int, r:float):
        """
        a: index of the chosen arm
        r: reward of the chosen arm
        """
        ## count update
        self.counts[a] += 1
        
        ## q update
        value = self.qs[a]
        n = self.counts[a]
        new_value = (((n-1)/n)*value) + ((1/n)*r)
        self.qs[a] = new_value


class UCBNaive(MAB):
    def __init__(self, n_arms:int, sigma:float, alpha:float, delta:float=0.1):
        self.n_arms = n_arms
        self.alpha = alpha
        self.delta = delta
        self.sigma = sigma
        self.counts = np.zeros(self.n_arms)
        self.qs = np.zeros(self.n_arms)
        self.ucbs = np.array([np.iinfo(np.int32).max for _ in range(self.n_arms)])
        self.step = 0
    
    def choose(self):
        self.step += 1
        returns = self.qs + self.ucbs
        argmaxes = np.where(returns == np.max(returns))[0]
        return np.random.choice(argmaxes)
    
    def update(self, a:int, r:float):
        """
        a: index of the chosen arm
        r: reward of the chosen arm
        """
        ## count update
        self.counts[a] += 1
        
        ## q update
        value = self.qs[a]
        n = self.counts[a]
        new_value = (((n-1)/n)*value) + ((1/n)*r)
        self.qs[a] = new_value
        
        ## ucb update
        inside = 2 * (self.sigma ** 2) * np.log(self.step/self.delta)
        self.ucbs[a] = self.alpha * np.sqrt(inside)


class UCBDelta(UCBNaive):
    def __init__(self, n_arms:int, delta:float):
        # set default values for sigma and alpha
        sigma_default = 0.0
        alpha_default = 0.0
        super().__init__(n_arms, sigma_default, alpha_default, delta)
        self.n_arms = n_arms
        self.delta = delta
    
    def update(self, a:int, r:float):
        """
        a: index of the chosen arm
        r: reward of the chosen arm
        """
        ## count update
        self.counts[a] += 1
        
        ## q update
        value = self.qs[a]
        n = self.counts[a]
        new_value = (((n-1)/n)*value) + ((1/n)*r)
        self.qs[a] = new_value
        
        ## ucb update
        numerator = 2 * np.log(1/self.delta)
        self.ucbs[a] = np.sqrt(numerator / self.counts[a])
        
        
class UCBAsymptotic(UCBNaive):
    def __init__(self, n_arms:int):
        self.n_arms = n_arms
    
    def update(self, a:int, r:float):
        """
        a: index of the chosen arm
        r: reward of the chosen arm
        """
        ## count update
        self.counts[a] += 1
        
        ## q update
        value = self.qs[a]
        n = self.counts[a]
        new_value = (((n-1)/n)*value) + ((1/n)*r)
        self.qs[a] = new_value
        
        ## ucb update
        ft = 1 + (self.step * (np.log(self.step)**2))
        numerator = 2 * np.log(ft)
        self.ucbs[a] = np.sqrt(numerator / self.counts[a])
        

class UCBMOSS(UCBNaive):
    def __init__(self, n_arms:int, nsim:int):
        self.n_arms = n_arms
        self.nsim = nsim
        
    def update(self, a:int, r:float):
        """
        action: index of the chosen arm
        reward: reward of the chosen arm
        """
        ## count update
        self.counts[a] += 1
        
        ## q update
        value = self.qs[a]
        n = self.counts[a]
        new_value = (((n-1)/n)*value) + ((1/n)*r)
        self.qs[a] = new_value
        
        ## ucb update
        left = 4 / n
        right = np.log(np.maximum(1, (self.nsim / (self.n_arms*n))))
        self.ucbs[a] = np.sqrt(left * right)


class ThompsonSampling(MAB):
    def __init__(self, n_arms:int, bernoulli:bool):
        self.n_arms = n_arms
        self.bernoulli = bernoulli
        self.counts = np.zeros(shape=self.n_arms)
        self.qs = np.zeros(shape=self.n_arms)

        if self.bernoulli:
            self.alphas = np.ones(shape=self.n_arms)
            self.betas = np.ones(shape=self.n_arms)
        else:
            self.mus = np.zeros(shape=self.n_arms)
            self.devs = np.ones(shape=self.n_arms)
    
    def choose(self):
        if self.bernoulli:
            thetas = np.array([np.random.beta(a=alpha, b=beta) for (alpha, beta) in zip(self.alphas, self.betas)])
        else:
            thetas = np.array([np.random.normal(loc=mu, scale=var) for (mu, var) in zip(self.mus, self.devs)])
        argmaxes = np.where(thetas == np.max(thetas))[0]
        return np.random.choice(argmaxes)
    
    def update(self, a:int, r:float):
        """
        a: index of the chosen arm
        r: reward of the chosen arm
        """
        ## count update
        self.counts[a] += 1
        
        ## q update
        value = self.qs[a]
        n = self.counts[a]
        new_value = (((n-1)/n)*value) + ((1/n)*r)
        self.qs[a] = new_value
        
        ## parameter update
        if self.bernoulli:
            self.alphas[a] += r
            self.betas[a] += (1-r)
        else:
            self.mus[a] = new_value
            self.devs[a] = np.sqrt(1/n)


#############################################################################
############################ Contextual Bandits #############################
#############################################################################
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
        maxnorm = np.amax([vector_norm(x[i], type="l2") for i in range(arms)])
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
    

class RoLFLasso(ContextualBandit):
    def __init__(self, d:int, arms:int, p:float, delta:float, sigma:float, random_state:int):
        self.t = 0
        self.d = d
        self.K = arms
        self.mu_hat = np.zeros(self.K)
        self.impute_prev = np.zeros(self.K)
        self.main_prev = np.zeros(self.K)
        self.sigma = sigma          # variance of noise
        self.p = p                  # hyperparameter for action sampling
        self.delta = delta          # confidence parameter
        self.action_history = []    # history of chosen actions up to the current round
        self.reward_history = []    # history of observed rewards up to the current round
        self.matching = dict()      # history of rounds that the pseudo action and the chosen action matched
        self.random_state = random_state

    def choose(self, x: np.ndarray):
        # x : (K, d) augmented feature matrix where each row denotes the augmented features
        self.t += 1

        ## compute the \hat{a}_t
        # if self.t > self.K:
        #     decision_rule = x @ self.mu_hat
        #     print(f"Decision rule : {decision_rule}")
        #     a_hat = np.argmax(decision_rule)
        # else:
        #     a_hat = np.random.choice(np.arange(self.K))

        decision_rule = x @ self.mu_hat
        print(f"Decision rule : {decision_rule}")
        a_hat = np.argmax(decision_rule)

        ## sampling actions
        pseudo_action = -1
        chosen_action = -2
        count = 0
        max_iter = int(np.log((self.t + 1) ** 2 / self.delta) / np.log(1 / self.p))
        pseudo_dist = np.array([(1 - self.p) / (self.K - 1)] * self.K, dtype=float)
        pseudo_dist[a_hat] = self.p
        chosen_dist = np.array([(1 / np.sqrt(self.t)) / (self.K - 1)] * self.K, dtype=float)
        chosen_dist[a_hat] = 1 - (1 / np.sqrt(self.t))

        np.random.seed(self.random_state + self.t)
        while (pseudo_action != chosen_action) and (count <= max_iter):
            ## Sample the pseudo action
            pseudo_action = np.random.choice([i for i in range(self.K)], size=1, replace=False, p=pseudo_dist).item()
            ## Sample the chosen action
            chosen_action = np.random.choice([i for i in range(self.K)], size=1, replace=False, p=chosen_dist).item()
            count += 1

        self.action_history.append(chosen_action) # add to the history
        self.pseudo_action = pseudo_action
        self.chosen_action = chosen_action
        print(f"Round: {self.t}, a_hat: {a_hat}, pseudo_action: {pseudo_action}, chosen_action: {chosen_action}, count: {count}")
        return chosen_action

    def update(self, x: np.ndarray, r: float):
        # x : (K, d) augmented feature matrix
        # r : reward of the chosen_action
        self.reward_history.append(r)

        # lam_impute = 2 * self.p * self.sigma * np.sqrt(2 * self.t * np.log(2 * self.K * (self.t ** 2) / self.delta))
        # lam_main = (1 + 2 / self.p) * self.sigma * np.sqrt(2 * self.t * np.log(2 * self.K * (self.t ** 2) / self.delta))

        # lam_impute = self.p * np.sqrt(np.log(self.t))
        # lam_main = self.p * np.sqrt(np.log(self.t))

        lam_impute = self.p
        lam_main = self.p

        if self.pseudo_action == self.chosen_action:
            ## compute the imputation estimator
            data_impute = x[self.action_history, :]  # (t, d) matrix
            target_impute = np.array(self.reward_history)
            mu_impute = scipy.optimize.minimize(self.__imputation_loss, self.impute_prev, args=(data_impute, target_impute, lam_impute),
                                                method="SLSQP", options={'disp': False, "ftol":1e-6, "maxiter":10000}).x
            pseudo_rewards = x @ mu_impute
            pseudo_rewards[self.chosen_action] += (1 / self.p) * (r - (x[self.chosen_action, :] @ mu_impute))

            ## add the information
            self.matching[self.t] = ((self.pseudo_action == self.chosen_action), x, pseudo_rewards)

            ## compute the main estimator
            mu_main = scipy.optimize.minimize(self.__main_loss, self.main_prev, args=(lam_main, self.matching),
                                              method="SLSQP", options={'disp': False, "ftol":1e-6, "maxiter":10000}).x

            ## update the mu_hat
            self.mu_hat = mu_main
        else:
            self.matching[self.t] = ((self.pseudo_action == self.chosen_action), None, None)

    def __imputation_loss(self, beta:np.ndarray, X:np.ndarray, y:np.ndarray, lam:float):
        residuals = (y - (X @ beta)) ** 2
        loss = np.sum(residuals, axis=0)
        l1_norm = vector_norm(beta, type="l1")
        return loss + (lam * l1_norm)

    def __main_loss(self, beta:np.ndarray, lam:float, matching_history:dict):
        ## matching_history : dict[t] = (bool, X, y) - bool denotes whether the matching event occurred or not
        loss = 0
        for key in matching_history:
            matched, X, pseudo_rewards = matching_history[key]
            if matched:
                residuals = (pseudo_rewards - (X @ beta)) ** 2
                interim_loss = np.sum(residuals, axis=0)
            else:
                interim_loss = 0
            loss += interim_loss
        l1_norm = vector_norm(beta, type="l1")
        return loss + (lam * l1_norm)