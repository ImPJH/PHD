import numpy as np
from util import *
from abc import ABC, abstractmethod
from calculate_alpha import *
import scipy
from sklearn.linear_model import Lasso, LinearRegression
import statsmodels.api as sm
from typing import Callable, Optional


#############################################################################
############################ Multi-Armed Bandits ############################
#############################################################################
class MAB(ABC):
    @abstractmethod
    def choose(self):
        pass

    @abstractmethod
    def update(self, a, r):
        pass


class eGreedyMAB(MAB):
    def __init__(
        self, arms: int, epsilon: float, alpha: float = 1.0, initial: float = 0
    ):
        self.arms = arms
        self.epsilon = epsilon
        self.alpha = alpha
        self.initial = initial
        self.counts = np.zeros(self.arms)
        self.values = np.zeros(self.arms) + self.initial
        self.t = 0

    def choose(self):
        self.t += 1
        # print(f"Round : {self.t}, Epsilon: {self.epsilon}")
        if np.random.random() < self.epsilon:
            return np.random.choice(self.arms)
        else:
            (argmaxes,) = np.where(self.values == np.max(self.values))
            return np.random.choice(argmaxes)

    def update(self, a, r):
        ## count update
        self.counts[a] += 1

        ## value update
        value = self.values[a]
        n = self.counts[a]
        new_value = (((n - 1) / n) * value) + ((1 / n) * r)
        self.values[a] = new_value

        ## epsilon update
        self.epsilon *= self.alpha


class ETC(MAB):
    ## Explore-then-Commit
    def __init__(self, arms: int, explore: int, horizon: int, initial: float = 0):
        assert (
            explore * arms <= horizon
        ), "Explore must be less than or equal to horizon"
        self.explore = explore
        self.arms = arms
        self.initial = initial
        self.counts = np.zeros(self.arms)
        self.values = np.zeros(self.arms) + self.initial
        self.t = 0

    def choose(self):
        ## Exploration Step
        self.t += 1
        if (self.t - 1) <= self.explore * self.arms:
            return (self.t - 1) % self.arms

        ## Exploitation Step
        (argmaxes,) = np.where(self.values == np.max(self.values))
        return np.random.choice(argmaxes)

    def update(self, a, r):
        ## count update
        self.counts[a] += 1

        ## value update
        value = self.values[a]
        n = self.counts[a]
        new_value = (((n - 1) / n) * value) + ((1 / n) * r)
        self.values[a] = new_value


class UCBNaive(MAB):
    def __init__(
        self, n_arms: int, sigma: float = 0.1, alpha: float = 0.1, delta: float = 0.1
    ):
        self.n_arms = n_arms
        self.alpha = alpha
        self.delta = delta
        self.sigma = sigma
        self.counts = np.zeros(self.n_arms)
        self.qs = np.zeros(self.n_arms)
        self.ucbs = np.array([np.iinfo(np.int32).max for _ in range(self.n_arms)])
        self.t = 0

    def choose(self):
        self.t += 1
        returns = self.qs + self.ucbs
        argmaxes = np.where(returns == np.max(returns))[0]
        return np.random.choice(argmaxes)

    def update(self, a: int, r: float):
        """
        a: index of the chosen arm
        r: reward of the chosen arm
        """
        ## count update
        self.counts[a] += 1

        ## q update
        value = self.qs[a]
        n = self.counts[a]
        new_value = (((n - 1) / n) * value) + ((1 / n) * r)
        self.qs[a] = new_value

        ## ucb update
        inside = 2 * (self.sigma**2) * np.log(self.t / self.delta)
        self.ucbs[a] = self.alpha * np.sqrt(inside)


class UCBDelta(UCBNaive):
    def __init__(self, n_arms: int, delta: float):
        # set default values for sigma and alpha
        self.n_arms = n_arms
        self.delta = delta
        super().__init__(self.n_arms, delta=self.delta)

    def update(self, a: int, r: float):
        """
        a: index of the chosen arm
        r: reward of the chosen arm
        """
        ## count update
        self.counts[a] += 1

        ## q update
        value = self.qs[a]
        n = self.counts[a]
        new_value = (((n - 1) / n) * value) + ((1 / n) * r)
        self.qs[a] = new_value

        ## ucb update
        numerator = 2 * np.log(1 / self.delta)
        self.ucbs[a] = np.sqrt(numerator / self.counts[a])


class UCBAsymptotic(UCBNaive):
    def __init__(self, arms: int):
        self.arms = arms
        super().__init__(self.n_arms)

    def update(self, a, r):
        ## count update
        self.counts[a] += 1

        ## q update
        value = self.qs[a]
        n = self.counts[a]
        new_value = (((n - 1) / n) * value) + ((1 / n) * r)
        self.qs[a] = new_value

        ## ucb update
        ft = 1 + (self.t * (np.log(self.t) ** 2))
        numerator = 2 * np.log(ft)
        self.ucbs[a] = np.sqrt(numerator / self.counts[a])


class UCBMOSS(UCBNaive):
    def __init__(self, arms: int, horizon: int):
        self.arms = arms
        self.horizon = horizon
        super().__init__(self.n_arms)

    def update(self, a, r):
        ## count update
        self.counts[a] += 1

        ## q update
        value = self.qs[a]
        n = self.counts[a]
        new_value = (((n - 1) / n) * value) + ((1 / n) * r)
        self.qs[a] = new_value

        ## ucb update
        left = 4 / n
        right = np.log(np.maximum(1, (self.horizon / (self.n_arms * n))))
        self.ucbs[a] = np.sqrt(left * right)


class ThompsonSampling(MAB):
    def __init__(self, arms: int, distribution: str):
        self.arms = arms
        assert distribution.lower() in [
            "bernoulli",
            "gaussian",
        ], "Distribution must be either Bernoulli or Gaussian"
        self.distribution = distribution
        if distribution.lower() == "bernoulli":
            self.alphas = np.ones(self.arms)
            self.betas = np.ones(self.arms)
        elif distribution.lower() == "gaussian":
            self.mus = np.zeros(self.arms)
            self.sigmas = np.ones(self.arms)
        self.counts = np.zeros(shape=self.arms)
        self.qs = np.zeros(shape=self.arms)

    def choose(self):
        if self.distribution.lower() == "bernoulli":
            thetas = np.array(
                [
                    np.random.beta(a=alpha, b=beta)
                    for (alpha, beta) in zip(self.alphas, self.betas)
                ]
            )
        elif self.distribution.lower() == "gaussian":
            thetas = np.array(
                [
                    np.random.normal(loc=mu, scale=var)
                    for (mu, var) in zip(self.mus, self.sigmas)
                ]
            )
        (argmaxes,) = np.where(thetas == np.max(thetas))
        return np.random.choice(argmaxes)

    def update(self, a, r):
        ## count update
        self.counts[a] += 1

        ## q update
        value = self.qs[a]
        n = self.counts[a]
        new_value = (((n - 1) / n) * value) + ((1 / n) * r)
        self.qs[a] = new_value

        ## parameter update
        if self.distribution.lower() == "bernoulli":
            self.alphas[a] += r
            self.betas[a] += 1 - r
        elif self.distribution.lower() == "gaussian":
            self.mus[a] = new_value
            self.sigmas[a] = np.sqrt(1 / n)


#############################################################################
############################ Contextual Bandits #############################
#############################################################################
class ContextualBandit(ABC):
    @abstractmethod
    def choose(self, x):
        pass

    @abstractmethod
    def update(self, x, r):
        pass


class LinUCB(ContextualBandit):
    def __init__(self, d: int, lbda: float, delta: float) -> None:
        self.d = d
        self.xty = np.zeros(d)
        self.Vinv = (1 / lbda) * np.identity(d)
        self.theta_hat = np.zeros(d)
        self.delta = delta
        self.t = 0

    def choose(self, x: np.ndarray) -> int:
        # x: action set at each round (N, d)
        self.t += 1

        ## compute the ridge estimator
        self.theta_hat = self.Vinv @ self.xty

        ## compute the ucb scores for each arm
        alpha = linucb_alpha(delta=self.delta) * np.sqrt(np.log(self.t))
        expected = x @ self.theta_hat  # (N, ) theta_T @ x_t
        width = np.sqrt(np.einsum("Ni, ij, Nj -> N", x, self.Vinv, x))  # (N, ) widths
        ucb_scores = expected + (alpha * width)  # (N, ) ucb score

        ## chose the argmax the ucb score
        maximum = np.max(ucb_scores)
        (argmax,) = np.where(ucb_scores == maximum)
        self.chosen_action = np.random.choice(argmax)
        return self.chosen_action

    def update(self, x: np.ndarray, r: float) -> None:
        # x: context of the chosen action (d, )
        chosen_context = x[self.chosen_action, :]
        self.Vinv = shermanMorrison(self.Vinv, chosen_context)
        self.xty += r * chosen_context

    def __get_param(self):
        return {"param": self.theta_hat}


class LinTS(ContextualBandit):
    def __init__(
        self, d: int, lbda: float, horizon: int, reward_std: float, delta: float
    ) -> None:
        self.d = d
        self.Binv = (1 / lbda) * np.identity(d)
        self.xty = np.zeros(d)
        self.theta_hat = np.zeros(d)
        self.horizon = horizon
        self.reward_std = reward_std
        self.delta = delta
        self.t = 0

    def choose(self, x: np.ndarray) -> int:
        # x: action set at each round (N, d)
        self.t += 1

        ## compute the ridge estimator
        self.theta_hat = self.Binv @ self.xty

        ## parameter sampling
        # self.alpha_ = self.alpha * np.sqrt(np.log(self.t))
        # alpha = lints_alpha(d=self.d, horizon=self.horizon, reward_std=self.reward_std, delta=self.delta) * np.sqrt(np.log(self.t))
        alpha = lints_alpha(d=self.d, reward_std=self.reward_std, delta=self.delta)
        tilde_theta = np.random.multivariate_normal(
            mean=self.theta_hat, cov=(alpha**2) * self.Binv
        )  # (d, ) random matrix

        ## compute estimates and choose the argmax
        expected = x @ tilde_theta  # (N, ) vector
        maximum = np.max(expected)
        (argmax,) = np.where(expected == maximum)
        self.chosen_action = np.random.choice(argmax)
        return self.chosen_action

    def update(self, x: np.ndarray, r: float) -> None:
        # x: (K, d)
        # r: reward seen (scalar)
        chosen_context = x[self.chosen_action, :]
        self.Binv = shermanMorrison(self.Binv, chosen_context)
        self.xty += r * chosen_context

    def __get_param(self):
        return {"param": self.theta_hat}


class RoLFLasso(ContextualBandit):
    def __init__(
        self,
        d: int,
        arms: int,
        p: float,
        delta: float,
        sigma: float,
        random_state: int,
        explore: bool = False,
        init_explore: int = 0,
    ):
        self.t = 0
        self.d = d
        self.K = arms
        self.mu_hat = np.zeros(self.d)  # main estimator - should match feature dimension
        self.mu_check = np.zeros(self.d)  # imputation estimator
        self.impute_prev = np.zeros(self.d)
        self.main_prev = np.zeros(self.d)
        self.sigma = sigma  # variance of noise
        self.p = p  # hyperparameter for action sampling
        self.delta = delta  # confidence parameter
        self.action_history = []  # history of chosen actions up to the current round
        self.reward_history = []  # history of observed rewards up to the current round
        self.matching = (
            dict()
        )  # history of rounds that the pseudo action and the chosen action matched
        self.random_state = random_state
        self.explore = explore
        self.init_explore = init_explore

    def choose(self, x: np.ndarray, y: Optional[np.ndarray] = None):
        # x : (K, d) augmented feature matrix where each row denotes the augmented features
        self.t += 1
        
        # Ensure estimator arrays match the actual feature dimension on first call
        if self.t == 1 and len(self.mu_hat) != x.shape[1]:
            self.mu_hat = np.zeros(x.shape[1])
            self.mu_check = np.zeros(x.shape[1])
            self.impute_prev = np.zeros(x.shape[1])
            self.main_prev = np.zeros(x.shape[1])

        ## compute the \hat{a}_t
        if self.explore:
            if self.t > self.init_explore:
                decision_rule = x @ self.mu_hat
                # print(f"Decision rule : {decision_rule}")
                a_hat = np.argmax(decision_rule)
            else:
                a_hat = np.random.choice(np.arange(self.K))
        else:
            decision_rule = x @ self.mu_hat
            # print(f"Decision rule : {decision_rule}")
            a_hat = np.argmax(decision_rule)

        self.a_hat = a_hat

        ## sampling actions
        pseudo_action = -1
        chosen_action = -2
        count = 0

        ## ~! rho_t !~ ##
        max_iter = int(np.log((self.t + 1) ** 2 / self.delta) / np.log(1 / self.p))

        ## ~! phi_t !~ ##
        pseudo_dist = np.array([(1 - self.p) / (self.K - 1)] * self.K, dtype=float)
        pseudo_dist[a_hat] = self.p

        ## ~! epsilon(sqrt(t))-greedy ~! ##
        chosen_dist = np.array(
            [(1 / np.sqrt(self.t)) / (self.K - 1)] * self.K, dtype=float
        )
        chosen_dist[a_hat] = 1 - (1 / np.sqrt(self.t))

        np.random.seed(self.random_state + self.t)
        while (pseudo_action != chosen_action) and (count <= max_iter):
            ## Sample the pseudo action
            pseudo_action = np.random.choice(
                [i for i in range(self.K)], size=1, replace=False, p=pseudo_dist
            ).item()
            ## Sample the chosen action
            chosen_action = np.random.choice(
                [i for i in range(self.K)], size=1, replace=False, p=chosen_dist
            ).item()
            count += 1

        self.action_history.append(chosen_action)  # add to the history
        self.pseudo_action = pseudo_action
        self.chosen_action = chosen_action
        return chosen_action

    def update(self, x: np.ndarray, r: float):
        # x : (K, K) augmented feature matrix
        # r : reward of the chosen_action
        self.reward_history.append(r)

        # lam_impute = 2 * self.p * self.sigma * np.sqrt(2 * self.t * np.log(2 * self.K * (self.t ** 2) / self.delta))
        # lam_main = (1 + 2 / self.p) * self.sigma * np.sqrt(2 * self.t * np.log(2 * self.K * (self.t ** 2) / self.delta))

        # lam_impute = self.p * np.sqrt(np.log(self.t))
        # lam_main = self.p * np.sqrt(np.log(self.t))

        lam_impute = self.p
        lam_main = self.p

        # print(f"x : {x.shape}")
        # gram = x.T @ x
        # gram_sqrt = matrix_sqrt(gram)

        if self.pseudo_action == self.chosen_action:
            ## compute the imputation estimator
            # Check if we're in a bilinear setting by comparing max action with data dimensions
            max_action_in_history = max(self.action_history) if self.action_history else 0
            
            if max_action_in_history >= x.shape[0]:
                # We're in bilinear mode - need to convert global actions to row indices
                # Infer N from the relationship: max_action = M*N - 1, where M = x.shape[0]
                M = x.shape[0]
                # Estimate N based on the maximum action seen
                estimated_total_actions = max_action_in_history + 1
                N = estimated_total_actions // M
                if N * M < estimated_total_actions:
                    N += 1
                
                i_history = [action // N for action in self.action_history]
                data_impute = x[i_history, :]  # (t, d) matrix
            else:
                # Regular case where actions directly correspond to rows
                data_impute = x[self.action_history, :]  # (t, d) matrix
            
            target_impute = np.array(self.reward_history)
            # print(f"gram_sqrt : {gram_sqrt.shape}")
            # print(f"impute_prev : {self.impute_prev.shape}")
            mu_impute = scipy.optimize.minimize(
                self.__imputation_loss,
                self.impute_prev,
                args=(data_impute, target_impute, lam_impute),
                method="SLSQP",
                options={"disp": False, "ftol": 1e-6, "maxiter": 10000},
            ).x
            
            # Ensure mu_impute has the correct dimension to match x.shape[1]
            if len(mu_impute) != x.shape[1]:
                # If dimensions don't match, resize mu_impute to match x.shape[1]
                if len(mu_impute) < x.shape[1]:
                    # Pad with zeros if too small
                    mu_impute = np.pad(mu_impute, (0, x.shape[1] - len(mu_impute)))
                else:
                    # Truncate if too large
                    mu_impute = mu_impute[:x.shape[1]]

            ## compute and update the pseudo rewards
            if self.matching:
                for key in self.matching:
                    matched, data, _, chosen, reward = self.matching[key]
                    if matched:
                        new_pseudo_rewards = data @ mu_impute
                        
                        # Check if we're in bilinear mode for the stored action
                        if chosen >= data.shape[0]:
                            # Convert global action to row index
                            M = data.shape[0]
                            estimated_total_actions = chosen + 1
                            N = estimated_total_actions // M
                            if N * M < estimated_total_actions:
                                N += 1
                            chosen_i = chosen // N
                            new_pseudo_rewards[chosen_i] += (1 / self.p) * (
                                reward - (data[chosen_i, :] @ mu_impute)
                            )
                        else:
                            new_pseudo_rewards[chosen] += (1 / self.p) * (
                                reward - (data[chosen, :] @ mu_impute)
                            )
                        # overwrite the value
                        self.matching[key] = (
                            matched,
                            data,
                            new_pseudo_rewards,
                            chosen,
                            reward,
                        )

            ## compute the pseudo rewards for the current data
            pseudo_rewards = x @ mu_impute
            
            # Check if we're in bilinear mode for the current chosen action
            if self.chosen_action >= x.shape[0]:
                # Convert global action to row index
                M = x.shape[0]
                estimated_total_actions = self.chosen_action + 1
                N = estimated_total_actions // M
                if N * M < estimated_total_actions:
                    N += 1
                chosen_i = self.chosen_action // N
                pseudo_rewards[chosen_i] += (1 / self.p) * (
                    r - (x[chosen_i, :] @ mu_impute)
                )
            else:
                pseudo_rewards[self.chosen_action] += (1 / self.p) * (
                    r - (x[self.chosen_action, :] @ mu_impute)
                )
            self.matching[self.t] = (
                (self.pseudo_action == self.chosen_action),
                x,
                pseudo_rewards,
                self.chosen_action,
                r,
            )

            ## compute the main estimator
            # Ensure main_prev has the correct dimension to match x.shape[1]
            main_prev_adjusted = self.main_prev
            if len(main_prev_adjusted) != x.shape[1]:
                if len(main_prev_adjusted) < x.shape[1]:
                    # Pad with zeros if too small
                    main_prev_adjusted = np.pad(main_prev_adjusted, (0, x.shape[1] - len(main_prev_adjusted)))
                else:
                    # Truncate if too large
                    main_prev_adjusted = main_prev_adjusted[:x.shape[1]]
                    
            mu_main = scipy.optimize.minimize(
                self.__main_loss,
                main_prev_adjusted,
                args=(lam_main, self.matching),
                method="SLSQP",
                options={"disp": False, "ftol": 1e-6, "maxiter": 10000},
            ).x

            ## update the mu_hat
            self.mu_hat = mu_main
            self.mu_check = mu_impute
            
            # Update prev arrays for next round (ensure they match current dimensions)
            self.impute_prev = mu_impute.copy()
            self.main_prev = mu_main.copy()
        else:
            self.matching[self.t] = (
                (self.pseudo_action == self.chosen_action),
                None,
                None,
                None,
                None,
            )

    # def __main_loss(self, beta:np.ndarray, lam:float, matching_history:dict):
    #     ## matching_history : dict[t] = (bool, X, y) - bool denotes whether the matching event occurred or not
    #     loss = 0
    #     for key in matching_history:
    #         matched, X, pseudo_rewards, _, _ = matching_history[key]
    #         if matched:
    #             residuals = (pseudo_rewards - (X @ beta)) ** 2
    #             interim_loss = np.sum(residuals, axis=0)
    #         else:
    #             interim_loss = 0
    #         loss += interim_loss
    #     l1_norm = vector_norm(beta, type="l1")
    #     return loss + (lam * l1_norm)

    # def __main_loss(self, beta:np.ndarray, lam:float, matching_history:dict):
    #     ## matching_history : dict[t] = (bool, X, y) - bool denotes whether the matching event occurred or not
    #     loss = 0
    #     for key in matching_history:
    #         matched, X, pseudo_rewards, _, _ = matching_history[key]
    #         if matched:
    #             residuals = (pseudo_rewards - (X @ beta)) ** 2
    #             interim_loss = np.sum(residuals, axis=0)
    #         else:
    #             interim_loss = 0
    #         loss += interim_loss
    #     l1_norm = vector_norm(beta, type="l1")
    #     return loss + (lam * l1_norm)

    def __main_loss(self, beta: np.ndarray, lam: float, matching_history: dict):
        # Extract matched keys and data
        matched_keys = [
            key for key, value in matching_history.items() if value[0]
        ]  # Filter matched entries
        X_list = [
            matching_history[key][1] for key in matched_keys
        ]  # List of X matrices
        pseudo_rewards_list = [
            matching_history[key][2] for key in matched_keys
        ]  # List of pseudo_rewards

        # Compute residuals for matched keys
        residuals_list = [
            (pseudo_rewards - (X @ beta)) ** 2
            for X, pseudo_rewards in zip(X_list, pseudo_rewards_list)
        ]

        # Sum all residuals efficiently
        residuals_sum = sum(np.sum(residuals, axis=0) for residuals in residuals_list)

        # L1 regularization
        l1_norm = np.sum(np.abs(beta))

        # Total loss
        return residuals_sum + lam * l1_norm

    def __imputation_loss(
        self, beta: np.ndarray, X: np.ndarray, y: np.ndarray, lam: float
    ):
        """
        Generic imputation loss for RoLFLasso.

        Handles two shapes of beta:
        1) beta has length equal to X.shape[1]  → treats beta as a (d,) vector.
        2) beta has length > X.shape[1] (e.g., flattened square matrix) →
           reshapes beta to (d, d2) and predicts with an all-ones item vector.
        """
        d = X.shape[1]                  # feature dimension on user side

        if beta.size == d:
            # Standard contextual case: beta is a (d,) vector
            preds = X @ beta            # (t,)
        else:
            # Expanded case: beta represents a matrix Φ of shape (d, d2)
            # Infer d2 from beta length
            d2 = beta.size // d
            Phi = beta.reshape(d, d2)   # (d, d2)
            preds = X @ Phi @ np.ones(d2)  # (t,)

        residuals = (y - preds) ** 2
        loss = np.sum(residuals)

        # ℓ₁ regularization (vector or matrix flattened)
        l1_norm = np.sum(np.abs(beta))

        return loss + lam * l1_norm

    def __get_param(self):
        return {"param": self.mu_hat, "impute": self.mu_check}


class RoLFRidge(ContextualBandit):
    def __init__(
        self,
        d: int,
        arms: int,
        p: float,
        delta: float,
        sigma: float,
        random_state: int,
        explore: bool = False,
        init_explore: int = 0,
    ):
        self.t = 0
        self.d = d
        self.K = arms
        self.mu_hat = np.zeros(self.d)  # main estimator - should match feature dimension
        self.mu_check = np.zeros(self.d)  # imputation estimator
        self.sigma = sigma  # variance of noise
        self.p = p  # hyperparameter for action sampling
        self.delta = delta  # confidence parameter
        self.matching = (
            dict()
        )  # history of rounds that the pseudo action and the chosen action matched
        self.Vinv_impute = self.p * np.identity(self.d)
        self.xty_impute = np.zeros(self.d)
        self.random_state = random_state
        self.explore = explore
        self.init_explore = init_explore

    def choose(self, x: np.ndarray, y: Optional[np.ndarray] = None):
        # x : (K, d) augmented feature matrix where each row denotes the augmented features
        self.t += 1
        
        # Ensure estimator arrays match the actual feature dimension on first call
        if self.t == 1 and len(self.mu_hat) != x.shape[1]:
            self.mu_hat = np.zeros(x.shape[1])
            self.mu_check = np.zeros(x.shape[1])
            self.Vinv_impute = self.p * np.identity(x.shape[1])
            self.xty_impute = np.zeros(x.shape[1])

        ## compute the \hat{a}_t
        if self.explore:
            if self.t > self.init_explore:
                decision_rule = x @ self.mu_hat
                # print(f"Decision rule : {decision_rule}")
                a_hat = np.argmax(decision_rule)
            else:
                a_hat = np.random.choice(np.arange(self.K))
        else:
            decision_rule = x @ self.mu_hat
            # print(f"Decision rule : {decision_rule}")
            a_hat = np.argmax(decision_rule)

        self.a_hat = a_hat

        ## sampling actions
        pseudo_action = -1
        chosen_action = -2
        count = 0
        max_iter = int(np.log((self.t + 1) ** 2 / self.delta) / np.log(1 / self.p))
        pseudo_dist = np.array([(1 - self.p) / (self.K - 1)] * self.K, dtype=float)
        pseudo_dist[a_hat] = self.p
        chosen_dist = np.array(
            [(1 / np.sqrt(self.t)) / (self.K - 1)] * self.K, dtype=float
        )
        chosen_dist[a_hat] = 1 - (1 / np.sqrt(self.t))

        np.random.seed(self.random_state + self.t)
        while (pseudo_action != chosen_action) and (count <= max_iter):
            ## Sample the pseudo action
            pseudo_action = np.random.choice(
                [i for i in range(self.K)], size=1, replace=False, p=pseudo_dist
            ).item()
            ## Sample the chosen action
            chosen_action = np.random.choice(
                [i for i in range(self.K)], size=1, replace=False, p=chosen_dist
            ).item()
            count += 1

        self.pseudo_action = pseudo_action
        self.chosen_action = chosen_action
        # print(f"Round: {self.t}, a_hat: {a_hat}, pseudo_action: {pseudo_action}, chosen_action: {chosen_action}, count: {count}")
        return chosen_action

    def update(self, x: np.ndarray, r: float):
        # x : (K, K) augmented feature matrix
        # r : reward of the chosen_action
        if self.pseudo_action == self.chosen_action:
            ## compute the imputation estimator based on history
            # Check if we're in a bilinear setting by comparing chosen_action with data dimensions
            if self.chosen_action >= x.shape[0]:
                # We're in bilinear mode - need to convert global action to row index
                M = x.shape[0]
                # Estimate N based on the chosen action
                estimated_total_actions = self.chosen_action + 1
                N = estimated_total_actions // M
                if N * M < estimated_total_actions:
                    N += 1
                chosen_i = self.chosen_action // N
                chosen_context = x[chosen_i, :]
            else:
                # Regular case where action directly corresponds to row
                chosen_context = x[self.chosen_action, :]
                
            self.Vinv_impute = shermanMorrison(self.Vinv_impute, chosen_context)
            self.xty_impute += r * chosen_context
            mu_impute = self.Vinv_impute @ self.xty_impute

            ## compute and update the pseudo rewards
            if self.matching:
                for key in self.matching:
                    matched, data, _, chosen, reward = self.matching[key]
                    if matched:
                        new_pseudo_rewards = data @ mu_impute
                        
                        # Check if we're in bilinear mode for the stored action
                        if chosen >= data.shape[0]:
                            # Convert global action to row index
                            M = data.shape[0]
                            estimated_total_actions = chosen + 1
                            N = estimated_total_actions // M
                            if N * M < estimated_total_actions:
                                N += 1
                            chosen_i = chosen // N
                            new_pseudo_rewards[chosen_i] += (1 / self.p) * (
                                reward - (data[chosen_i, :] @ mu_impute)
                            )
                        else:
                            new_pseudo_rewards[chosen] += (1 / self.p) * (
                                reward - (data[chosen, :] @ mu_impute)
                            )
                        # overwrite the value
                        self.matching[key] = (
                            matched,
                            data,
                            new_pseudo_rewards,
                            chosen,
                            reward,
                        )

            ## compute the pseudo rewards for the current data
            pseudo_rewards = x @ mu_impute
            
            # Check if we're in bilinear mode for the current chosen action
            if self.chosen_action >= x.shape[0]:
                # Convert global action to row index
                M = x.shape[0]
                estimated_total_actions = self.chosen_action + 1
                N = estimated_total_actions // M
                if N * M < estimated_total_actions:
                    N += 1
                chosen_i = self.chosen_action // N
                pseudo_rewards[chosen_i] += (1 / self.p) * (
                    r - (x[chosen_i, :] @ mu_impute)
                )
            else:
                pseudo_rewards[self.chosen_action] += (1 / self.p) * (
                    r - (x[self.chosen_action, :] @ mu_impute)
                )
            self.matching[self.t] = (
                (self.pseudo_action == self.chosen_action),
                x,
                pseudo_rewards,
                self.chosen_action,
                r,
            )

            ## compute the main estimator
            mu_main = self.__main_estimation(self.matching, dimension=x.shape[1])

            ## update the mu_hat
            self.mu_hat = mu_main
            self.mu_check = mu_impute
        else:
            self.matching[self.t] = (
                (self.pseudo_action == self.chosen_action),
                None,
                None,
                None,
                None,
            )

    # def __main_estimation(self, matching_history:dict, dimension:int):
    #     ## matching_history : dict[t] = (bool, X, y) - bool denotes whether the matching event occurred or not
    #     inv = np.identity(dimension)
    #     score = np.zeros(dimension, dtype=float)
    #     for key in matching_history:
    #         matched, X, pseudo_rewards, _, _ = matching_history[key]
    #         if matched:
    #             # inverse matrix
    #             inv_init = np.zeros(shape=(dimension, dimension))
    #             for a in range(X.shape[0]):
    #                 inv_init += np.outer(X[a, :], X[a, :])
    #             inv += inv_init

    #             # score
    #             score_init = np.zeros(shape=dimension, dtype=float)
    #             for a in range(X.shape[0]):
    #                 score_init += pseudo_rewards[a] * X[a, :]
    #             score += score_init

    #     return scipy.linalg.inv(inv) @ score

    def __main_estimation(self, matching_history: dict, dimension: int):
        # Initialize inv and score
        inv = np.identity(dimension)
        score = np.zeros(dimension, dtype=float)

        # Filter matched entries
        matched_entries = [value for key, value in matching_history.items() if value[0]]

        # Process matched entries
        for _, X, pseudo_rewards, _, _ in matched_entries:
            # Update inv (outer products of rows in X)
            inv += X.T @ X

            # Update score (weighted sum of rows in X)
            score += X.T @ pseudo_rewards

        # Compute final estimation
        return scipy.linalg.inv(inv) @ score

    def __get_param(self):
        return {"param": self.mu_hat, "impute": self.mu_check}


class DRLassoBandit(ContextualBandit):
    def __init__(
        self, d: int, arms: int, lam1: float, lam2: float, zT: float, tr: bool
    ):
        ## learning params
        self.d = d
        self.arms = arms
        self.lam1 = lam1
        self.lam2 = lam2
        self.tr = tr
        self.zT = zT

        ## initialization
        self.beta_prev = np.zeros(self.d)
        self.beta_hat = np.zeros(self.d)
        self.pi_t = 0
        self.x = []  # containing context history
        self.r = []  # containing reward history
        self.t = 0  # learning round

    def choose(self, x):
        ## x : (K, d) array - all contexts observed at t
        self.t += 1
        if self.t <= self.zT:
            # forced sampling
            self.action = np.random.choice(self.arms, replace=False)
            self.pi_t = 1 / self.arms
        else:
            # UCB
            expected_reward = x @ self.beta_hat  # (K, ) array
            lam1 = self.lam1 * np.sqrt((np.log(self.t) + np.log(self.d)) / self.t)
            lam1 = np.minimum(1, np.maximum(0, lam1))
            self.mt = np.random.choice([0, 1], p=[1 - lam1, lam1])
            if self.mt == 1:
                self.action = np.random.choice(self.arms)
            else:
                self.action = np.argmax(expected_reward)

            self.pi_t = (lam1 / self.arms) + (
                (1 - lam1) * (self.action == np.argmax(expected_reward))
            )

        bar_x = np.mean(x, axis=0)
        self.x.append(bar_x)
        self.rhat = x @ self.beta_hat

        return self.action

    def update(self, x, r):
        ## x : (K, d) array - context of the all actions in round t
        ## r : float - reward
        # Handle potential out-of-bounds action in bilinear mode
        idx = self.action % x.shape[0]
        r_hat = np.mean(self.rhat) + (
            (r - (x[idx] @ self.beta_hat)) / (self.arms * self.pi_t)
        )
        if self.tr:
            r_hat = np.minimum(3.0, np.maximum(-3.0, r_hat))
        self.r.append(r_hat)

        lam2 = self.lam2 * np.sqrt((np.log(self.t) + np.log(self.d)) / self.t)
        data, target = np.vstack(self.x), np.array(self.r)
        self.beta_hat = scipy.optimize.minimize(
            self.__lasso_loss,
            self.beta_prev,
            args=(data, target, lam2),
            method="SLSQP",
            options={"disp": False, "ftol": 1e-6, "maxiter": 30000},
        ).x

    def __lasso_loss(self, beta: np.ndarray, X: np.ndarray, y: np.ndarray, lam: float):
        loss = np.sum((y - X @ beta) ** 2, axis=0)
        l1norm = np.sum(np.abs(beta))
        return loss + (lam * l1norm)

    def __get_param(self):
        return self.beta_hat


class LassoBandit(ContextualBandit):
    def __init__(
        self,
        arms: int,
        horizon: int,
        d: int,
        q: int,
        h: float,
        lam1: float,
        lam2: float,
    ):
        ## input params for algorithms
        self.q = q  # input param 1 - for forced-sampling
        self.h = h  # input param 2
        self.lam1 = lam1  # input param 3
        self.lam2 = lam2  # input param 4

        ## basic params for bandits
        self.arms = arms  # the number of arms
        self.horizon = horizon  # learning horizon
        self.d = d  # dimension of features
        self.t = 0  # learning round; t <= horizon
        self.n = 0  # sample size

        ## sets
        self.Tx = {i: [] for i in range(self.arms)}
        self.Sx = {i: [] for i in range(self.arms)}
        self.Tr = {i: [] for i in range(self.arms)}
        self.Sr = {i: [] for i in range(self.arms)}

        ## estmators
        self.beta_t = np.zeros((self.arms, d))  # forced-sample estimators
        self.beta_s = np.zeros((self.arms, d))  # all samples estimators
        self.lasso_t = Lasso(alpha=lam1)

    def choose(self, x: np.ndarray):
        ## x: (d, ) array - context vector of time t
        self.t += 1

        flag = (((2**self.n) - 1) * self.arms * self.q) + 1
        if self.t == flag:
            self.set = np.arange(self.t, self.t + (self.q * self.arms))
            self.n += 1

        if self.t in self.set:
            ## if t is in T_i for any i
            ind = list(self.set).index(self.t)
            self.action = ind // self.q
            self.Tx[self.action].append(x)
        else:
            ## if indices is none
            expected_T = self.beta_t @ x
            max_expected_T = np.amax(expected_T)
            K_hat = np.argwhere(expected_T >= (max_expected_T - (self.h / 2))).flatten()

            expected_S = self.beta_s @ x
            filtered_expected = expected_S[K_hat]
            argmax = np.argmax(filtered_expected)
            self.action = K_hat[argmax]

        self.Sx[self.action].append(
            x
        )  # append the context of the actually chosen action
        return self.action

    def update(self, r: float):
        if self.t in self.set:
            self.Tr[self.action].append(r)
            ## update beta_t using Lasso
            data_t, target_t = np.vstack(self.Tx[self.action]), np.array(
                self.Tr[self.action]
            )
            # print(data_t.shape)
            beta_t = scipy.optimize.minimize(
                self.__lasso_loss,
                np.zeros(d),
                args=(data_t, target_t, self.lam1),
                method="SLSQP",
                options={"disp": False, "ftol": 1e-6, "maxiter": 10000},
            ).x

            self.beta_t[self.action] = beta_t

        self.Sr[self.action].append(r)
        ## update beta_s using Lasso
        lam2_t = self.lam2 * np.sqrt(((np.log(self.t) + np.log(self.d)) / self.t))
        data_s, target_s = np.vstack(self.Sx[self.action]), np.array(
            self.Sr[self.action]
        )
        # print(f"action : {self.action}, data : {data_s.shape}")
        beta_s = scipy.optimize.minimize(
            self.__lasso_loss,
            np.zeros(d),
            args=(data_s, target_s, lam2_t),
            method="SLSQP",
            options={"disp": False, "ftol": 1e-6, "maxiter": 10000},
        ).x

        self.beta_s[self.action] = beta_s

    def __lasso_loss(self, beta: np.ndarray, X: np.ndarray, y: np.ndarray, lam: float):
        # print(X)
        # print(f"X : {X.shape}, beta : {beta.shape}")
        loss = np.sum((y - X @ beta) ** 2, axis=0)
        l1norm = np.sum(np.abs(beta))
        return loss + (lam * l1norm)


class BiRoLFLasso(ContextualBandit):
    def __init__(
        self,
        M: int,
        N: int,
        sigma: float,
        random_state: int,
        delta: float,
        p: float,
        explore: bool = False,
        init_explore: int = 0,
        theoretical_init_explore: bool = False,
    ):
        self.t = 0
        self.explore = explore
        ## TODO: make theoretical C_e
        if theoretical_init_explore:
            # self.init_explore = ((8*M*N)**3)
            pass
        else:
            self.init_explore = init_explore
        self.M = M
        self.N = N
        self.delta = delta
        self.p = p
        self.p1 = self.p
        self.p2 = self.p
        self.random_state = random_state
        self.sigma = sigma

        self.action_history = []
        self.reward_history = []

        self.matching = dict()
        self.Phi_hat = np.zeros((self.M, self.N))
        self.Phi_check = np.zeros((self.M, self.N))
        self.impute_prev = np.zeros((self.M, self.N))
        self.main_prev = np.zeros((self.M, self.N))
        
        # Initialize p1 and p2 for bilinear case
        self.p1 = self.p
        self.p2 = self.p

    def choose(self, x: np.ndarray, y: np.ndarray):
        # x : (M, M) augmented feature matrix where each row denotes the augmented features
        # y : (N, N) augmented feature matrix where each row denotes the augmented features

        self.t += 1

        ## compute the \hat{a}_t
        if self.explore:
            if self.t > self.init_explore:
                decision_rule = x @ self.Phi_hat @ y.T
                # print(f"Decision rule : {decision_rule}")
                a_hat = np.argmax(decision_rule)
            else:
                a_hat = np.random.choice(np.arange(self.M * self.N))
        else:
            ## decision_rule : (M,N)
            decision_rule = x @ self.Phi_hat @ y.T
            # print(f"Decision rule : {decision_rule}")
            a_hat = np.argmax(decision_rule)

        i_hat, j_hat = action_to_ij(a_hat, self.N)

        self.a_hat = a_hat
        self.i_hat = i_hat
        self.j_hat = j_hat

        ## sampling actions
        pseudo_action_i = -1
        pseudo_action_j = -1
        chosen_action_i = -2
        chosen_action_j = -2
        count1 = 0
        count2 = 0

        ## ~! rho_t !~ ##
        max_iter1 = int(
            np.log(2 * ((self.t + 1) ** 2) / self.delta) / np.log(1 / (1 - self.p1))
        )
        max_iter2 = int(
            np.log(2 * ((self.t + 1) ** 2) / self.delta) / np.log(1 / (1 - self.p2))
        )

        ## ~! phi_t !~ ##
        pseudo_dist_x = np.array([(1 - self.p1) / (self.M - 1)] * self.M, dtype=float)
        pseudo_dist_x[i_hat] = self.p1

        pseudo_dist_y = np.array([(1 - self.p2) / (self.N - 1)] * self.N, dtype=float)
        pseudo_dist_y[j_hat] = self.p2

        ## ~! epsilon(sqrt(t))-greedy ~! ##
        chosen_dist_x = np.array(
            [(1 / np.sqrt(self.t)) / (self.M - 1)] * self.M,
            dtype=float,
        )
        chosen_dist_x[i_hat] = 1 - (1 / np.sqrt(self.t))

        chosen_dist_y = np.array(
            [(1 / np.sqrt(self.t)) / (self.N - 1)] * self.N,
            dtype=float,
        )
        chosen_dist_y[j_hat] = 1 - (1 / np.sqrt(self.t))

        np.random.seed(self.random_state + self.t)

        # Initialize variables for sampling
        pseudo_action_i = -1
        chosen_action_i = -2
        pseudo_action_j = -1
        chosen_action_j = -2

        while (pseudo_action_i != chosen_action_i) and (count1 <= max_iter1):
            ## Sample the pseudo action
            pseudo_action_i = np.random.choice(
                [i for i in range(self.M)], size=1, replace=False, p=pseudo_dist_x
            ).item()

            ## Sample the chosen action
            chosen_action_i = np.random.choice(
                [i for i in range(self.M)], size=1, replace=False, p=chosen_dist_x
            ).item()

            count1 += 1

        while (pseudo_action_j != chosen_action_j) and (count2 <= max_iter2):
            ## Sample the pseudo action
            pseudo_action_j = np.random.choice(
                [i for i in range(self.N)], size=1, replace=False, p=pseudo_dist_y
            ).item()

            ## Sample the chosen action
            chosen_action_j = np.random.choice(
                [i for i in range(self.N)], size=1, replace=False, p=chosen_dist_y
            ).item()

            count2 += 1

        pseudo_action = pseudo_action_i * self.N + pseudo_action_j
        chosen_action = chosen_action_i * self.N + chosen_action_j

        self.action_history.append(chosen_action)  # add to the history
        self.pseudo_action = pseudo_action
        self.chosen_action = chosen_action
        return chosen_action

    def update(self, x: np.ndarray, y: np.ndarray, r: float):
        # x : (M, M) augmented feature matrix
        # y : (N, N) augmented feature matrix
        # r : reward of the chosen_action
        self.reward_history.append(r)

        # lam_impute = 2 * self.p * self.sigma * np.sqrt(2 * self.t * np.log(2 * self.K * (self.t ** 2) / self.delta))
        # lam_main = (1 + 2 / self.p) * self.sigma * np.sqrt(2 * self.t * np.log(2 * self.K * (self.t ** 2) / self.delta))

        # lam_impute = self.p * np.sqrt(np.log(self.t))
        # lam_main = self.p * np.sqrt(np.log(self.t))

        # lam_impute = self.p
        # lam_main = self.p

        kappa_x = np.power(np.sum(np.power(np.max(np.abs(x), axis=1), 4)), 0.25)
        kappa_y = np.power(np.sum(np.power(np.max(np.abs(y), axis=1), 4)), 0.25)

        lam_impute = (
            2
            * self.sigma
            * kappa_x
            * kappa_y
            * np.sqrt(2 * self.t * np.log(2 * self.M * self.N / self.delta))
        )
        lam_main = (4 * self.sigma * kappa_x * kappa_y / (self.p**2)) * np.sqrt(
            2 * self.t * np.log(2 * self.M * self.N * self.t**2 / self.delta)
        )
        # No gram/gram_sqrt calculation needed anymore

        if self.pseudo_action == self.chosen_action:
            ## compute the imputation estimator
<<<<<<< Updated upstream
            # Convert action indices to row indices for x matrix
            action_rows = [action_to_ij(action, self.N)[0] for action in self.action_history]
            action_cols = [action_to_ij(action, self.N)[1] for action in self.action_history]
            data_impute = x[action_rows, :]  # (t, d) matrix
=======
            # Recover user-side row indices from stored global actions
            i_history = [action // self.N for action in self.action_history]
            data_impute = x[i_history, :]
>>>>>>> Stashed changes
            target_impute = np.array(self.reward_history)
            # Prepare a 1D initial guess for the imputation estimator (no gram_sqrt)
            x0_impute = self.impute_prev.flatten()
            result_impute = scipy.optimize.minimize(
                self.__imputation_loss,
<<<<<<< Updated upstream
                self.impute_prev.flatten(),
                args=(data_impute, target_impute, lam_impute),
                method="SLSQP",
                options={"disp": False, "ftol": 1e-6, "maxiter": 10000},
            ).x.reshape(self.M, self.N)
=======
                x0_impute,
                args=(data_impute, target_impute, lam_impute),
                method="SLSQP",
                options={"disp": False, "ftol": 1e-6, "maxiter": 10000},
            )
            Phi_impute = result_impute.x.reshape(self.M, self.N)
>>>>>>> Stashed changes

            ## compute and update the pseudo rewards
            if self.matching:
                for key in self.matching:
                    if len(self.matching[key]) == 6:
                        matched, data_x, data_y, _, chosen, reward = self.matching[key]
                    else:
                        matched, data_x, data_y, _, _ = self.matching[key]
                        chosen, reward = None, None
                    if matched:
                        chosen_i, chosen_j = action_to_ij(chosen, self.N)
                        new_pseudo_rewards = data_x @ Phi_impute @ data_y.T
                        new_pseudo_rewards[chosen_i, chosen_j] += (
                            (1 / self.p) ** 2
                        ) * (
                            reward
                            - (data_x[chosen_i, :] @ Phi_impute @ data_y[chosen_j, :])
                        )
                        # overwrite the value
                        self.matching[key] = (
                            matched,
                            data_x,
                            data_y,
                            new_pseudo_rewards,
                            chosen,
                            reward,
                        )

            ## compute the pseudo rewards for the current data
            pseudo_rewards = x @ Phi_impute @ y.T
            chosen_i, chosen_j = action_to_ij(self.chosen_action, self.N)
            pseudo_rewards[chosen_i, chosen_j] += ((1 / self.p) ** 2) * (
                r - (x[chosen_i, :] @ Phi_impute @ y[chosen_j, :])
            )
            self.matching[self.t] = (
                (self.pseudo_action == self.chosen_action),
                x,
                y,
                pseudo_rewards,
                self.chosen_action,
                r,
            )

            ## compute the main estimator
            # Prepare a 1D initial guess for the main estimator
            x0_main = self.main_prev.flatten()
            result_main = scipy.optimize.minimize(
                self.__main_loss,
<<<<<<< Updated upstream
                self.main_prev.flatten(),
                args=(lam_main, self.matching),
                method="SLSQP",
                options={"disp": False, "ftol": 1e-6, "maxiter": 10000},
            ).x.reshape(self.M, self.N)
=======
                x0_main,
                args=(lam_main, self.matching),
                method="SLSQP",
                options={"disp": False, "ftol": 1e-6, "maxiter": 10000},
            )
            Phi_main = result_main.x.reshape(self.M, self.N)
>>>>>>> Stashed changes

            ## update the Phi_hat
            self.Phi_hat = Phi_main
            self.Phi_check = Phi_impute
            # Store current estimators for use as initial guesses in next round
            self.impute_prev = Phi_impute
            self.main_prev = Phi_main
        else:
            self.matching[self.t] = (
                (self.pseudo_action == self.chosen_action),
                None,
                None,
                None,
                None,
                None,
            )

    def __imputation_loss(
        self, beta: np.ndarray, X: np.ndarray, r: np.ndarray, lam: float
    ):
<<<<<<< Updated upstream
        # Reshape beta to matrix form
        beta_matrix = beta.reshape(self.M, self.N)
        # For bilinear case, we need to compute the reward for each action
        # X contains the row indices for each action, so we need to extract the corresponding rows
        # Since we're dealing with bilinear case, we need to handle this differently
        # For now, let's assume we're computing the reward for the chosen actions
        predicted_rewards = np.array([np.sum(X[i] * beta_matrix) for i in range(len(r))])
        residuals = (r - predicted_rewards) ** 2
        loss = np.sum(residuals)
        l1_norm = matrix_norm(beta_matrix, type="l1l1")
        return loss + (lam * l1_norm)
=======
        """
        Imputation loss for BiRoLFLasso.

        beta : flattened vector (M*N,) representing Φ
        X    : (t, M) matrix of user-side features
        r    : (t,) vector of observed rewards
        lam  : ℓ₁ regularization coefficient
        """
        # Reshape beta into the (M, N) matrix Φ
        Phi = beta.reshape(self.M, self.N)  # (M, N)

        # Predicted rewards using an all-ones item vector
        preds = X @ Phi @ np.ones(self.N)  # (t,)

        # Squared residuals
        residuals = (r - preds) ** 2
        loss = np.sum(residuals)

        # ℓ₁,₁-norm regularization
        l1_norm = np.sum(np.abs(beta))

        return loss + lam * l1_norm
>>>>>>> Stashed changes

    # def __main_loss(self, beta:np.ndarray, lam:float, matching_history:dict):
    #     ## matching_history : dict[t] = (bool, X, y) - bool denotes whether the matching event occurred or not
    #     loss = 0
    #     for key in matching_history:
    #         matched, X, pseudo_rewards, _, _ = matching_history[key]
    #         if matched:
    #             residuals = (pseudo_rewards - (X @ beta)) ** 2
    #             interim_loss = np.sum(residuals, axis=0)
    #         else:
    #             interim_loss = 0
    #         loss += interim_loss
    #     l1_norm = vector_norm(beta, type="l1")
    #     return loss + (lam * l1_norm)

    # matching_history: (matched,x,y,pseudo_rewards,chosen_action,r,)
    def __main_loss(self, beta: np.ndarray, lam: float, matching_history: dict):
<<<<<<< Updated upstream
        # Reshape beta to matrix form
=======
        # Reshape beta back to matrix form for computation
>>>>>>> Stashed changes
        beta_matrix = beta.reshape(self.M, self.N)
        
        # Extract matched keys and data
        matched_keys = [
            key for key, value in matching_history.items() if value[0]
        ]  # Filter matched entries

        X_list = [
            matching_history[key][1] for key in matched_keys
        ]  # List of X matrices

        Y_list = [
            matching_history[key][2] for key in matched_keys
        ]  # List of Y matrices

        pseudo_rewards_list = [
            matching_history[key][3] for key in matched_keys
        ]  # List of pseudo_rewards

        # Compute residuals for matched keys
        residuals_list = [
            (pseudo_rewards - x @ beta_matrix @ y.T) ** 2
            for x, y, pseudo_rewards in zip(X_list, Y_list, pseudo_rewards_list)
        ]

        # Sum all residuals efficiently
        residuals_sum = sum(np.sum(residuals) for residuals in residuals_list)

        # L1 regularization
        l1_norm = np.sum(np.abs(beta_matrix))

        # Total loss
        return residuals_sum + lam * l1_norm

    def __get_param(self):
        return {"param": self.Phi_hat, "impute": self.Phi_check}
