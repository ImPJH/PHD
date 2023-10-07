import os
import numpy as np
from tqdm.auto import tqdm
from typing import List
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from datetime import datetime
from get_primes import get_primes

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

###################################### Learning Algorithm ######################################
class UCB:
    def __init__(self, arms, sigma, delta=0.1, alpha=1.0):
        self.arms = arms  # number of arms
        self.t = 0  # total play count
        self.sigma = sigma
        self.n = np.zeros(arms, dtype=int)  # play count for each arm
        self.mean_reward = np.zeros(arms, dtype=float)  # mean reward for each arm
        self.delta = delta
        self.alpha = alpha

    def choose(self):
        # Increment the playing round
        self.t += 1
        
        # Play each arm once first
        for i in range(self.arms):
            if self.n[i] == 0:
                return i

        ucb_values = np.zeros(self.arms, dtype=float)
        for arm in range(self.arms):
            bonus = self.sigma * np.sqrt((2 * np.log(self.t / self.delta)) / float(self.n[arm]))
            ucb_values[arm] = self.mean_reward[arm] + (bonus * self.alpha)
            
        max_value = np.amax(ucb_values)
        argmaxes, = np.where(ucb_values == max_value)

        return np.random.choice(argmaxes)

    def update(self, chosen_arm, reward):
        self.n[chosen_arm] += 1
        n = self.n[chosen_arm]

        # Update the mean reward for the chosen arm
        value = self.mean_reward[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.mean_reward[chosen_arm] = new_value
################################################################################################

def save_plot(fig:Figure, path:str, fname:str) -> None:
    os.makedirs(path, exist_ok=True)
    fig.savefig(f"{path}/{fname}.png")
    print("Plot is Saved Completely!")

def run(horizon:int, arms:List[GaussianArm], agent:UCB, use_tqdm:bool=True) -> np.ndarray:
    if use_tqdm:
        bar = tqdm(range(horizon))
    else:
        bar = range(horizon)
    
    regrets = np.zeros(horizon, dtype=float)
    
    expected_reward = np.array([arm.mu for arm in arms])
    optimal_reward = np.amax(expected_reward)
    for t in bar:
        chosen_arm = agent.choose()
        chosen_reward = arms[chosen_arm].draw()
        regrets[t] = optimal_reward - expected_reward[chosen_arm]
        agent.update(chosen_arm=chosen_arm, reward=chosen_reward)
    return np.cumsum(regrets)

def run_trials(trials:int, horizon:int, n_arms:int, sigma:float, alpha:float=1.0, random_state:int=None) -> np.ndarray:
    regret_container = np.zeros(trials, dtype=object)
    for trial in range(trials):
        random_state_ = random_state + (11111*(trial+1))
        # parameter sampling
        if random_state is not None:
            np.random.seed(random_state_)
        mus = np.random.uniform(low=-1., high=1., size=n_arms)
        arms = [GaussianArm(mu, sigma) for mu in mus]
        print(f"Trial {trial+1}\tReward range : [{np.amin(mus):.5f}, {np.amax(mus):.5f}]")
        
        # agent initializing
        agent = UCB(arms=n_arms, sigma=sigma)
        regrets = run(horizon=horizon, arms=arms, agent=agent)
        regret_container[trial] = regrets
    return regret_container

def show_result(regrets:dict, horizon:int, label_name:str, arm_name:str, seed:int, figsize:tuple=(13, 5)) -> Figure:
    NROWS, NCOLS = 1, 2
    fig, (ax1, ax2) = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=figsize)
    
    for key in regrets:
        item = regrets[key]
        ax1.plot(np.mean(item, axis=0), label=f"{label_name}={key}")
    ax1.grid(True)
    ax1.set_xlabel("Round")
    ax1.set_ylabel(r"$R_t$")
    ax1.set_title(r"$\bar{R}_t$")
    ax1.legend()
    
    for key in regrets:
        item = regrets[key]
        mean = np.mean(item, axis=0)
        std = np.std(item, axis=0, ddof=1)
        ax2.plot(mean, label=f"{label_name}={key}")
        ax2.fill_between(np.arange(horizon), mean-std, mean+std, alpha=0.2)
    ax2.grid(True)
    ax2.set_xlabel("Round")
    ax2.set_ylabel(r"$R_t$")
    ax2.set_title(r"$\bar{R}_t\pm 1SD$")
    ax2.legend()
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(f"UCB with {arm_name}, seed = {seed}")
    return fig


if __name__ == "__main__":
    T = 30000
    TRIALS = 10
    SIGMA = 0.1
    ALPHA = 0.3
    # SEED = get_primes(start=300, end=1500)
    SEED = [i for i in range(350, 400)]
    PATH = "/home/sungwoopark/bandit-research/latent-contextual-bandit/modules/seed_comparison/figures/UCB/arms"
    
    for seed in SEED:
        num_actions = [5] + [10*(i+1) for i in range(4)]
        regret_results = dict()
        for n_arms in num_actions:
            print(f"SEED = {seed}\t|A_t| = {n_arms}")
            random_state = seed + (11111*n_arms)
            regrets = run_trials(trials=TRIALS, horizon=T, n_arms=n_arms, sigma=SIGMA, alpha=ALPHA, random_state=random_state)
            regret_results[n_arms] = regrets
            
        label_name = r"$\vert \mathcal{A}_t\vert$"
        fig = show_result(regrets=regret_results, horizon=T, label_name=label_name, seed=seed, arm_name="Gaussian Arms")
        fname = f"UCB_seed_{seed}_{datetime.now()}"
        save_plot(fig=fig, path=PATH, fname=fname)
    