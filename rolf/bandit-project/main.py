import numpy as np
from cfg import get_cfg
from models import *
from utils import *
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_optimal(arms:np.ndarray, true_param:np.ndarray, 
                latent_param:np.ndarray, verbose:bool=True):
    _, num_arms = arms.shape
    rewards = np.zeros(shape=num_arms)
    for i in range(num_arms):
        feature = arms[:, i]
        latent_feature = np.zeros(shape=num_arms)
        latent_feature[i] = 1
        reward = np.absolute(get_reward(x=np.concatenate((feature, latent_feature)), 
                                        param=np.concatenate((true_param, latent_param)), 
                                        noise=0, 
                                        bound=(1+inherent_C)))
        rewards[i] = reward
    optimal_reward = np.max(rewards)
    optimal_arm = np.argmax(rewards)
    
    if verbose:
        print(f"True parameter: {true_param}\nInherent parameter: {latent_param}")
        print(f"Reward vector: {rewards}")
    
    return optimal_arm, optimal_reward

##################################### Main Function #####################################
if __name__ == "__main__":
    ## configuration
    cfg = get_cfg()
    
    ## define some hyper-parameters
    d = 5 # dimension of each feature vector
    k = 5 # the number of arms
    inherent_C = 1 # bound of the inherent theta (a vector of mus)
    nsim = 10
    T = 2000
    
    ## distributions
    true_dist = {
        "distribution": "gaussian",
        "params": {
            "loc": 0,
            "scale": 1
        }
    }
    
    mu_dist = {
        "distribution": "gaussian",
        "params": {
            "loc": 3,
            "scale": 2
        }
    }
    
    arm_dist = {
        "distribution": "gaussian",
        "params": {
            "loc": 2,
            "scale": 1
        }
    }
        
    ## true parameter for observable features
    true_theta = np.absolute(get_random_vector(dimension=d, 
                                               distribution=true_dist['distribution'], 
                                               params=true_dist['params'], 
                                               is_element_bounded=False, 
                                               is_norm_bounded=True, 
                                               bound=1))
    inherent_theta = np.absolute(get_random_vector(dimension=k, 
                                                   distribution=mu_dist['distribution'], 
                                                   params=mu_dist['params'], 
                                                   is_element_bounded=True, 
                                                   is_norm_bounded=False, 
                                                   bound=inherent_C))
    
    ## noises
    reward_noises = subgaussian_noise(distribution="gaussian", size=T)
    
    ## observable_arms
    observable_arms = generate_arms(distribution=arm_dist['distribution'], 
                                    params=arm_dist['params'], 
                                    dimension=d, 
                                    num_arms=k)
    
    ## inherent arms
    inherent_arms = np.identity(k)
    
    ## get optimal arm and reward
    optimal_arm, optimal_reward = get_optimal(arms=observable_arms,
                                              true_param=true_theta,
                                              latent_param=inherent_theta)
    print(f"Optimal arm: {optimal_arm}\tOptimal reward: {optimal_reward}")
    
    ## run simulation
    learning_arm = np.concatenate((observable_arms, inherent_arms), axis=0)
    delta = 0.95
    alpha = np.sqrt(0.5 * np.log(2*T*k / delta))
    learner = LinUCB(arms=learning_arm, alpha=1)
    # learner = OFUL(arms=learning_arm, delta=delta, lam=1, bound=(1+inherent_C),
    #                sigma=np.std(reward_noises, ddof=1), distribution=true_dist)
    
    # regret_list = []
    # for i in range(nsim):
    #     reward_list = []
        
    #     for t in range(T):
    #         chosen_idx = learner.choose()   # index
    #         chosen_arm = learning_arm[:, chosen_idx]
    #         reward = np.absolute(get_reward(x=chosen_arm, 
    #                                         param=np.concatenate((true_theta, inherent_theta)), 
    #                                         noise=reward_noises[t], 
    #                                         bound=(1+inherent_C)))
    #         print(f"Round {t}\tChosen arm: {chosen_idx}\tReward: {reward}")
    #         reward_list.append(reward)
    #         learner.update(a=chosen_idx, r=reward)
        
    #     regrets = [(optimal_reward - reward) for reward in reward_list]
    #     regret_list.append(regrets)
        
    # mean_regret = np.array(regret_list).mean(axis=0)
    
    reward_list = []
    for t in range(T):
        chosen_idx = learner.choose()   # index
        chosen_arm = learning_arm[:, chosen_idx]
        reward = np.absolute(get_reward(x=chosen_arm, 
                                        param=np.concatenate((true_theta, inherent_theta)), 
                                        noise=reward_noises[t], 
                                        bound=(1+inherent_C)))
        print(f"Round {t}\tChosen arm: {chosen_idx}\tReward: {reward}\tRegret: {optimal_reward - reward}")
        reward_list.append(reward)
        learner.update(a=chosen_idx, r=reward)
    regrets = [(optimal_reward - reward) for reward in reward_list]
    
    plt.figure(figsize=(8, 6))
    plt.plot(np.cumsum(regrets))
    plt.grid(True)
    plt.xlabel('Rounds')
    plt.ylabel('Regret')
    plt.title(f"Regret of the {learner.__class__.__name__}")
    plt.savefig(f"./{learner.__class__.__name__}.png")
