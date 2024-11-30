from env import *
from neuralucb import *
from neuralts import *
import matplotlib.pyplot as plt
import torch
from tqdm import trange
from util import *

M = 50000 # action space size
d = 10 # dimension of "mapped" features
k = 8 # dimension of "latent" features
# half=$((k / 2))
# almost=$((k - 1))
# feat_bound=1
sigma = 0.1
# param_bound=1
T = 10000 # total horizon
K = 20
delta=0.00001
TRIALS = 10 # number of trials
lamb = 1
SEED = 103
reward_noise_dist, reward_noise_std = "gaussian", 0.1
m = 1
func_type = "glm"
learning_rates = [0.035, 0.03, 0.025, 0.01, 9e-4, 7e-4, 8e-4]

if __name__ == "__main__":
    for lr in learning_rates:
        fig, ax = plt.subplots(figsize=(6, 4))
        env = Env(k=k, d=d, arms=K, action_space=M, 
                seed=SEED, func_type=func_type, bias_dist="gaussian", 
                num_visibles=m, check_specs=True)

        neuralucb = NeuralUCB(env.d, env.arms, beta=1, lamb=1, lr=lr)
        NeuralUCBregret = [0]
        played_optimal = 0
        for i in trange(T):
            context, optimal_arm, expected_reward = env.step()
            reward_noise = subgaussian_noise(distribution=reward_noise_dist, size=env.arms, 
                                             std=reward_noise_std, random_state=SEED+i)
            reward = expected_reward + reward_noise
            arm = neuralucb.take_action(context)
            NeuralUCBregret += [NeuralUCBregret[-1] + expected_reward[optimal_arm] - expected_reward[arm]]
            played_optimal += (arm == optimal_arm)
            neuralucb.update(context, arm, reward[arm])

        ax.plot(NeuralUCBregret, label=f'NeuralUCB')
        print(f'NeuralUCB, lr={lr}, Regret={NeuralUCBregret[-1]}, Played optimal={played_optimal/T}')

        p_neuralucb = PartialNeuralUCB(env.d, env.arms, beta=1, lamb=1, lr=lr)
        PartialNeuralUCBregret = [0]
        played_optimal = 0
        for i in trange(T):
            context, optimal_arm, expected_reward = env.step()
            context = np.concatenate([context, np.identity(env.arms)], axis=1)
            reward_noise = subgaussian_noise(distribution=reward_noise_dist, size=env.arms, 
                                            std=reward_noise_std, random_state=SEED+i)
            reward = expected_reward + reward_noise
            arm = p_neuralucb.take_action(context)
            PartialNeuralUCBregret += [PartialNeuralUCBregret[-1] + expected_reward[optimal_arm] - expected_reward[arm]]
            played_optimal += (arm == optimal_arm)
            p_neuralucb.update(context, arm, reward[arm])

        ax.plot(PartialNeuralUCBregret, label=f"lr={lr}")
        print(f'NeuralUCB-PO, lr={lr}, Regret={PartialNeuralUCBregret[-1]}, Played optimal={played_optimal/T}')

        # neuralts = NeuralTS(env.d, env.arms, beta=1, lamb=1)
        # NeuralTSregret = [0]


        # for i in trange(T):
        #     context, optimal_arm, expected_reward = env.step()
        #     reward_noise = subgaussian_noise(distribution=reward_noise_dist, size=env.arms, 
        #                                      std=reward_noise_std, random_state=SEED+i)
        #     reward = expected_reward + reward_noise
        #     arm = neuralts.take_action(context)
        #     NeuralTSregret += [NeuralTSregret[-1] + expected_reward[optimal_arm] - expected_reward[arm]]
        #     neuralts.update(context, arm, reward[arm])

        # ax.plot(NeuralTSregret, label='NeuralTS')
        # print('neuralts:', NeuralTSregret[-1])

        ax.grid()
        ax.legend()
        save_plot(fig, path="./", fname=f"result_num_lr_{lr}_visibles={m}_func_type_{func_type}")