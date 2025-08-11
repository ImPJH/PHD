import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from neural_exploration import *

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def quadratic(x):
    return x ** 2

def cosine(x):
    return np.cos(2 * np.pi * x)

def show_result(regrets:dict, horizon:int, reward_title:str, num_visibles:int, figsize:tuple=(6, 5)):
    fig, ax = plt.subplots(figsize=figsize)

    # 마커 스타일과 색상 설정
    period = horizon // 10
    markers = ['o', 's', '^', 'd', 'p']
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    title = r"$m$"

    # 각 알고리즘에 대해 에러바와 함께 그래프 그리기
    keys = reversed(list(regrets.keys()))
    for (marker, color), key in zip(zip(markers, colors), keys):
        rounds = np.arange(horizon)
        mean = np.mean(regrets[key], axis=0)
        std = np.std(regrets[key], axis=0, ddof=1)
        
        # 마커와 에러 바가 있는 라인을 주기적으로 표시
        ax.errorbar(rounds[::period], mean[::period], yerr=std[::period], label=key, 
                    fmt=marker, color=color, capsize=3, elinewidth=1)
        
        # 주기적인 마커 없이 전체 라인을 표시
        ax.plot(rounds, mean, color=color, linewidth=2)

    ax.grid()
    ax.set_xlabel(r"Round ($t$)")
    ax.set_ylabel("Cumulative Regret")
    ax.set_title(f"{reward_title}, {title}={num_visibles}")
    ax.legend()

    fig.tight_layout() 
    return fig

def save_plot(fig:Figure, path:str, fname:str, ext:str="pdf"):
    os.makedirs(path, exist_ok=True)
    fig.savefig(f"{path}/{fname}.{ext}")
    print("Plot is Saved Completely!")

if __name__ == "__main__":
    T = int(3000)
    n_arms = 20
    n_obs_features = 16
    n_latent_features = 12
    num_visibles = [n_latent_features//2]
    noise_std = 0.1

    confidence_scaling_factor = noise_std

    n_sim = 10

    SEED = 103
    np.random.seed(SEED)

    p = 0.2
    train_every = 10
    use_cuda = False

    PARAM_DICT = {
        1: {
            'hidden_size': 32,
            'epochs': 40,
            'reg_factor': 2.0,
            'lr': 0.005
        },
        n_latent_features//2 : {
            'hidden_size': 48,
            'epochs': 40,
            'reg_factor': 2.0,
            'lr': 0.003
        },
        n_latent_features-1: {
            'hidden_size': 32,
            'epochs': 40,
            'reg_factor': 2.0,
            'lr': 0.005
        }
    }
    
    for m in num_visibles:
        print(f"Mapped features : {m}")
        hidden_size = PARAM_DICT[m]['hidden_size']
        epochs = PARAM_DICT[m]['epochs']
        ### mean reward function
        a = np.random.randn(n_latent_features)
        a /= np.linalg.norm(a, ord=2)
        a = a[:m]
        reward_func = lambda x: quadratic(np.dot(a, x))
        
        regret_dict = dict()

        print("NeuralUCB-PO")
        bandit = ContextualBandit(T=T, n_arms=n_arms, n_obs_features=n_obs_features, 
                                n_latent_features=n_latent_features,
                                h=reward_func, num_visibles=m,
                                noise_std=noise_std, seed=SEED, is_partial=True)

        regrets = np.empty((n_sim, T))

        for i in range(n_sim):
            bandit.reset_rewards()
            model = NeuralUCB_PO(bandit, hidden_size=hidden_size, reg_factor=PARAM_DICT[m]['reg_factor'],
                                 delta=0.1, confidence_scaling_factor=confidence_scaling_factor*0.5,
                                 training_window=100, p=p, learning_rate=PARAM_DICT[m]['lr'],
                                 epochs=epochs, train_every=train_every, use_cuda=use_cuda)
                
            model.run()
            regrets[i] = np.cumsum(model.regrets)
            
        regret_dict['NeuralUCB-PO'] = regrets

        print("NeuralUCB")
        bandit = ContextualBandit(T=T, n_arms=n_arms, n_obs_features=n_obs_features, 
                                n_latent_features=n_latent_features,
                                h=reward_func, num_visibles=m,
                                noise_std=noise_std, seed=SEED)

        regrets = np.empty((n_sim, T))

        for i in range(n_sim):
            bandit.reset_rewards()
            model = NeuralUCB(bandit, hidden_size=hidden_size, reg_factor=PARAM_DICT[m]['reg_factor'],
                              delta=0.1, confidence_scaling_factor=confidence_scaling_factor*0.5,
                              training_window=100, p=p, learning_rate=PARAM_DICT[m]['lr'],
                              epochs=epochs, train_every=train_every, use_cuda=use_cuda)
                
            model.run()
            regrets[i] = np.cumsum(model.regrets)

        regret_dict['NeuralUCB'] = regrets
        
        fig = show_result(regrets=regret_dict, horizon=T, reward_title=r"$f_2(\mathbf{x}) = (\mathbf{x}^\top\mathbf{a})^2$", num_visibles=m)
        fname = f"seed_{SEED}_quadratic_m={m}_final_v2"
        save_plot(fig, path=".", fname=fname)