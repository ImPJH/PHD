from cfg import get_cfg
from models import *
from util import *

# MOTHER_PATH = "/home/sungwoopark/bandit-research/rolf"
MOTHER_PATH = "."

DIST_DICT = {
    "gaussian": "g",
    "uniform": "u"
}

AGENT_DICT = {
    "mab_egreedy": r"MAB-$\epsilon$-greedy",
    "mab_ucb": "MAB-UCB",
    "rolf": "RoLF"
}

cfg = get_cfg()

def run_trials(agent_type:str, trials:int, horizon:int, x:np.ndarray, noise_dist:str, noise_std:float, 
               feat_bound:float, feat_bound_method:str, feat_bound_type:str, random_state:int, verbose:bool):
    # x: non-augmented feature (d, K) - each column denotes the feature vector
    d, arms = x.shape
    regret_container = np.zeros(trials, dtype=object)

    for trial in range(trials):
        if agent_type == "mab_egreedy":
            agent = eGreedyMAB(n_arms=arms, epsilon=cfg.epsilon)
        elif agent_type == "mab_ucb":
            agent = UCBDelta(n_arms=arms, delta=cfg.delta)
        else:
            agent = RoLF(d=d, arms=arms, p=cfg.p, delta=cfg.delta, sigma=noise_std)
        
        if random_state is not None:
            random_state_ = random_state + (7137 * (trial+1))
        else:
            random_state_ = None

        ## sample the basis and augment the feature factor
        basis = orthogonal_complement_basis(x) # (K, K-d) matrix and each column vector denotes the orthogonal basis
        x_aug = np.hstack((x.T, basis)) # augmented into (K, K) matrix and each row vector denotes the augmented feature

        ## sample reward parameter after augmentation and compute the expected rewards
        reward_param = param_generator(dimension=arms, distribution=cfg.param_dist, disjoint=cfg.param_disjoint, bound=cfg.param_bound, 
                                       bound_type=cfg.param_bound_type, uniform_rng=cfg.param_uniform_rng, random_state=random_state_)
        exp_rewards = x_aug @ reward_param # (K, ) vector

        ## run and collect the regrets
        regrets = run(agent=agent, horizon=horizon, exp_rewards=exp_rewards, x=x_aug, noise_dist=noise_dist, noise_std=noise_std, feat_bound=feat_bound, 
                      feat_bound_method=feat_bound_method, feat_bound_type=feat_bound_type, random_state=random_state_, verbose=verbose)
        regret_container[trial] = regrets
    return regret_container


def run(agent:Union[MAB, ContextualBandit], horizon:int, exp_rewards:np.ndarray, x:np.ndarray, noise_dist:str, noise_std:float, 
        feat_bound:float, feat_bound_method:str, feat_bound_type:str, random_state:int, verbose:bool):
    # x: augmented feature if the agent is RoLF (K, K)
    arms = x.shape[0]

    regrets = np.zeros(horizon, dtype=float)

    if not verbose:
        bar = tqdm(range(horizon))
    else:
        bar = range(horizon)

    for t in bar:
        if random_state is not None:
            random_state_ = random_state + int(3113 * t)
            np.random.seed(random_state_)
        else:
            random_state_ = None

        if t == 0:
            print(f"Number of actions : {x.shape[0]}\tReward range : [{np.amin(exp_rewards):.5f}, {np.amax(exp_rewards):.5f}]")
        
        ## bound the action set
        if feat_bound_method is not None:
            bounding(type="feature", v=x, bound=feat_bound, method=feat_bound_method, norm_type=feat_bound_type)
        
        ## sample the reward noise and compute the reward
        reward_noise = subgaussian_noise(distribution=noise_dist, size=arms, std=noise_std, random_state=random_state_)
        rewards = exp_rewards + reward_noise

        ## compute the optimal action and the best action
        optimal_action = np.argmax(exp_rewards)
        # print(f"optimal action : {optimal_action}")
        optimal_reward = exp_rewards[optimal_action]
        if isinstance(agent, ContextualBandit):
            chosen_action = agent.choose(x)
        else:
            chosen_action = agent.choose()
        chosen_reward = rewards[chosen_action]

        ## compute the regret
        regrets[t] = optimal_reward - exp_rewards[chosen_action]

        ## update the agent
        if isinstance(agent, ContextualBandit):
            agent.update(x=x, r=chosen_reward)
        else:
            agent.update(a=chosen_action, r=chosen_reward)

    return np.cumsum(regrets)


def show_result(regrets:dict, horizon:int, label_name:str, figsize:tuple=(6, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    period = horizon // 10
    
    # 각 알고리즘에 대해 에러바와 함께 그래프 그리기
    for color, (key, item) in zip(colors, regrets.items()):
        rounds = np.arange(horizon)
        mean = np.mean(item, axis=0)
        std = np.std(item, axis=0, ddof=1)
        
        # 마커와 에러 바가 있는 라인을 주기적으로 표시
        ax.errorbar(rounds[::period], mean[::period], yerr=std[::period], label=f"{label_name}={key}", 
                    fmt='s', color=color, capsize=3, elinewidth=1)
        
        # 주기적인 마커 없이 전체 라인을 표시
        ax.plot(rounds, mean, color=color, linewidth=2)
    
    ax.grid(True)
    ax.set_xlabel(r"Round ($t$)")
    ax.set_ylabel("Cumulative Regret")
    # ax.set_title(f"{bias_label}{label_dict[cfg.bias_dist]}, {param_label}{label_dict[cfg.param_dist]}, {title}")
    ax.legend()
    
    fig.tight_layout()  
    # fig.tight_layout(rect=[0, 0, 1, 0.95])
    # fig.suptitle(f"$Z${FEAT_DICT[(feat_dist_label, feat_disjoint)]}, $\sigma_\eta=${context_label}, $\sigma_\epsilon=${reward_label}, seed={SEED}, num_visibles={cfg.num_visibles}")
    return fig


if __name__ == "__main__":
    ## hyper-parameters
    num_actions = cfg.num_actions # List[int]
    d = cfg.obs_dim
    T = cfg.horizon
    SEED = cfg.seed
    # AGENTS = ["rolf", "mab_ucb", "mab_egreedy"]
    AGENTS = ["rolf"]

    RESULT_PATH = f"{MOTHER_PATH}/results"
    FIGURE_PATH = f"{MOTHER_PATH}/figures"

    ## generate the observable features
    X = feature_sampler(dimension=d, feat_dist=cfg.feat_dist, size=num_actions, disjoint=cfg.feat_disjoint, cov_dist=cfg.feat_cov_dist, bound=cfg.feat_feature_bound, 
                        bound_method=cfg.feat_bound_method, bound_type=cfg.feat_bound_type, uniform_rng=cfg.feat_uniform_rng, random_state=SEED).T
    
    regret_results = dict()
    for agent_type in AGENTS:
        print(f"Agent Type : {agent_type}")
        regrets = run_trials(agent_type=agent_type, trials=cfg.trials, horizon=T, x=X, noise_dist="gaussian", 
                             noise_std=cfg.reward_std, feat_bound=cfg.feat_feature_bound, feat_bound_method=cfg.feat_bound_method, 
                             feat_bound_type=cfg.feat_bound_type, random_state=SEED, verbose=False)
        key = AGENT_DICT[agent_type]
        regret_results[key] = regrets
    
    fname = f"{SEED}_K_{num_actions}_d_{d}_feat_{DIST_DICT[cfg.feat_dist]}_param_{DIST_DICT[cfg.param_dist]}_{datetime.now()}"
    fig = show_result(regrets=regret_results, horizon=T, label_name="Agent Type")
    save_plot(fig, path=FIGURE_PATH, fname=fname)