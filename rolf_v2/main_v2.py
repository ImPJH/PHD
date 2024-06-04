from cfg import get_cfg
from models_v2 import *
from util import *

# MOTHER_PATH = "/home/sungwoopark/bandit-research/rolf"
MOTHER_PATH = "."

DIST_DICT = {
    "gaussian": "g",
    "uniform": "u"
}

AGENT_DICT = {
    "mab_ucb": "MAB-UCB",
    "linucb": "LinUCB",
    "lints": "LinTS",
    "rolf_lasso": "RoLF-Lasso",
    "rolf_ridge": "RoLF-Ridge"
}

cfg = get_cfg()

def run_trials(agent_type:str, trials:int, horizon:int, d:int, arms:int, noise_std:float, random_state:int, verbose:bool):
    exp_map = {
        "double": (2 * arms),
        "sqr": (arms ** 2),
        "K": arms,
        "triple": (3 * arms),
        "quad": (4 * arms)
    }
    regret_container = np.zeros(trials, dtype=object)
    for trial in range(trials):
        if random_state is not None:
            random_state_ = random_state + (713317 * trial)
        else:
            random_state_ = None

        if agent_type == "linucb":
            agent = LinUCB(d=d, lbda=cfg.p, delta=cfg.delta)
        elif agent_type == "lints":
            agent = LinTS(d=d, lbda=cfg.p, horizon=horizon, reward_std=noise_std, delta=cfg.delta)
        elif agent_type == "mab_ucb":
            agent = UCBDelta(n_arms=arms, delta=cfg.delta)
        elif agent_type == "rolf_lasso":
            if cfg.explore:
                agent = RoLFLasso(d=d, arms=arms, p=cfg.p, delta=cfg.delta, sigma=noise_std, random_state=random_state_, 
                                  explore=cfg.explore, init_explore=exp_map[cfg.init_explore])
            else:
                agent = RoLFLasso(d=d, arms=arms, p=cfg.p, delta=cfg.delta, sigma=noise_std, random_state=random_state_)                
        elif agent_type == "rolf_ridge":
            if cfg.explore:
                agent = RoLFRidge(d=d, arms=arms, p=cfg.p, delta=cfg.delta, sigma=noise_std, random_state=random_state_,
                                  explore=cfg.explore, init_explore=exp_map[cfg.init_explore])
            else:
                agent = RoLFRidge(d=d, arms=arms, p=cfg.p, delta=cfg.delta, sigma=noise_std, random_state=random_state_)

        ## sample the observable features and orthogonal basis, then augment the feature factor
        X = feature_sampler(dimension=d, feat_dist=cfg.feat_dist, size=arms, disjoint=cfg.feat_disjoint, 
                            cov_dist=cfg.feat_cov_dist, bound=cfg.feat_feature_bound, bound_method=cfg.feat_bound_method, 
                            bound_type=cfg.feat_bound_type, uniform_rng=cfg.feat_uniform_rng, random_state=random_state_).T
        basis = orthogonal_complement_basis(X) # (K, K-d) matrix and each column vector denotes the orthogonal basis
        x_aug = np.hstack((X.T, basis)) # augmented into (K, K) matrix and each row vector denotes the augmented feature
        bounding(type="feature", v=x_aug, bound=cfg.feat_feature_bound, method=cfg.feat_bound_method, norm_type="lsup")

        ## sample reward parameter after augmentation and compute the expected rewards
        reward_param = param_generator(dimension=arms, distribution=cfg.param_dist, disjoint=cfg.param_disjoint, bound=cfg.param_bound, 
                                       bound_type=cfg.param_bound_type, uniform_rng=cfg.param_uniform_rng, random_state=random_state_)
        exp_rewards = x_aug @ reward_param # (K, ) vector

        ## run and collect the regrets
        if isinstance(agent, LinUCB) or isinstance(agent, LinTS):
            data = X.T
        else:
            data = x_aug
        print(data.shape)
        regrets = run(trial=trial, agent=agent, horizon=horizon, exp_rewards=exp_rewards, x=data, 
                      noise_dist=cfg.reward_dist, noise_std=noise_std, random_state=random_state_, verbose=verbose)
        regret_container[trial] = regrets
    return regret_container


def run(trial:int, agent:Union[MAB, ContextualBandit], horizon:int, exp_rewards:np.ndarray, 
        x:np.ndarray, noise_dist:str, noise_std:float, random_state:int, verbose:bool):
    # x: augmented feature if the agent is RoLF (K, K)
    regrets = np.zeros(horizon, dtype=float)

    if not verbose:
        bar = tqdm(range(horizon))
    else:
        bar = range(horizon)

    for t in bar:
        if random_state is not None:
            random_state_ = random_state + int(313 * t)
        else:
            random_state_ = None

        if t == 0:
            print(f"Number of actions : {x.shape[0]}\tReward range : [{np.amin(exp_rewards):.5f}, {np.amax(exp_rewards):.5f}]")
        
        ## compute the optimal action
        optimal_action = np.argmax(exp_rewards)
        optimal_reward = exp_rewards[optimal_action]

        ## choose the best action
        noise = subgaussian_noise(distribution=noise_dist, size=1, std=noise_std, random_state=random_state_)
        if isinstance(agent, ContextualBandit):
            chosen_action = agent.choose(x)
        else:
            chosen_action = agent.choose()
        chosen_reward = exp_rewards[chosen_action] + noise
        if verbose:
            try:
                print(f"Trial : {trial}, p : {cfg.p}, Agent: {agent.__class__.__name__}, Round: {t+1}\toptimal : {optimal_action}\ta_hat: {agent.a_hat}\tpseudo : {agent.pseudo_action}\tchosen : {agent.chosen_action}\t")
            except:
                print(f"Trial : {trial}, p : {cfg.p}, Agent: {agent.__class__.__name__}, Round: {t+1}\toptimal : {optimal_action}\tchosen : {chosen_action}")

        ## compute the regret
        regrets[t] = optimal_reward - exp_rewards[chosen_action]

        ## update the agent
        if isinstance(agent, ContextualBandit):
            agent.update(x=x, r=chosen_reward)
        else:
            agent.update(a=chosen_action, r=chosen_reward)

    return np.cumsum(regrets)


def show_result(regrets:dict, horizon:int, arms:int, figsize:tuple=(6, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    period = horizon // 10
    
    # 각 알고리즘에 대해 에러바와 함께 그래프 그리기
    for color, (key, item) in zip(colors, regrets.items()):
        rounds = np.arange(horizon)
        mean = np.mean(item, axis=0)
        std = np.std(item, axis=0, ddof=1)
        
        # 마커와 에러 바가 있는 라인을 주기적으로 표시
        ax.errorbar(rounds[::period], mean[::period], yerr=std[::period], label=f"{key}", 
                    fmt='s', color=color, capsize=3, elinewidth=1)
        
        # 주기적인 마커 없이 전체 라인을 표시
        ax.plot(rounds, mean, color=color, linewidth=2)
    
    ax.grid(True)
    ax.set_xlabel(r"Round ($t$)")
    ax.set_ylabel("Cumulative Regret")
    # ax.set_title(fr"$K$ = {arms}")
    ax.legend()
    
    fig.tight_layout()  
    # fig.tight_layout(rect=[0, 0, 1, 0.95])
    # fig.suptitle(f"$Z${FEAT_DICT[(feat_dist_label, feat_disjoint)]}, $\sigma_\eta=${context_label}, $\sigma_\epsilon=${reward_label}, seed={SEED}, num_visibles={cfg.num_visibles}")
    return fig


if __name__ == "__main__":
    ## hyper-parameters
    arms = cfg.arms # List[int]
    d = cfg.dim
    T = cfg.horizon
    SEED = cfg.seed
    AGENTS = ["rolf_lasso", "rolf_ridge", "linucb", "lints", "mab_ucb"]
    # AGENTS = ["linucb", "lints"]

    RESULT_PATH = f"{MOTHER_PATH}/results_v2/p_{cfg.p}"
    FIGURE_PATH = f"{MOTHER_PATH}/figures_v2/p_{cfg.p}"
   
    regret_results = dict()
    for agent_type in AGENTS:
        regrets = run_trials(agent_type=agent_type, trials=cfg.trials, horizon=T, d=d, arms=arms, 
                             noise_std=cfg.reward_std, random_state=SEED, verbose=True)
        key = AGENT_DICT[agent_type]
        regret_results[key] = regrets
    
    fname = f"Seed_{SEED}_K_{arms}_d_{d}_T_{T}_p_{cfg.p}_delta_{cfg.delta}_explored_{cfg.init_explore}_param_{DIST_DICT[cfg.param_dist]}"
    fig = show_result(regrets=regret_results, horizon=T, arms=arms)
    save_plot(fig, path=FIGURE_PATH, fname=fname)