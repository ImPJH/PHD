from cfg import get_cfg
from models import *
from util import *

RESULT_PATH = "./results"
FIGURE_PATH = "./figures"

FEAT_DICT = {
    ("gaussian", True): "$\sim N(0, I)$",
    ("gaussian", False): "$\sim N(0, \Sigma)$",
    ("uniform", True): "$\sim Unif_{I}$",
    ("uniform", False): "$\sim Unif_{\Sigma}$"
}


def run(mode:str, agent:Union[LinUCB, LineGreedy, PartialLinUCB], num_actions:int, horizon:int, obs:np.ndarray, 
        latent:np.ndarray, reward_params:np.ndarray, reward_noise_var:float, use_tqdm:bool, verbose:bool, random_state:int):
    action_space_size, _ = obs.shape
    assert action_space_size >= num_actions, "The cardinality of the entire action space must be larger than the number of actions."
    
    if use_tqdm:
        bar = tqdm(range(horizon))
    else:
        bar = range(horizon)
    
    regrets = np.zeros(horizon)
    for t in bar:
        if action_space_size == num_actions:
            is_fixed = True
            ## fixed action space
            indices = np.arange(num_actions)
            if mode == "partial":
                ## here the random state must be different to that used for generating reward parameters
                inherent_rewards = param_generator(dimension=num_actions, distribution=cfg.param_dist, disjoint=cfg.param_disjoint, 
                                                   uniform_rng=cfg.param_uniform_rng, random_state=random_state)
            else:
                inherent_rewards = 0.
            
        else:
            is_fixed = False
            indices = np.random.randint(0, action_space_size, num_actions)
            if mode == "partial":
                ## here the random state must be different to that used for generating reward parameters
                ## additionally, as arms are newly sampled at each round, so are their inherent rewards
                ## thus the iterator, t, is added to the seed
                inherent_rewards = param_generator(dimension=num_actions, distribution=cfg.param_dist, disjoint=cfg.param_disjoint, 
                                                   uniform_rng=cfg.param_uniform_rng, random_state=random_state+t)
            else:
                inherent_rewards = 0.
                
        ## generate the reward noise (num_actions, )
        reward_noise = subgaussian_noise(distribution="gaussian", size=num_actions, random_state=random_state+t, std=reward_noise_var)

        if t == 0:
            print(f"Mode: {mode}\tInherent Rewards: {inherent_rewards}\tFixed Arm Set: {is_fixed}")
        
        ## observe the actions (num_actions, d), (num_actions, k or m)
        action_set, latent_action_set = obs[indices], latent[indices]
        if mode == "partial":
            onehot = np.identity(num_actions)
            action_set = np.c_[action_set, onehot]
        
        ## compute the rewards and the optimal actions
        expected_rewards = latent_action_set @ reward_params + inherent_rewards
        true_rewards = expected_rewards + reward_noise
        if t == 0:
            print(f"Reward range = [{np.amin(expected_rewards):.4f}, {np.amax(expected_rewards):.4f}]")
        optimal_action = np.argmax(expected_rewards)
        optimal_reward = expected_rewards[optimal_action]
        
        ## choose the best action
        chosen_action = agent.choose(action_set)
        chosen_reward = true_rewards[chosen_action]
        chosen_context = action_set[chosen_action]
        
        ## compute the regret
        instance_regret = optimal_reward - expected_rewards[chosen_action]
        regrets[t] = instance_regret
        
        ## update the agent
        agent.update(chosen_context, chosen_reward)
        
        if verbose: 
            print(f"round {t+1}\toptimal action : {optimal_action}\toptimal reward : {optimal_reward:.3f}")
            print(f"\tchosen action : {chosen_action}\trealized reward : {chosen_reward:.3f}, expected reward: {expected_rewards[chosen_action]:.3f}")
            print(f"\tinstance regret : {instance_regret:.3f}, cumulative regret : {np.sum(regrets):.3f}")
            
    return regrets


def show_rewards(exp_rewards:np.ndarray, true_rewards:np.ndarray):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(11, 5))

    ax1.hist(exp_rewards)
    ax1.grid(True)
    ax1.set_xlabel("$\mathbb{E}[T]$")
    
    ax2.hist(true_rewards)
    ax2.grid(True)
    ax2.set_xlabel("$\mathbb{E}[T]+\epsilon_t$")
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    exp_label = "$\mathbb{E}[T]$"
    fig.suptitle(f"{exp_label}$\in$[{np.amin(exp_rewards):.4f}, {np.amax(exp_rewards):.4f}]")
    return fig


def show_result(result:dict, label_name:str, feat_dist_label:str, feat_disjoint:bool, context_label:str, reward_label:str):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    for key, value in result.items():
        mean = np.mean(value, axis=0)
        ax1.plot(mean, label=f"{label_name}={key}")
    ax1.grid(True)
    ax1.set_xlabel("Round")
    ax1.set_ylabel("$R_T$")
    ax1.legend()
    
    for key, value in result.items():
        mean = np.mean(value, axis=0)
        std = np.std(value, axis=0, ddof=1)
        ax2.plot(mean, label=f"{label_name}={key}")
        ax2.fill_between(np.arange(T), mean-std, mean+std, alpha=0.2)
    ax2.grid(True)
    ax2.set_xlabel("Round")
    ax2.set_ylabel("$R_T$")
    ax2.legend()
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(f"Z{FEAT_DICT[(feat_dist_label, feat_disjoint)]}, $\sigma_\eta=${context_label}, $\sigma_\epsilon=${reward_label}, bound={cfg.bound_method}")    
    return fig


if __name__ == "__main__":
    ## import argumens from terminal
    cfg = get_cfg()
    
    ## hyper-parameters
    trials = cfg.trials
    mode = cfg.mode
    action_space_size = cfg.action_space_size
    num_actions = cfg.num_actions
    d = cfg.obs_dim
    k = cfg.latent_dim
    m = cfg.num_visibles
    T = cfg.horizon
    GEN_SEED, RUN_SEED = cfg.seed
    ALPHAS = cfg.alphas
    ARMS = cfg.arms
    if "T" in cfg.context_var:
        exp = cfg.context_var.split("T")[-1]
        context_var = T ** float(exp)
        context_label = f"$T^{float(exp)}$"
    else:
        context_var = float(cfg.context_var)
        context_label = cfg.context_var
    
    ## generate the latent variable whose dimension is (action_space_size, k)
    Z = feature_sampler(dimension=k, feat_dist=cfg.feat_dist, size=action_space_size, disjoint=cfg.feat_disjoint, cov_dist=cfg.feat_cov_dist, 
                        bound=cfg.latent_feature_bound, bound_method=cfg.bound_method, uniform_rng=cfg.feat_uniform_rng, random_state=GEN_SEED)
       
    ## generate the context noise of shape (action_space_size, d)
    context_noise = subgaussian_noise(distribution="gaussian", size=action_space_size*d, random_state=GEN_SEED, std=context_var).reshape(action_space_size, d)
    
    ## generate the observable context
    if mode == "full":
        ## generate the decoder mapping A : Z -> X (d, k), X (action_space_size, d), and theta (k, )
        A = mapping_generator(latent_dim=k, obs_dim=d, distribution=cfg.map_dist, lower_bound=cfg.map_lower_bound, 
                              upper_bound=cfg.map_upper_bound, uniform_rng=cfg.map_uniform_rng, random_state=GEN_SEED)
        X = Z @ A.T + context_noise # (action_space_size, d)
        theta = param_generator(dimension=k, distribution=cfg.param_dist, disjoint=cfg.param_disjoint, 
                                uniform_rng=cfg.param_uniform_rng, random_state=GEN_SEED)
    else:
        ## generate the decoder mapping A : Z_visible -> X (d, m)
        A = mapping_generator(latent_dim=m, obs_dim=d, distribution=cfg.map_dist, lower_bound=cfg.map_lower_bound, 
                              upper_bound=cfg.map_upper_bound, uniform_rng=cfg.map_uniform_rng, random_state=GEN_SEED)
        Z = Z[:, :m]
        X = Z @ A.T + context_noise # (action_space_size, d)
        theta = param_generator(dimension=m, distribution=cfg.param_dist, disjoint=cfg.param_disjoint, 
                                uniform_rng=cfg.param_uniform_rng, random_state=GEN_SEED)
        
    if cfg.obs_feature_bound is not None:
        # observable context must be scaled, not clipped
        norms = np.array([l2norm(X[i, :]) for i in range(action_space_size)])
        max_norm = np.max(norms)
        for i in range(action_space_size):
            X[i, :] *= (cfg.obs_feature_bound / max_norm)
        
    if mode == "full":
        print(f"Mapping shape: {A.shape}\tParameter shape: {theta.shape}")
        assert ALPHAS is not None
        result = dict()
        for alpha in ALPHAS:
            print(f"alpha={alpha}")
            result_container = np.zeros(trials, dtype=object)
            for trial in range(trials):
                agent = LinUCB(d=d, alpha=alpha)
                regret = run(mode=mode, agent=agent, num_actions=num_actions, horizon=T, obs=X, latent=Z, reward_params=theta, 
                             reward_noise_var=cfg.reward_noise_var, use_tqdm=cfg.tqdm, verbose=False, random_state=RUN_SEED+(trial*T))
                result_container[trial] = np.cumsum(regret)
            result[alpha] = result_container
        label_name = "alpha"
    else:
        print(f"Mapping shape: {A.shape}\tParameter shape: {theta.shape}")
        if cfg.check_arms:
            assert ARMS is not None
            result = dict()
            alpha = ALPHAS[-1]
            for arms in ARMS:
                print(f"Number of Arms={arms}")
                result_container = np.zeros(trials, dtype=object)
                for trial in range(trials):
                    agent = PartialLinUCB(d=d, num_actions=arms, alpha=alpha)
                    regret = run(mode=mode, agent=agent, num_actions=arms, horizon=T, obs=X, latent=Z, reward_params=theta, 
                                 reward_noise_var=cfg.reward_noise_var, use_tqdm=cfg.tqdm, verbose=False, random_state=RUN_SEED+(trial*T))
                    result_container[trial] = np.cumsum(regret)
                result[arms] = result_container
            label_name = "Number of arms"
        else:
            assert ALPHAS is not None
            result = dict()
            for alpha in ALPHAS:
                print(f"alpha={alpha}")
                result_container = np.zeros(trials, dtype=object)
                for trial in range(trials):
                    agent = PartialLinUCB(d=d, num_actions=num_actions, alpha=alpha)
                    regret = run(mode=mode, agent=agent, num_actions=num_actions, horizon=T, obs=X, latent=Z, reward_params=theta, 
                                 reward_noise_var=cfg.reward_noise_var, use_tqdm=cfg.tqdm, verbose=False, random_state=RUN_SEED+(trial*T))
                    result_container[trial] = np.cumsum(regret)
                result[alpha] = result_container
            label_name = "alpha"
    
    ## save the result plot
    fname = f"experiment_result_{datetime.now()}_{cfg.bound_method}"
    fig = show_result(result=result, label_name=label_name, feat_dist_label=cfg.feat_dist, feat_disjoint=cfg.feat_disjoint, 
                    context_label=context_label, reward_label=str(cfg.reward_noise_var))
    save_plot(fig, path=FIGURE_PATH, fname=fname)
    
    out = vars(cfg)
    save_result(result=out, path=RESULT_PATH, fname=fname, filetype=cfg.filetype)

