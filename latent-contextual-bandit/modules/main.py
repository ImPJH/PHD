from cfg import get_cfg
from models import *
from util import *

PATH = "./results"

FEAT_DICT = {
    ("gaussian", True): "$\sim N(0, I)$",
    ("gaussian", False): "$\sim N(0, \Sigma)$",
    ("uniform", True): "$\sim Unif_{I}$",
    ("uniform", False): "$\sim Unif_{\Sigma}$"
}


def run(mode:str, agent:Union[LinUCB, LineGreedy, PartialLinUCB], num_actions:int, horizon:int, obs:np.ndarray, latent:np.ndarray, 
        params:np.ndarray, noise:np.ndarray, use_tqdm:bool, verbose:bool):
    action_space_size, _ = obs.shape
    assert action_space_size >= num_actions, "The cardinality of the entire action space must be larger than the number of actions."
    
    if use_tqdm:
        bar = tqdm(range(horizon))
    else:
        bar = range(horizon)
    
    regrets = np.zeros(horizon)
    for t in bar:
        if action_space_size == num_actions:
            ## fixed action space
            indices = np.arange(num_actions)
            if mode == "partial":
                inherent_rewards = param_generator(dimension=num_actions, distribution=cfg.param_dist, disjoint=cfg.param_disjoint, 
                                                   uniform_rng=cfg.param_uniform_rng, random_state=SEED+10)
            else:
                inherent_rewards = 0.
        else:
            indices = np.random.randint(0, action_space_size, num_actions)
            if mode == "partial":
                inherent_rewards = param_generator(dimension=num_actions, distribution=cfg.param_dist, disjoint=cfg.param_disjoint, 
                                                   uniform_rng=cfg.param_uniform_rng, random_state=SEED+(10+t))
            else:
                inherent_rewards = 0.
        
        ## observe the actions (num_actions, d), (num_actions, k or m)
        action_set, latent_action_set = obs[indices], latent[indices]
        if mode == "partial":
            onehot = np.identity(num_actions)
            action_set = np.c_[action_set, onehot]
        
        ## compute the rewards and the optimal actions
        expected_rewards = latent_action_set @ params + inherent_rewards
        true_rewards = expected_rewards + noise
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


def show_plot(result:dict, label_name:str, feat_dist_label:str, feat_disjoint:bool, context_label:str, reward_label:str):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
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
    
    fig.suptitle(f"Z{FEAT_DICT[(feat_dist_label, feat_disjoint)]}, $\sigma_\eta=${context_label}, $\sigma_\epsilon=${reward_label}")    
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
    SEED = cfg.seed
    ALPHAS = cfg.alphas
    NUM_ARMS = cfg.num_arms
    if "T" in cfg.context_var:
        exp = cfg.context_var.split("T")[-1]
        context_var = T ** float(exp)
        context_label = f"$T^{exp}$"
    else:
        context_var = float(cfg.context_var)
        context_label = cfg.context_var
    
    ## generate the latent variable whose dimension is (action_space_size, k)
    Z = feature_sampler(dimension=k, feat_dist=cfg.feat_dist, size=action_space_size, disjoint=cfg.feat_disjoint,
                        cov_dist=cfg.cov_dist, bound=cfg.latent_feature_bound, uniform_rng=tuple(cfg.feat_uniform_rng), random_state=SEED)
       
    ## generate the context noise of shape (action_space_size, d)
    context_noise = subgaussian_noise(distribution="gaussian", size=action_space_size*d, random_state=SEED, std=context_var).reshape(action_space_size, d)
    
    ## generate the observable context
    if mode == "full":
        ## generate the decoder mapping A : Z -> X (k, d), X (action_space_size, d), and theta (k, )
        A = mapping_generator(latent_dim=k, obs_dim=d, distribution=cfg.map_dist, lower_bound=cfg.map_lower_bound, 
                              upper_bound=cfg.map_upper_bound, uniform_rng=cfg.map_uniform_rng, random_state=SEED)
        X = Z @ A + context_noise # (action_space_size, d)
        theta = param_generator(dimension=k, distribution=cfg.param_dist, disjoint=cfg.param_disjoint, 
                                uniform_rng=cfg.param_uniform_rng, random_state=SEED)
    else:
        ## generate the decoder mapping A : Z_visible -> X (m, d)
        A = mapping_generator(latent_dim=m, obs_dim=d, distribution=cfg.map_dist, lower_bound=cfg.map_lower_bound, 
                              upper_bound=cfg.map_upper_bound, uniform_rng=cfg.map_uniform_rng, random_state=SEED)
        Z = Z[:, :m]
        X = Z @ A + context_noise # (action_space_size, d)
        theta = param_generator(dimension=m, distribution=cfg.param_dist, disjoint=cfg.param_disjoint, 
                                uniform_rng=cfg.param_uniform_rng, random_state=SEED)
        
    if cfg.obs_feature_bound is not None:
        norms = np.array([l2norm(X[i, :]) for i in range(action_space_size)])
        max_norm = np.max(norms)
        for i in range(action_space_size):
            X[i, :] *= (cfg.obs_feature_bound / max_norm)
    
    ## generate the reward noise (action_space_size, )
    reward_noise = subgaussian_noise(distribution="gaussian", size=action_space_size, random_state=SEED, std=cfg.reward_noise_var)
    
    if mode == "full":
        print(f"Mapping shape: {A.shape}\tParameter shape: {theta.shape}")
        assert ALPHAS is not None
        result = dict()
        for alpha in ALPHAS:
            print(f"alpha={alpha}")
            result_container = np.zeros(trials, dtype=object)
            for trial in trials:
                agent = LinUCB(d=d, alpha=alpha)
                regret = run(mode=mode, agent=agent, num_actions=num_actions, horizon=T, obs=X, 
                             latent=Z, params=theta, use_tqdm=True, verbose=False)
                result_container[trial] = regret
            result[alpha] = result_container
        label_name = "alpha"
    else:
        print(f"Mapping shape: {A.shape}\tParameter shape: {theta.shape}")
        if cfg.check_arms:
            assert NUM_ARMS is not None
            result = dict()
            alpha = ALPHAS[-1]
            for arms in NUM_ARMS:
                print(f"Number of Arms={arms}")
                result_container = np.zeros(trials, dtype=object)
                for trial in trials:
                    agent = PartialLinUCB(d=d, num_actions=arms, alpha=alpha)
                    regret = run(mode=mode, agent=agent, num_actions=arms, horizon=T, obs=X, 
                                 latent=Z, params=theta, use_tqdm=True, verbose=False)
                    result_container[trial] = regret
                result[arms] = result_container
            label_name = "Number of arms"
        else:
            assert ALPHAS is not None
            result = dict()
            for alpha in ALPHAS:
                print(f"alpha={alpha}")
                result_container = np.zeros(trials, dtype=object)
                for trial in trials:
                    agent = LinUCB(d=d, num_actions=num_actions alpha=alpha)
                    regret = run(mode=mode, agent=agent, num_actions=num_actions, horizon=T, obs=X, 
                                 latent=Z, params=theta, use_tqdm=True, verbose=False)
                    result_container[trial] = regret
                result[alpha] = result_container
            label_name = "alpha"
    
    fname = f"exp_{datetime.now()}"
    fig = show_plot(result=result, label_name=label_name, feat_dist_label=cfg.feat_dist, feat_disjoint=cfg.feat_disjoint, 
                           context_label=context_label, reward_label=str(reward_noise))
    save_plot(fig, path=PATH, fname=fname)
    
    out = vars(cfg)
    out['figure'] = fig
    save_result(result=out, path=PATH, fname=fname)
