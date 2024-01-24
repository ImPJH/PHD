from cfg import get_cfg
from models_v3 import *
from util import *
from calculate_alpha_v2 import lints_alpha, linucb_alpha

MODEL_DICT = {
    "linucb": LinUCB,
    "lints": LinTS,
    "plu": POLO
}

FEAT_DICT = {
    ("gaussian", True): r"$\sim N(0, I_k)$",
    ("gaussian", False): r"$\sim N(0, \Sigma_k)$",
    ("uniform", True): r"$\sim Unif_{I_k}$",
    ("uniform", False): r"$\sim Unif_{\Sigma_k}$"
}

MOTHER_PATH = "/home/sungwoopark/bandit-research/latent-contextual-bandit/modules/final"

PATH_DICT = {
    ("full", "fixed"): "full/fixed_vary/",
    ("full", "unfixed"): "full/unfixed_vary/",
    ("partial"): "partial/vary/",
}

DIST_DICT = {
    "gaussian": "g",
    "uniform": "u"
}

DEP_DICT = {
    True: "indep",
    False: "dep"
}

METHOD_DICT = {
    "scaling": "s",
    "clipping": "c"
}

def run_trials(model_name:str, trials:int, arms:int, lbda:float, horizon:int, latent:np.ndarray, decoder:np.ndarray, 
               reward_params:np.ndarray, noise_dist:Tuple[str], noise_std:List[Union[float, List[float]]], 
               feat_bound:float, feat_bound_method:str, random_state:int, is_fixed:str, verbose:bool=False):
    obs_dim, _ = decoder.shape
    context_noise_std, reward_noise_std = noise_std
    action_size = latent.shape[0]
    
    regret_container = np.zeros(trials, dtype=object)
    for trial in range(trials):
        if model_name == "plu": 
            agent = POLO(d=obs_dim, arms=arms, lbda=lbda, reward_std=reward_noise_std, context_std=context_noise_std, horizon=horizon)
        elif model_name == "linucb":
            agent = LinUCB(d=obs_dim, alpha=linucb_alpha(delta=cfg.delta), lbda=lbda)
        elif model_name == "lints":
            agent = LinTS(d=obs_dim, alpha=lints_alpha(d=obs_dim, horizon=horizon, reward_std=reward_noise_std, delta=cfg.delta), lbda=lbda)
        print(f"model={agent.__class__.__name__},\t|A|={arms}")
        
        random_state_ = random_state + (121212*(trial+1)) + (999999*arms)
        inherent_rewards = param_generator(dimension=arms, distribution=cfg.bias_dist, disjoint=cfg.param_disjoint, 
                                            bound=cfg.param_bound, uniform_rng=cfg.param_uniform_rng, random_state=random_state_)
        
        if is_fixed == "fixed":
            np.random.seed(random_state_)
            idx = np.random.choice(np.arange(action_size), size=arms, replace=False)
            latent_ = latent[idx, :].copy()
            action_space_size = arms
        else:
            latent_ = latent.copy()
            action_space_size = action_size
        
        print(f"Running seed : {random_state_}, Shape of the latent features : {latent_.shape}")
        regrets = run(model_name=model_name, agent=agent, horizon=horizon, action_size=action_space_size, arms=arms, latent=latent_, 
                      decoder=decoder, reward_params=reward_params, inherent_rewards=inherent_rewards, noise_dist=noise_dist,
                      noise_std=noise_std, feat_bound=feat_bound, feat_bound_method=feat_bound_method, random_state=random_state_, verbose=verbose)
        regret_container[trial] = regrets
    
    return regret_container


def run(model_name:str, agent:Union[LinUCB, LinTS, POLO], horizon:int, action_size:int, arms:int, 
        latent:np.ndarray, decoder:np.ndarray, reward_params:np.ndarray, inherent_rewards:Union[np.ndarray, float], noise_dist:Tuple[str], 
        noise_std:List[Union[float, List[float]]], feat_bound:float, feat_bound_method:str, random_state:int, verbose:bool):
    # action_size, _ = latent.shape
    obs_dim, _ = decoder.shape
    context_noise_dist, reward_noise_dist = noise_dist
    context_noise_std, reward_noise_std = noise_std
    
    ## make the mapped features in advance before iteration
    observe_space = latent @ decoder.T   # (M, k) @ (k, d) -> (M, d) or (M, m) @ (m, d) -> (M, d)
    regrets = np.zeros(horizon, dtype=float)
    
    if not verbose:
        bar = tqdm(range(horizon))
    else:
        bar = range(horizon)
    
    for t in bar:
        if random_state is not None:
            random_state_ = random_state + int(11111 * t)
            np.random.seed(random_state_)
        
        if action_size > arms:
            idx = np.random.choice(np.arange(action_size), size=arms, replace=False)
        else:
            idx = np.arange(action_size)
        latent_set, mapped_set = latent[idx, :], observe_space[idx, :]
        
        ## sample the context noise and construct the observable features
        if isinstance(context_noise_std, float):
            context_noise = subgaussian_noise(distribution=context_noise_dist, size=(arms*obs_dim), 
                                              std=context_noise_std, random_state=random_state_).reshape(arms, d)
        else:
            context_noise = subgaussian_noise(distribution=context_noise_dist, size=(arms*obs_dim), 
                                              std=context_noise_std[t], random_state=random_state_).reshape(arms, d)            
        action_set = mapped_set + context_noise
        
        ## bound the action set
        if feat_bound_method is not None:
            bounding(type="feature", v=action_set, bound=feat_bound, method=feat_bound_method)
        
        if model_name == "plu":
            action_set = np.concatenate([action_set, np.identity(arms)], axis=1)
        
        ## sample the reward noise and compute the reward
        reward_noise = subgaussian_noise(distribution=reward_noise_dist, size=arms, std=reward_noise_std, random_state=random_state_)
        expected_reward = latent_set @ reward_params + inherent_rewards
        
        if t == 0:
            print(f"Fixed arm set : {(action_size == arms)}\tReward range : [{np.amin(expected_reward):.5f}, {np.amax(expected_reward):.5f}]")
        realized_reward = expected_reward + reward_noise
        optimal_arm = np.argmax(expected_reward)
        optimal_reward = expected_reward[optimal_arm]
        
        ## choose the best action
        chosen_arm = agent.choose(action_set)
        chosen_reward = realized_reward[chosen_arm]
        chosen_context = action_set[chosen_arm]
        
        ## compute the regret and errors, if necessary
        regrets[t] = optimal_reward - expected_reward[chosen_arm]
        
        if verbose: 
            print(f"round {t+1}\toptimal action : {optimal_arm}\toptimal reward : {optimal_reward:.3f}")
            print(f"\tchosen action : {chosen_arm}\trealized reward : {chosen_reward:.3f}, expected reward: {expected_reward[chosen_arm]:.3f}")
            print(f"\tinstance regret : {regrets[t]:.3f}, cumulative regret : {np.sum(regrets):.3f}")
        
        ## update the agent
        agent.update(x=chosen_context, r=chosen_reward)
    return np.cumsum(regrets)


def show_result(regrets:dict, figsize:tuple=(13, 5)):
    NROWS, NCOLS = 1, 2
    fig, (ax1, ax2) = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=figsize)
    
    for key in regrets:
        item = regrets[key]
        ax1.plot(np.mean(item, axis=0), label=key)
    ax1.grid(True)
    ax1.set_xlabel("Round")
    ax1.set_ylabel(r"Regret")
    ax1.set_title(r"10 Trials Average $R_T$")
    ax1.legend()
    
    for key in regrets:
        item = regrets[key]
        mean = np.mean(item, axis=0)
        std = np.std(item, axis=0, ddof=1)
        ax2.plot(mean, label=key)
        ax2.fill_between(np.arange(T), mean-std, mean+std, alpha=0.2)
    ax2.grid(True)
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Regret")
    ax2.set_title(r"10 Trials Average $R_T \pm 1SD$")
    ax2.legend()
    
    fig.tight_layout()    
    return fig

if __name__ == "__main__":
    cfg = get_cfg()
    
    ## hyper-parameters
    action_spaces = cfg.action_spaces # int
    num_actions = cfg.num_actions # List[int]
    d = cfg.obs_dim
    k = cfg.latent_dim
    m = cfg.num_visibles
    T = cfg.horizon
    SEED = cfg.seed

    if "T" in cfg.context_std:
        power = cfg.context_std.split("T")[-1]
        context_std = T ** float(power)
        # context_std = [(t+1) ** float(power) for t in range(T)]
        context_label = f"$t^{{{power}}}$"
    else:
        context_std = float(cfg.context_std)
        context_label = cfg.context_std      
            
    run_flag = "fixed"
    RESULT_PATH = f"{MOTHER_PATH}/model_comparison_v3/results/{PATH_DICT[(cfg.mode)]}"
    FIGURE_PATH = f"{MOTHER_PATH}/model_comparison_v3/figures/{PATH_DICT[(cfg.mode)]}"
    
    ## generate the latent feature
    Z = feature_sampler(dimension=k, feat_dist=cfg.feat_dist, size=action_spaces, disjoint=cfg.feat_disjoint, 
                        cov_dist=cfg.feat_cov_dist, bound=cfg.latent_feature_bound, bound_method=cfg.latent_bound_method, 
                        uniform_rng=cfg.feat_uniform_rng, random_state=SEED)
    
    ## generate the decoder mapping
    A = mapping_generator(latent_dim=k, obs_dim=d, distribution=cfg.map_dist, lower_bound=cfg.map_lower_bound, 
                          upper_bound=cfg.map_upper_bound, uniform_rng=cfg.map_uniform_rng, random_state=SEED+1)
    
    ## generate the true parameter -> corresponds to "mu"
    if cfg.mode == "partial":
        Z = Z[:, :m]            # (M, k) -> (M, m)
        A = A[:, :m]            # (d, k) -> (d, m)
        true_mu = param_generator(dimension=m, distribution=cfg.param_dist, disjoint=cfg.param_disjoint, 
                                  bound=cfg.param_bound, uniform_rng=cfg.param_uniform_rng, random_state=SEED-1)
        models = ["linucb", "lints", "plu"]
    # else:
    #     true_mu = param_generator(dimension=k, distribution=cfg.param_dist, disjoint=cfg.param_disjoint, 
    #                               bound=cfg.param_bound, uniform_rng=cfg.param_uniform_rng, random_state=SEED-1)
    #     models = ["linucb", "lints"]
    
    regret_results = dict()
    for model in models:
        if cfg.check_specs:
            key = MODEL_DICT[model].__name__
            print(f"Model : {key}, Feature : {cfg.feat_dist}, Bias : {cfg.bias_dist}, Parameter : {cfg.param_dist}")
            print(f"Context std : {cfg.context_std}, Original seed : {SEED}, Number of influential variables : {m}")
            print(f"The maximum norm of the latent features : {np.amax([l2norm(feat) for feat in Z]):.4f}")
            print(f"Shape of the decoder mapping : {A.shape},\tNumber of reward parameters : {true_mu.shape[0]}")
            print(f"Lambda : {cfg.lbda}\tL2 norm of the true mu : {l2norm(true_mu):.4f}")
            
        regrets = run_trials(model_name=model, trials=cfg.trials, arms=num_actions[-1], lbda=cfg.lbda, horizon=T, latent=Z, decoder=A, 
                             reward_params=true_mu, noise_dist=("gaussian", "gaussian"), noise_std=[context_std, cfg.reward_std], 
                             feat_bound=cfg.obs_feature_bound, feat_bound_method=cfg.obs_bound_method, random_state=SEED, is_fixed=run_flag)
        regret_results[key] = regrets
    
    fname = (f"{cfg.mode}_{SEED}_noise_{cfg.context_std}_nvisibles_{cfg.num_visibles}_" 
             f"{METHOD_DICT[cfg.latent_bound_method]}_feat_{DIST_DICT[cfg.feat_dist]}_"
             f"{DEP_DICT[cfg.feat_disjoint]}_bias_{DIST_DICT[cfg.bias_dist]}_map_{DIST_DICT[cfg.map_dist]}_"
             f"param_{DIST_DICT[cfg.param_dist]}_{DEP_DICT[cfg.param_disjoint]}_arm_{num_actions[-1]}")
    fig = show_result(regrets=regret_results)
    save_plot(fig, path=FIGURE_PATH, fname=fname)
    save_result(result=vars(cfg), path=RESULT_PATH, fname=fname, filetype=cfg.filetype)