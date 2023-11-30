from cfg import get_cfg
from models import *
from util import *

MODEL_DICT = {
    "linucb": LinUCB,
    "linegreedy": LineGreedy,
    "lints": LinTS
}

FEAT_DICT = {
    ("gaussian", True): r"$\sim N(0, I_k)$",
    ("gaussian", False): r"$\sim N(0, \Sigma_k)$",
    ("uniform", True): r"$\sim Unif_{I_k}$",
    ("uniform", False): r"$\sim Unif_{\Sigma_k}$"
}

MOTHER_PATH = "/home/sungwoopark/bandit-research/latent-contextual-bandit/modules"

PATH_DICT = {
    ("full", "fixed"): "full/fixed_vary/",
    ("full", "unfixed"): "full/unfixed_vary/",
    ("partial"): "partial/vary/",
}

def run_trials(model_name:str, mode:str, trials:int, alpha:float, arms:int, lbda:float, epsilon:float, horizon:int, 
               latent:np.ndarray, decoder:np.ndarray, reward_params:np.ndarray, noise_dist:Tuple[str], noise_std:Tuple[float], 
               feat_bound:float, feat_bound_method:str, random_state:int, is_fixed:str, egreedy:bool=False, verbose:bool=False):
    obs_dim, latent_dim = decoder.shape
    action_size = latent.shape[0]
    
    regret_container = np.zeros(trials, dtype=object)
        
    for trial in range(trials):
        if mode == "full":
            assert model_name != "plu"
            agent = MODEL_DICT[model_name](d=obs_dim, alpha=alpha, lbda=lbda)
        else:
            if model_name == "plu": 
                agent = PartialLinUCB(d=obs_dim, arms=arms, alpha=alpha, lbda=lbda)
            else:
                agent = MODEL_DICT[model_name](d=obs_dim, alpha=alpha, lbda=lbda)
        print(f"\u03B1={alpha},\t|A|={arms},\tmodel={agent.__class__.__name__}")
        random_state_ = random_state + (11111*(trial+1)) + int(11111*alpha) + (11111*arms)
        
        if mode == "partial":
            inherent_rewards = param_generator(dimension=arms, distribution=cfg.param_dist, disjoint=cfg.param_disjoint, 
                                               bound=cfg.param_bound, uniform_rng=cfg.param_uniform_rng, random_state=random_state_)
        else:
            inherent_rewards = 0.
        
        if is_fixed == "fixed":
            np.random.seed(random_state_)
            idx = np.random.choice(np.arange(action_size), size=arms, replace=False)
            latent_ = latent[idx, :].copy()
            action_space_size = arms
        else:
            latent_ = latent.copy()
            action_space_size = action_size
        
        print(f"Running seed : {random_state_}, Shape of the latent features : {latent_.shape}")
        # print(latent)
        regrets = run(model_name=model_name, mode=mode, agent=agent, horizon=horizon, action_size=action_space_size, arms=arms, latent=latent_, 
                      decoder=decoder, reward_params=reward_params, inherent_rewards=inherent_rewards, noise_dist=noise_dist, noise_std=noise_std, 
                      feat_bound=feat_bound, feat_bound_method=feat_bound_method, random_state=random_state_, verbose=verbose)
        regret_container[trial] = regrets
    
    return regret_container


def run(mode:str, model_name:str, agent:Union[LinUCB, LineGreedy, LinTS, PartialLinUCB], horizon:int, action_size:int, arms:int, 
        latent:np.ndarray, decoder:np.ndarray, reward_params:np.ndarray, inherent_rewards:Union[np.ndarray, float],
        noise_dist:Tuple[str], noise_std:Tuple[float], feat_bound:float, feat_bound_method:str, random_state:int, verbose:bool):
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
            random_state_ = random_state + t
            np.random.seed(random_state_)
        
        if action_size > arms:
            idx = np.random.choice(np.arange(action_size), size=arms, replace=False)
        else:
            idx = np.arange(action_size)
        latent_set, mapped_set = latent[idx, :], observe_space[idx, :]
        
        ## sample the context noise and construct the observable features
        context_noise = subgaussian_noise(distribution=context_noise_dist, size=(arms*obs_dim), 
                                          std=context_noise_std, random_state=random_state_).reshape(arms, d)
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
            print(f"Mode : {mode}\tFixed arm set : {(action_size == arms)}\tReward range : [{np.amin(expected_reward):.5f}, {np.amax(expected_reward):.5f}]")
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


def show_result(regrets:dict, feat_dist_label:str, feat_disjoint:bool, context_label:str, reward_label:str, figsize:tuple=(13, 5)):
    NROWS, NCOLS = 1, 2
    fig, (ax1, ax2) = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=figsize)
    
    for key in regrets:
        item = regrets[key]
        ax1.plot(np.mean(item, axis=0), label=key)
    ax1.grid(True)
    ax1.set_xlabel("Round")
    ax1.set_ylabel(r"$R_t$")
    ax1.set_title(r"$\bar{R}_t$")
    ax1.legend()
    
    for key in regrets:
        item = regrets[key]
        mean = np.mean(item, axis=0)
        std = np.std(item, axis=0, ddof=1)
        ax2.plot(mean, label=key)
        ax2.fill_between(np.arange(T), mean-std, mean+std, alpha=0.2)
    ax2.grid(True)
    ax2.set_xlabel("Round")
    ax2.set_ylabel(r"$R_t$")
    ax2.set_title(r"$\bar{R}_t \pm 1SD$")
    ax2.legend()
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(f"$Z${FEAT_DICT[(feat_dist_label, feat_disjoint)]}, $\sigma_\eta=${context_label}, $\sigma_\epsilon=${reward_label}, seed={SEED}, num_visibles={cfg.num_visibles}")
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
    ALPHAS = cfg.alphas
    
    if cfg.mode == "full":
        if cfg.fixed:
            path_flag = "fixed"
            run_flag = "fixed"
        else:
            path_flag = "unfixed"
            run_flag = "unfixed"
        RESULT_PATH = f"{MOTHER_PATH}/model_comparison/results/{PATH_DICT[(cfg.mode, path_flag)]}"
        FIGURE_PATH = f"{MOTHER_PATH}/model_comparison/figures/{PATH_DICT[(cfg.mode, path_flag)]}"        
            
    if cfg.mode == "partial":
        run_flag = "fixed"
        RESULT_PATH = f"{MOTHER_PATH}/model_comparison/results/{PATH_DICT[(cfg.mode)]}"
        FIGURE_PATH = f"{MOTHER_PATH}/model_comparison/figures/{PATH_DICT[(cfg.mode)]}"
            
    if "T" in cfg.context_std:
        power = cfg.context_std.split("T")[-1]
        context_std = T ** float(power)
        context_label = f"$T^{{{power}}}$"
    else:
        context_std = float(cfg.context_std)
        context_label = cfg.context_std
    
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
        models = ["linucb", "linegreedy", "lints", "plu"]
    else:
        true_mu = param_generator(dimension=k, distribution=cfg.param_dist, disjoint=cfg.param_disjoint, 
                                  bound=cfg.param_bound, uniform_rng=cfg.param_uniform_rng, random_state=SEED-1)
        models = ["linucb", "linegreedy", "lints"]
    
    regret_results = dict()
    for model in models:
        if cfg.check_specs:
            if model != "plu":
                key = MODEL_DICT[model].__name__
            else:
                key = PartialLinUCB.__name__
            print(f"Model : {key}")
            print(f"Context std : {context_std:.6f}, Original seed : {SEED}, Number of influential variables : {m}")
            print(f"The maximum norm of the latent features : {np.amax([l2norm(feat) for feat in Z]):.4f}")
            print(f"Shape of the decoder mapping : {A.shape},\tNumber of reward parameters : {true_mu.shape[0]}")
            print(f"L2 norm of the true mu : {l2norm(true_mu):.4f}")
            
        regrets = run_trials(model_name=model, mode=cfg.mode, trials=cfg.trials, alpha=ALPHAS[0], arms=num_actions[0], lbda=cfg.lbda, epsilon=cfg.epsilon, 
                             horizon=T, latent=Z, decoder=A, reward_params=true_mu, noise_dist=("gaussian", "gaussian"), noise_std=(context_std, cfg.reward_std), 
                             feat_bound=cfg.obs_feature_bound, feat_bound_method=cfg.obs_bound_method, random_state=SEED, is_fixed=run_flag)
        regret_results[key] = regrets
    
    fname = f"{cfg.mode}_seed_{SEED}_noise_{cfg.context_std}_nvisibles_{cfg.num_visibles}_{cfg.latent_bound_method}_feat_{cfg.feat_dist}_map_{cfg.map_dist}_param_{cfg.param_dist}"
    fig = show_result(regrets=regret_results, feat_dist_label=cfg.feat_dist, feat_disjoint=cfg.feat_disjoint, 
                      context_label=context_label, reward_label=str(cfg.reward_std))
    save_plot(fig, path=FIGURE_PATH, fname=fname)
    save_result(result=vars(cfg), path=RESULT_PATH, fname=fname, filetype=cfg.filetype)