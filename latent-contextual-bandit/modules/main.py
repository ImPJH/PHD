from cfg import get_cfg
from models import *
from util import *

FEAT_DICT = {
    ("gaussian", True): r"$\sim N(0, I_k)$",
    ("gaussian", False): r"$\sim N(0, \Sigma_k)$",
    ("uniform", True): r"$\sim Unif_{I_k}$",
    ("uniform", False): r"$\sim Unif_{\Sigma_k}$"
}

def run_trials(mode:str, trials:int, alpha_list:list, action_list:list, lbda:float, epsilon:float, horizon:int, latent:np.ndarray, 
               num_visibles:int, decoder:np.ndarray, reward_params:np.ndarray, noise_dist:Tuple[str], noise_std:Tuple[float], 
               feat_bound:float, feat_bound_method:str, random_state:int, egreedy:bool=False, verbose:bool=False):
    assert len(alpha_list) == 1 or len(action_list) == 1, f"Either one of the alphas or num_actions must be one."
    
    obs_dim, _ = decoder.shape
    result = dict()
    
    for alpha in alpha_list:
        for arms in action_list:
            print(f"\u03B1={alpha}\t|A|={arms}")
            regret_container = np.zeros(trials, dtype=object)
            error_container = np.zeros(trials, dtype=object)
            for trial in range(trials):
                if mode == "full":
                    if not egreedy:
                        agent = LinUCB(d=obs_dim, alpha=alpha, lbda=lbda)
                    else:
                        agent = LineGreedy(d=obs_dim, alpha=alpha, lbda=lbda, epsilon=epsilon)    
                else:                                                                                                                       
                    agent = PartialLinUCB(d=obs_dim, num_actions=arms, alpha=alpha, lbda=lbda)
                random_state_ = random_state + (1000000*trial) + int(999999*alpha)
                regrets, errors = run(mode=mode, agent=agent, horizon=horizon, num_actions=arms, latent=latent, num_visibles=num_visibles, 
                                      decoder=decoder, reward_params=reward_params, noise_dist=noise_dist, noise_std=noise_std, 
                                      feat_bound=feat_bound, feat_bound_method=feat_bound_method, random_state=random_state_, verbose=verbose)
                regret_container[trial] = regrets
                error_container[trial] = errors
            
            if len(alpha_list) == 1:
                key = arms
            else:
                key = alpha
            result[key] = (regret_container, error_container)

    if len(alpha_list) == 1:
        label_name = r"$\vert \mathcal{A}\vert$"
    else:
        label_name = r"$\alpha$"
    return result, label_name


def run(mode:str, agent:Union[LinUCB, LineGreedy, PartialLinUCB], horizon:int, num_actions:int, latent:np.ndarray, num_visibles:int, decoder:np.ndarray, 
        reward_params:np.ndarray, noise_dist:Tuple[str], noise_std:Tuple[float], feat_bound:float, feat_bound_method:str, random_state:int, verbose:bool):
    action_space_size, _ = latent.shape
    obs_dim, _ = decoder.shape
    context_noise_dist, reward_noise_dist = noise_dist
    context_noise_std, reward_noise_std = noise_std
    decoder_inv = left_pseudo_inverse(decoder)
    theta_star = decoder_inv.T @ reward_params
    
    if mode == "partial":
        assert action_space_size == num_actions, f"If the mode is {mode}, the action space must be fixed"
        inherent_rewards = param_generator(dimension=num_actions, distribution=cfg.param_dist, disjoint=cfg.param_disjoint,
                                           bound=cfg.param_bound, uniform_rng=cfg.param_uniform_rng, random_state=random_state)
    else:
        inherent_rewards = 0.
    
    regrets = np.zeros(horizon, dtype=float)
    errors = np.zeros(horizon, dtype=float)

    if not verbose:
        bar = tqdm(range(horizon))
    else:
        bar = range(horizon)
    
    for t in bar:
        if random_state is not None:
            random_state += t
        idx = np.random.choice(np.arange(action_space_size), size=num_actions, replace=False)
        latent_set = latent[idx, :]
        if mode == "partial":
            latent_set = latent_set[:, :num_visibles]
        
        ## sample the context noise and generate the observable features
        context_noise = subgaussian_noise(distribution=context_noise_dist, size=(num_actions*obs_dim), 
                                          std=context_noise_std, random_state=random_state).reshape(num_actions, d)
        action_set = latent_set @ decoder.T + context_noise
        
        ## bound the action set
        bounding(type="feature", v=action_set, bound=feat_bound, method=feat_bound_method)
        
        if mode == "partial":
            action_set = np.concatenate([action_set, np.identity(num_actions)], axis=1)
        
        ## sample the reward noise and compute the reward
        reward_noise = subgaussian_noise(distribution=reward_noise_dist, size=num_actions, std=reward_noise_std, random_state=random_state)
        expected_reward = latent_set @ reward_params + inherent_rewards
        if t == 0:
            print(f"Mode: {mode}\tFixed arm set: {(action_space_size == num_actions)}\tReward range: [{np.amin(expected_reward):.5f}, {np.amax(expected_reward):.5f}]")
        realized_reward = expected_reward + reward_noise
        optimal_arm = np.argmax(expected_reward)
        optimal_reward = expected_reward[optimal_arm]
        
        ## choose the best action
        chosen_arm = agent.choose(action_set)
        chosen_reward = realized_reward[chosen_arm]
        chosen_context = action_set[chosen_arm]
        
        ## compute the regret and errors, if necessary
        regrets[t] = optimal_reward - expected_reward[chosen_arm]
        errors[t] = l2norm(theta_star - agent.theta_hat)
        
        if verbose: 
            print(f"round {t+1}\toptimal action : {optimal_arm}\toptimal reward : {optimal_reward:.3f}")
            print(f"\tchosen action : {chosen_arm}\trealized reward : {chosen_reward:.3f}, expected reward: {expected_reward[chosen_arm]:.3f}")
            print(f"\ttheta empirical error : {errors[t]}, instance regret : {regrets[t]:.3f}, cumulative regret : {np.sum(regrets):.3f}")
        
        ## update the agent
        agent.update(x=chosen_context, r=chosen_reward)
    return np.cumsum(regrets), errors


def show_result(result:dict, label_name:str, feat_dist_label:str, feat_disjoint:bool, context_label:str, reward_label:str, figsize:tuple=(14, 10)):
    NROWS, NCOLS = 2, 2
    fig, ax = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=figsize)
    for i in range(NROWS):
        for j in range(NROWS):
            for key in result:
                item = result[key][i]
                if j == 0:
                    ax[i][j].plot(np.mean(item, axis=0), label=f"{label_name}={key}")
                else:
                    mean = np.mean(item, axis=0)
                    std = np.std(item, axis=0, ddof=1)
                    ax[i][j].plot(mean, label=f"alpha={key}")
                    ax[i][j].fill_between(np.arange(T), mean-std, mean+std, alpha=0.2)
                ax[i][j].set_xlabel("Round")
                if i == 0:
                    ax[i][j].set_ylabel(r"$R_t$")
                    ax[i][j].set_title("Regret")
                else:
                    ax[i][j].set_ylabel(r"${\Vert \hat{\theta}_t - \theta_*\Vert}_2$")
                    ax[i][j].set_title(r"Parameter Empirical Error")
                ax[i][j].grid(True)
                ax[i][j].legend()
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(f"$Z${FEAT_DICT[(feat_dist_label, feat_disjoint)]}, $\sigma_\eta=${context_label}, $\sigma_\epsilon=${reward_label}, $Z$ bound={cfg.latent_bound_method}, $X$ bound={cfg.obs_bound_method}")    
    return fig


if __name__ == "__main__":
    cfg = get_cfg()
    
    ## hyper-parameters
    action_space_size = cfg.action_space_size # int
    num_actions = cfg.num_actions # list
    for arm_cnt in num_actions:
        assert action_space_size >= arm_cnt, "The cardinality of the entire action space must be larger than the number of actions."
    
    d = cfg.obs_dim
    k = cfg.latent_dim
    m = cfg.num_visibles
    T = cfg.horizon
    GEN_SEED, RUN_SEED = cfg.seed
    ALPHAS = cfg.alphas
    
    if "T" in cfg.context_std:
        power = cfg.context_std.split("T")[-1]
        context_std = T ** float(power)
        context_label = f"$T^{{{power}}}$"
    else:
        context_std = float(cfg.context_std)
        context_label = cfg.context_std
    print(f"Context std = {context_std}")
    
    RESULT_PATH = f"./results/{cfg.mode}/"
    FIGURE_PATH = f"./figures/{cfg.mode}/"
    
    if cfg.check_param_error:
        assert cfg.mode == "full" and d == k, "You can check the empirical error of the parameter only if the mode is full and the latent dimension and observable dimension are equal."
    
    ## generate the latent feature
    Z = feature_sampler(dimension=k, feat_dist=cfg.feat_dist, size=action_space_size, disjoint=cfg.feat_disjoint, 
                        cov_dist=cfg.feat_cov_dist, bound=cfg.latent_feature_bound, bound_method=cfg.latent_bound_method, 
                        uniform_rng=cfg.feat_uniform_rng, random_state=GEN_SEED)
    
    ## generate the decoder mapping
    A = mapping_generator(latent_dim=k, obs_dim=d, distribution=cfg.map_dist, lower_bound=cfg.map_lower_bound,
                          upper_bound=cfg.map_upper_bound, uniform_rng=cfg.map_uniform_rng, random_state=GEN_SEED+1)
    
    ## generate the true parameter
    true_mu = param_generator(dimension=k, distribution=cfg.param_dist, disjoint=cfg.param_disjoint, bound=cfg.param_bound, 
                                 uniform_rng=cfg.param_uniform_rng, random_state=GEN_SEED+2)
    
    if cfg.check_specs:
        print(f"Shape of the latent feature matrix: {Z.shape}")
        print(f"The maximum norm of the latent features: {np.amax([l2norm(latent) for latent in Z]):.4f}")
        print(f"Shape of the decoder mapping: {A.shape}")
        print(f"L2 norm of the true theta: {l2norm(true_mu):.4f}")
        
    result, label_name = run_trials(agent_type=cfg.agent_type, trials=cfg.trials, alpha_list=ALPHAS, action_list=num_actions, lbda=cfg.lbda, epsilon=cfg.epsilon, 
                                    mode=cfg.mode, horizon=T, latent=Z, num_visibles=m, decoder=A, reward_params=true_mu, noise_dist=("gaussian", "gaussian"), 
                                    noise_std=(context_std, cfg.reward_std), feat_bound=cfg.obs_feature_bound, feat_bound_method=cfg.obs_bound_method, random_state=RUN_SEED)
    
    ## save the results
    fname = f"experiment_result_{datetime.now()}_latent_{cfg.latent_bound_method}_obs_{cfg.obs_bound_method}"
    fig = show_result(result=result, label_name=label_name, feat_dist_label=cfg.feat_dist, feat_disjoint=cfg.feat_disjoint, 
                      context_label=context_label, reward_label=str(cfg.reward_std))
    save_plot(fig, path=FIGURE_PATH, fname=fname)
    save_result(result=vars(cfg), path=RESULT_PATH, fname=fname, filetype=cfg.filetype)
