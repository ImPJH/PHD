from cfg import get_cfg
from models import *
from util import *

FEAT_DICT = {
    ("gaussian", True): r"$\sim N(0, I_k)$",
    ("gaussian", False): r"$\sim N(0, \Sigma_k)$",
    ("uniform", True): r"$\sim Unif_{I_k}$",
    ("uniform", False): r"$\sim Unif_{\Sigma_k}$"
}

def run_trials(mode:str, trials:int, alpha:float, arms:int, lbda:float, epsilon:float, horizon:int, 
               latent:np.ndarray, decoder:np.ndarray, reward_params:np.ndarray, noise_dist:Tuple[str], noise_std:Tuple[float], 
               feat_bound:float, feat_bound_method:str, random_state:int, egreedy:bool=False, verbose:bool=False):
    obs_dim, _ = decoder.shape
    
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
            agent = PartialLinUCB(d=obs_dim, arms=arms, alpha=alpha, lbda=lbda)
        random_state_ = random_state + (999999*trial) + int(1000000*alpha) + (1000001*arms)
        regrets, errors = run(mode=mode, agent=agent, horizon=horizon, arms=arms, latent=latent, decoder=decoder, 
                              reward_params=reward_params, noise_dist=noise_dist, noise_std=noise_std, feat_bound=feat_bound, 
                              feat_bound_method=feat_bound_method, random_state=random_state_, verbose=verbose)
        regret_container[trial] = regrets
        error_container[trial] = errors
    
    return regret_container, error_container


def run(mode:str, agent:Union[LinUCB, LineGreedy, PartialLinUCB], horizon:int, arms:int, latent:np.ndarray, decoder:np.ndarray, 
        reward_params:np.ndarray, noise_dist:Tuple[str], noise_std:Tuple[float], feat_bound:float, feat_bound_method:str, random_state:int, verbose:bool):
    action_size, _ = latent.shape
    obs_dim, _ = decoder.shape
    context_noise_dist, reward_noise_dist = noise_dist
    context_noise_std, reward_noise_std = noise_std
    
    ## make the mapped features in advance before iteration
    observe_space = latent @ decoder.T          # (M, k) @ (k, d) -> (M, d) or (M, m) @ (m, d) -> (M, d)
    decoder_inv = left_pseudo_inverse(decoder)
    true_theta = decoder_inv.T @ reward_params  # (d, m) @ (m, ) -> (d, )

    if mode == "partial":
        assert action_size == arms, f"If the mode is '{mode}', the action space must be fixed"
        inherent_rewards = param_generator(dimension=arms, distribution=cfg.param_dist, disjoint=cfg.param_disjoint,
                                           bound=cfg.param_bound, uniform_rng=cfg.param_uniform_rng, random_state=random_state)
        true_theta = np.concatenate([true_theta, inherent_rewards], axis=0) # (d, ) -> (d+N, )
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
            random_state_ = random_state + t
            
        if action_size > arms:
            idx = np.random.choice(np.arange(action_size), size=arms, replace=False)
        else:
            idx = np.arange(arms)
        latent_set, mapped_set = latent[idx, :], observe_space[idx, :]
        
        ## sample the context noise and generate the observable features
        context_noise = subgaussian_noise(distribution=context_noise_dist, size=(arms*obs_dim), 
                                          std=context_noise_std, random_state=random_state_).reshape(arms, d)
        action_set = mapped_set + context_noise
        
        ## bound the action set
        if feat_bound_method is not None:
            bounding(type="feature", v=action_set, bound=feat_bound, method=feat_bound_method)
        
        if mode == "partial":
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
        errors[t] = l2norm(true_theta - agent.theta_hat)
        
        if verbose: 
            print(f"round {t+1}\toptimal action : {optimal_arm}\toptimal reward : {optimal_reward:.3f}")
            print(f"\tchosen action : {chosen_arm}\trealized reward : {chosen_reward:.3f}, expected reward: {expected_reward[chosen_arm]:.3f}")
            print(f"\ttheta empirical error : {errors[t]}, instance regret : {regrets[t]:.3f}, cumulative regret : {np.sum(regrets):.3f}")
        
        ## update the agent
        agent.update(x=chosen_context, r=chosen_reward)
    return np.cumsum(regrets), errors


def show_result(regrets:dict, errors:dict, label_name:str, feat_dist_label:str, feat_disjoint:bool, context_label:str, reward_label:str, figsize:tuple=(14, 10)):
    NROWS, NCOLS = 2, 2
    fig, ax = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=figsize)
    
    for key in regrets:
        item = regrets[key]
        ax[0][0].plot(np.mean(item, axis=0), label=f"{label_name}={key}")
    ax[0][0].grid(True)
    ax[0][0].set_xlabel("Round")
    ax[0][0].set_ylabel(r"$R_t$")
    ax[0][0].set_title("Regret")
    ax[0][0].legend()
    
    for key in regrets:
        item = regrets[key]
        mean = np.mean(item, axis=0)
        std = np.std(item, axis=0, ddof=1)
        ax[0][1].plot(mean, label=f"{label_name}={key}")
        ax[0][1].fill_between(np.arange(T), mean-std, mean+std, alpha=0.2)
    ax[0][1].grid(True)
    ax[0][1].set_xlabel("Round")
    ax[0][1].set_ylabel(r"$R_t$")
    ax[0][1].set_title("Regret")
    ax[0][1].legend()
    
    for key in errors:
        item = errors[key]
        ax[1][0].plot(np.mean(item, axis=0), label=f"{label_name}={key}")
    ax[1][0].grid(True)
    ax[1][0].set_xlabel("Round")
    ax[1][0].set_ylabel(r"${\Vert \hat{\theta}_t - \theta_*\Vert}_2$")
    ax[1][0].set_title("Parameter Empirical Error")
    ax[1][0].set_ylim(-0.1, None)
    ax[1][0].legend()
    
    for key in errors:
        item = errors[key]
        mean = np.mean(item, axis=0)
        std = np.std(item, axis=0, ddof=1)
        ax[1][1].plot(mean, label=f"{label_name}={key}")
        ax[1][1].fill_between(np.arange(T), mean-std, mean+std, alpha=0.2)
    ax[1][1].grid(True)
    ax[1][1].set_xlabel("Round")
    ax[1][1].set_ylabel(r"${\Vert \hat{\theta}_t - \theta_*\Vert}_2$")
    ax[1][1].set_title("Parameter Empirical Error")
    ax[1][1].set_ylim(-0.1, None)
    ax[1][1].legend()
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(f"$Z${FEAT_DICT[(feat_dist_label, feat_disjoint)]}, $\sigma_\eta=${context_label}, $\sigma_\epsilon=${reward_label}, $Z$ bound={cfg.latent_bound_method}, $X$ bound={cfg.obs_bound_method}, seed={SEED}")
    return fig


if __name__ == "__main__":
    cfg = get_cfg()
    
    ## hyper-parameters
    action_spaces = cfg.action_spaces # List[int]
    num_actions = cfg.num_actions # List[int]
    d = cfg.obs_dim
    k = cfg.latent_dim
    m = cfg.num_visibles
    T = cfg.horizon
    SEED = cfg.seed
    ALPHAS = cfg.alphas
    
    if (np.array(action_spaces) == np.array(num_actions)).all():
        fixed_flag = "fixed"
    else:
        fixed_flag = "unfixed"
    
    if "T" in cfg.context_std:
        power = cfg.context_std.split("T")[-1]
        context_std = T ** float(power)
        context_label = f"$T^{{{power}}}$"
    else:
        context_std = float(cfg.context_std)
        context_label = cfg.context_std
    
    if cfg.seed_mode:
        RESULT_PATH = f"./seed_comparison/results/{cfg.mode}/{fixed_flag}/"
        FIGURE_PATH = f"./seed_comparison/figures/{cfg.mode}/{fixed_flag}/"
    else:
        RESULT_PATH = f"./results/{cfg.mode}/{fixed_flag}/"
        FIGURE_PATH = f"./figures/{cfg.mode}/{fixed_flag}/"
    
    regret_results = dict()
    error_results = dict()
    for action_size, arms in zip(action_spaces, num_actions):
        assert action_size >= arms, "The cardinality of the entire action space must be larger than the number of actions."
        assert len(num_actions) == 1 or len(ALPHAS) == 1, "Either `num_actions` or `ALPHAS` is required to have only one element."
        
        ## generate the latent feature
        Z = feature_sampler(dimension=k, feat_dist=cfg.feat_dist, size=action_size, disjoint=cfg.feat_disjoint, 
                            cov_dist=cfg.feat_cov_dist, bound=cfg.latent_feature_bound, bound_method=cfg.latent_bound_method, 
                            uniform_rng=cfg.feat_uniform_rng, random_state=(SEED*11)//3)
        
        ## generate the decoder mapping
        A = mapping_generator(latent_dim=k, obs_dim=d, distribution=cfg.map_dist, lower_bound=cfg.map_lower_bound,
                            upper_bound=cfg.map_upper_bound, uniform_rng=cfg.map_uniform_rng, random_state=((SEED*11)//3)+1)
        
        ## generate the true parameter -> corresponds to "mu"
        true_mu = param_generator(dimension=k, distribution=cfg.param_dist, disjoint=cfg.param_disjoint, bound=cfg.param_bound, 
                                  uniform_rng=cfg.param_uniform_rng, random_state=((SEED*11)//3)+2)
        
        if cfg.mode == "partial":
            Z = Z[:, :m]            # (M, k) -> (M, m)
            A = A[:, :m]            # (d, k) -> (d, m)
            true_mu = true_mu[:m]   # (k, ) -> (m, )
        
        if cfg.check_specs:
            print(f"Context std = {context_std:.6f}, SEED = {SEED}")
            print(f"Shape of the latent feature matrix : {Z.shape}")
            print(f"The maximum norm of the latent features : {np.amax([l2norm(latent) for latent in Z]):.4f}")
            print(f"Shape of the decoder mapping : {A.shape},\tNumber of reward parameters : {true_mu.shape[0]}")
            print(f"L2 norm of the true theta : {l2norm(true_mu):.4f}")
        
        ## run an experiment
        for alpha in ALPHAS:
            regrets, errors = run_trials(mode=cfg.mode, trials=cfg.trials, alpha=alpha, arms=arms, lbda=cfg.lbda, epsilon=cfg.epsilon, horizon=T, latent=Z, 
                                         decoder=A, reward_params=true_mu, noise_dist=("gaussian", "gaussian"), noise_std=(context_std, cfg.reward_std), 
                                         feat_bound=cfg.obs_feature_bound, feat_bound_method=cfg.obs_bound_method, random_state=SEED)
            if len(ALPHAS) == 1:
                key = arms
            elif len(num_actions) == 1:
                key = alpha
            regret_results[key] = regrets
            error_results[key] = errors
    
    ## save the results        
    if len(ALPHAS) == 1:
        label_name = r"$\vert \mathcal{A}\vert$"
    else:
        label_name = r"$\alpha$"
    
    fname = f"experiment_result_{datetime.now()}_latent_{cfg.latent_bound_method}_obs_{cfg.obs_bound_method}_seed_{SEED}"
    fig = show_result(regrets=regret_results, errors=error_results, label_name=label_name, feat_dist_label=cfg.feat_dist, 
                      feat_disjoint=cfg.feat_disjoint, context_label=context_label, reward_label=str(cfg.reward_std))
    save_plot(fig, path=FIGURE_PATH, fname=fname)
    save_result(result=vars(cfg), path=RESULT_PATH, fname=fname, filetype=cfg.filetype)
