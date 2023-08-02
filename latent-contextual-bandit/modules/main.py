from cfg import get_cfg
from models import *
from util import *

FEAT_DICT = {
    ("gaussian", True): r"$\sim N(0, I)$",
    ("gaussian", False): r"$\sim N(0, \Sigma)$",
    ("uniform", True): r"$\sim Unif_{I}$",
    ("uniform", False): r"$\sim Unif_{\Sigma}$"
}


def run(mode:str, agent:Union[LinUCB, LineGreedy, PartialLinUCB], num_actions:int, horizon:int, obs:np.ndarray, latent:np.ndarray, 
        reward_params:np.ndarray, reward_noise_var:float, use_tqdm:bool, verbose:bool, random_state:int, is_fixed:bool, dimension_same:bool):
    
    ## check whether view a progress bar or not
    if use_tqdm:
        bar = tqdm(range(horizon))
    else:
        bar = range(horizon)
        
    ## check the mode
    if mode == "partial":
        assert is_fixed, f"The mode '{mode}' requires arm sets to be fixed."
        inherent_rewards = param_generator(dimension=num_actions, distribution=cfg.param_dist, disjoint=cfg.param_disjoint, 
                                           uniform_rng=cfg.param_uniform_rng, random_state=random_state)
    else:
        inherent_rewards = 0.
    
    regrets = np.zeros(horizon) ## array for instance regrets
    theta_errors = np.zeros(horizon) ## array for empirical error between theta_hat and true_theta
    for t in bar:
        if is_fixed:
            indices = np.arange(num_actions)
        else:
            ## assume that each action is different to one another in the same arm set
            indices = np.random.choice(np.arange(action_space_size), size=num_actions, replace=False)
            
        ## generate the reward noise (num_actions, )
        reward_noise = subgaussian_noise(distribution="gaussian", size=num_actions, random_state=random_state+t, std=reward_noise_var)
        
        ## observe the actions (num_actions, d), (num_actions, k or m)
        action_set, latent_action_set = obs[indices], latent[indices]
        if mode == "partial":
            action_set = np.concatenate([action_set, np.identity(num_actions)], axis=1)
        
        ## compute the rewards and the optimal actions
        expected_rewards = latent_action_set @ reward_params + inherent_rewards
        true_rewards = expected_rewards + reward_noise
        if t == 0:
            print(f"Mode: {mode},\tFixed Arm Set: {is_fixed},\tReward range = [{np.amin(expected_rewards):.4f}, {np.amax(expected_rewards):.4f}]")
        optimal_action = np.argmax(expected_rewards)
        optimal_reward = expected_rewards[optimal_action]
        
        ## choose the best action and compute the empirical error between theta_hat and true_theta
        if isinstance(agent, LineGreedy):
            chosen_action = agent.choose(action_set)
        else:
            chosen_action, theta_hat = agent.choose(action_set)
            if dimension_same:
                theta_errors[t] = l2norm((theta_hat - reward_params))
        chosen_reward = true_rewards[chosen_action]
        chosen_context = action_set[chosen_action]
        
        ## compute the regret
        regrets[t] = optimal_reward - expected_rewards[chosen_action]
        
        ## update the agent
        agent.update(chosen_context, chosen_reward)
        
        if verbose: 
            print(f"round {t+1}\toptimal action : {optimal_action}\toptimal reward : {optimal_reward:.3f}")
            print(f"\tchosen action : {chosen_action}\trealized reward : {chosen_reward:.3f}, expected reward: {expected_rewards[chosen_action]:.3f}")
            print(f"\ttheta empirical error : {theta_errors[t]}, instance regret : {regrets[t]:.3f}, cumulative regret : {np.sum(regrets):.3f}")
            
    return regrets, theta_errors


def run_trials(trials:int, dimension:int, alphas:list, arms:list, mode:str, agent_type:str, horizon:int, obs:np.ndarray, 
               latent:np.ndarray, reward_params:np.ndarray, reward_noise_var:float, lbda:float, epsilon:float,
               use_tqdm:bool, verbose:bool, random_state:int, is_fixed:bool, dimension_same:bool):

    assert agent_type.lower() in ["linucb", "linegreedy", "partial"], f"The agent type should be in ['LinUCB', 'LineGreedy', 'Partial'], but {agent_type} is passed."
    assert len(alphas) == 1 or len(arms) == 1, "Either the length of alphas or that of arms must be 1."

    regret_result = dict()
    theta_result = dict()
    for alpha in alphas:
        for arm in arms:
            alpha_text = "\u03B1"
            arms_text = "|A|"
            print(f"{alpha_text} = {alpha},\t{arms_text} = {arm}")
            result_container = np.zeros(trials, dtype=object)
            theta_error_container = np.zeros(trials, dtype=object)
            for trial in range(trials):
                if agent_type.lower() == "linucb":
                    agent = LinUCB(d=dimension, alpha=alpha, lbda=lbda)
                elif agent_type.lower() == "partial":
                    agent = PartialLinUCB(d=dimension, num_actions=arm, alpha=alpha, lbda=lbda)
                else:
                    agent = LineGreedy(d=dimension, alpha=alpha, lbda=lbda, epsilon=epsilon)
                regret, theta_error = run(mode=mode, agent=agent, num_actions=arm, horizon=horizon, obs=obs, latent=latent, reward_params=reward_params, 
                                          reward_noise_var=reward_noise_var, use_tqdm=use_tqdm, verbose=verbose, random_state=random_state+(trial*horizon), 
                                          is_fixed=is_fixed, dimension_same=dimension_same)
                result_container[trial] = np.cumsum(regret)
                theta_error_container[trial] = theta_error
                
            if len(alphas) == 1:
                key = arm
            else:
                key = alpha
            regret_result[key] = result_container
            if np.concatenate(theta_error_container).all():
                theta_result[key] = theta_error_container
            
    if len(alphas) == 1:
        label_name = r"$\vert \mathcal{A}\vert$"
    else:
        label_name = r"$\alpha$"
    
    return regret_result, theta_result, label_name


def show_result(regrets:dict, errors:dict, label_name:str, feat_dist_label:str, feat_disjoint:bool, 
                context_label:str, reward_label:str, figsize:tuple=(16, 11)):
    if errors is not None:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    
        for key, value in regrets.items():
            mean = np.mean(value, axis=0)
            ax[0][0].plot(mean, label=f"{label_name}={key}")
        ax[0][0].grid(True)
        ax[0][0].set_xlabel("Round")
        ax[0][0].set_ylabel(r"$R_t$")
        ax[0][0].set_title("Mean Regret")
        ax[0][0].legend()
        
        for key, value in regrets.items():
            mean = np.mean(value, axis=0)
            std = np.std(value, axis=0, ddof=1)
            ax[0][1].plot(mean, label=f"{label_name}={key}")
            ax[0][1].fill_between(np.arange(T), mean-std, mean+std, alpha=0.2)
        ax[0][1].grid(True)
        ax[0][1].set_xlabel("Round")
        ax[0][1].set_ylabel(r"$R_t$")
        ax[0][1].set_title(r"Mean Regret $\pm$ 1 std")
        ax[0][1].legend()
        
        for key, value in errors.items():
            mean = np.mean(value, axis=0)
            ax[1][0].plot(mean, label=f"{label_name}={key}")
        ax[1][0].grid(True)
        ax[1][0].set_xlabel("Round")
        ax[1][0].set_ylabel(r"${\Vert \hat{\theta}_{t}-{\theta}_* \Vert}_2$")
        ax[1][0].set_title("Mean Empirical Error")
        ax[1][0].legend()
        
        for key, value in errors.items():
            mean = np.mean(value, axis=0)
            std = np.std(value, axis=0, ddof=1)
            ax[1][1].plot(mean, label=f"{label_name}={key}")
            ax[1][1].fill_between(np.arange(T), mean-std, mean+std, alpha=0.2)
        ax[1][1].grid(True)
        ax[1][1].set_xlabel("Round")
        ax[1][1].set_ylabel(r"${\Vert \hat{\theta}-{\theta_*} \Vert}_2$")
        ax[1][1].set_title(r"Mean Empirical Error $\pm$ 1 std")
        ax[1][1].legend()
    
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

        for key, value in regrets.items():
            mean = np.mean(value, axis=0)
            ax1.plot(mean, label=f"{label_name}={key}")
        ax1.grid(True)
        ax1.set_xlabel("Round")
        ax1.set_ylabel(r"$R_t$")
        ax1.set_title("Mean Regret")
        ax1.legend()
        
        for key, value in regrets.items():
            mean = np.mean(value, axis=0)
            std = np.std(value, axis=0, ddof=1)
            ax2.plot(mean, label=f"{label_name}={key}")
            ax2.fill_between(np.arange(T), mean-std, mean+std, alpha=0.2)
        ax2.grid(True)
        ax2.set_xlabel("Round")
        ax2.set_ylabel(r"$R_t$")
        ax2.set_title(r"Mean Regret $\pm$ 1 std")
        ax2.legend()
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(f"Z{FEAT_DICT[(feat_dist_label, feat_disjoint)]}, $\sigma_\eta=${context_label}, $\sigma_\epsilon=${reward_label}, bound={cfg.bound_method}")    
    return fig


if __name__ == "__main__":
    ## import argumens from terminal
    cfg = get_cfg()
    
    ## hyper-parameters
    trials = cfg.trials
    mode = cfg.mode
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
    
    if "T" in cfg.context_var:
        exp = cfg.context_var.split("T")[-1]
        context_var = T ** float(exp)
        context_label = f"$T^{float(exp)}$"
    else:
        context_var = float(cfg.context_var)
        context_label = cfg.context_var
    print(f"Context std = {context_var}")
        
    if action_space_size == num_actions:
        fixed_label = "fixed"
        fixed_flag = True
    else:
        fixed_label = "unfixed"
        fixed_flag = False
    
    RESULT_PATH = f"./results/{mode}/{fixed_label}"
    FIGURE_PATH = f"./figures/{mode}/{fixed_label}"
    
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
        
    print(f"Mapping shape: {A.shape}\tParameter shape: {theta.shape}")
    regret_result, theta_result, label_name = run_trials(trials=trials, dimension=d, alphas=ALPHAS, arms=num_actions, mode=mode, 
                                                         agent_type=cfg.agent_type, horizon=T, obs=X, latent=Z, reward_params=theta, 
                                                         reward_noise_var=cfg.reward_noise_var, lbda=cfg.lbda, epsilon=cfg.epsilon, use_tqdm=cfg.tqdm, 
                                                         verbose=cfg.verbose, random_state=RUN_SEED, is_fixed=fixed_flag, dimension_same=(d==k))
    
    ## save the result plot
    fname = f"experiment_result_{datetime.now()}_{cfg.bound_method}"
    fig = show_result(regrets=regret_result, errors=theta_result, label_name=label_name, feat_dist_label=cfg.feat_dist, 
                      feat_disjoint=cfg.feat_disjoint, context_label=context_label, reward_label=str(cfg.reward_noise_var))
    save_plot(fig, path=FIGURE_PATH, fname=fname)
    
    out = vars(cfg)
    save_result(result=out, path=RESULT_PATH, fname=fname, filetype=cfg.filetype)

