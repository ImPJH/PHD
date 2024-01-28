from cfg2 import get_cfg
from models import *
from util import *

FEAT_DICT = {
    ("gaussian", True): r"$\sim N(0, I_k)$",
    ("gaussian", False): r"$\sim N(0, \Sigma_k)$",
    ("uniform", True): r"$\sim Unif_{I_k}$",
    ("uniform", False): r"$\sim Unif_{\Sigma_k}$"
}

# MOTHER_PATH = "/home/sungwoopark/bandit-research/hop/modules"
MOTHER_PATH = "./model_test"

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

def run_trials(trials:int, arms:int, lbda:float, horizon:int, latent:np.ndarray, decoder:np.ndarray, 
               reward_params:np.ndarray, noise_dist:Tuple[str], noise_std:List[Union[float, List[float]]], 
               feat_bound:float, feat_bound_method:str, random_state:int, verbose:bool=False):
    obs_dim, _ = decoder.shape
    action_space_size = latent.shape[0]
    regret_container = np.zeros(trials, dtype=object)
    # error_container = np.zeros(trials, dtype=object)
        
    for trial in range(trials):
        agent = HOPlinear(d=obs_dim, arms=arms, lbda=lbda, reward_std=cfg.reward_std, delta=cfg.delta, horizon=horizon)

        random_state_ = random_state + (12221*(trial+1)) + (49997*arms)        
        inherent_rewards = param_generator(dimension=arms, distribution=cfg.bias_dist, disjoint=cfg.param_disjoint, 
                                           bound=cfg.param_bound, uniform_rng=cfg.param_uniform_rng, random_state=random_state_)

        
        np.random.seed(random_state_)
        idx = np.random.choice(np.arange(action_space_size), size=arms, replace=False)
        latent_ = latent[idx, :].copy()
        action_space_size = arms        
        
        print(f"Running seed : {random_state_}, Shape of the latent features : {latent_.shape}")
        regrets = run(agent=agent, horizon=horizon, latent=latent_, decoder=decoder, reward_params=reward_params, 
                      inherent_rewards=inherent_rewards, noise_dist=noise_dist, noise_std=noise_std, 
                      feat_bound=feat_bound, feat_bound_method=feat_bound_method, random_state=random_state_, verbose=verbose)
        regret_container[trial] = regrets
        # error_container[trial] = errors
    
    # return regret_container, error_container
    return regret_container

def run(agent:HOPlinear, horizon:int, latent:np.ndarray, decoder:np.ndarray, reward_params:np.ndarray, 
        inherent_rewards:np.ndarray, noise_dist:Tuple[str], noise_std:List[Union[float, List[float]]], 
        feat_bound:float, feat_bound_method:str, random_state:int, verbose:bool):
    obs_dim, _ = decoder.shape
    arms, latent_dim = latent.shape
    context_noise_dist, reward_noise_dist = noise_dist
    context_noise_std, reward_noise_std = noise_std
    
    ## make the mapped features in advance before iteration
    observe_set = latent @ decoder.T          # (K, m) @ (m, d) -> (K, d)
    decoder_inv = left_pseudo_inverse(decoder)
    obs_mu = decoder_inv.T @ reward_params  # (d, m) @ (m, ) -> (d, )
    obs_mu = np.concatenate([obs_mu, inherent_rewards], axis=0) # (d, ) -> (d+N, )
    
    regrets = np.zeros(horizon, dtype=float)
    # errors = np.zeros(horizon, dtype=float)

    if not verbose:
        bar = tqdm(range(horizon))
    else:
        bar = range(horizon)
    
    for t in bar:
        if random_state is not None:
            random_state_ = random_state + int(21112 * t)
            np.random.seed(random_state_)
        
        ## sample the context noise and construct the observable features
        if isinstance(context_noise_std, float):
            context_noise = subgaussian_noise(distribution=context_noise_dist, size=(arms*obs_dim), 
                                              std=context_noise_std, random_state=random_state_).reshape(arms, d)
        else:
            context_noise = subgaussian_noise(distribution=context_noise_dist, size=(arms*obs_dim), 
                                              std=context_noise_std[t], random_state=random_state_).reshape(arms, d)            
        action_set = observe_set + context_noise
        
        ## bound the action set
        if feat_bound_method is not None:
            bounding(type="feature", v=action_set, bound=feat_bound, method=feat_bound_method)
        action_set = np.concatenate([action_set, np.identity(arms)], axis=1)
        
        ## sample the reward noise and compute the reward
        reward_noise = subgaussian_noise(distribution=reward_noise_dist, size=arms, std=reward_noise_std, random_state=random_state_)
        expected_reward = latent @ reward_params + inherent_rewards
        
        if t == 0:
            print(f"Number of actions : {action_set.shape[0]}\tReward range : [{np.amin(expected_reward):.5f}, {np.amax(expected_reward):.5f}]")
        realized_reward = expected_reward + reward_noise
        optimal_arm = np.argmax(expected_reward)
        optimal_reward = expected_reward[optimal_arm]
        
        ## choose the best action
        chosen_arm = agent.choose(action_set)
        chosen_reward = realized_reward[chosen_arm]
        chosen_context = action_set[chosen_arm]
        
        ## compute the regret and errors, if necessary
        regrets[t] = optimal_reward - expected_reward[chosen_arm]
        # errors[t] = l2norm(obs_mu - agent.theta_hat)
        
        if verbose: 
            print(f"round {t+1}\toptimal action : {optimal_arm}\toptimal reward : {optimal_reward:.3f}")
            print(f"\tchosen action : {chosen_arm}\trealized reward : {chosen_reward:.3f}, expected reward: {expected_reward[chosen_arm]:.3f}")
            # print(f"\ttheta empirical error : {errors[t]}, instance regret : {regrets[t]:.3f}, cumulative regret : {np.sum(regrets):.3f}")
            print(f"\tinstance regret : {regrets[t]:.3f}, cumulative regret : {np.sum(regrets):.3f}")
        
        ## update the agent
        agent.update(x=chosen_context, r=chosen_reward)
    return np.cumsum(regrets)


def show_result(regrets:dict, horizon:int, label_name:str, figsize:tuple=(6, 5)):
    # NROWS, NCOLS = 1, 2
    # fig, (ax1, ax2) = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=figsize)
    
    # for key in regrets:
    #     item = regrets[key]
    #     ax1.plot(np.mean(item, axis=0), label=f"{label_name}={key}")
    # ax1.grid(True)
    # ax1.set_xlabel("Round")
    # ax1.set_ylabel("Regret")
    # # ax1.set_title(r"10 Trials Average $R_T$")
    # ax1.legend()
    
    # for key in regrets:
    #     item = regrets[key]
    #     mean = np.mean(item, axis=0)
    #     std = np.std(item, axis=0, ddof=1)
    #     ax2.plot(mean, label=f"{label_name}={key}")
    #     ax2.fill_between(np.arange(T), mean-std, mean+std, alpha=0.2)
    # ax2.grid(True)
    # ax2.set_xlabel("Round")
    # ax2.set_ylabel("Regret")
    # # ax2.set_title(r"10 Trials Average $R_T \pm 1SD$")
    # ax2.legend()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 마커 스타일과 색상 설정
    # markers = ['o', 's', '^', 'd', 'p']
    label_dict = {
        "gaussian": r"$\mathcal{N}(0,1)$",
        "uniform": r"$\mathrm{Unif}(-1,1)$"
    }
    bias_label = r"$\{\delta_a\}_{a=1}^K\sim$"
    param_label = r"$\{\theta_*^i\}_{i=1}^m\sim$"
    
    if label_name == r"$\vert \mathcal{A}_t\vert$":
        title = r"$m=$" + f"{cfg.num_visibles[-1]}"
    elif label_name == "$m$":
        title = r"$\vert \mathcal{A}_t\vert$=" + f"{cfg.num_actions[-1]}"
    
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
    ax.set_title(f"{bias_label}{label_dict[cfg.bias_dist]}, {param_label}{label_dict[cfg.param_dist]}, {title}")
    ax.legend()
    
    fig.tight_layout()  
    # fig.tight_layout(rect=[0, 0, 1, 0.95])
    # fig.suptitle(f"$Z${FEAT_DICT[(feat_dist_label, feat_disjoint)]}, $\sigma_\eta=${context_label}, $\sigma_\epsilon=${reward_label}, seed={SEED}, num_visibles={cfg.num_visibles}")
    return fig


if __name__ == "__main__":
    cfg = get_cfg()
    
    ## hyper-parameters
    action_space_size = cfg.action_spaces # int
    num_actions = cfg.num_actions # List[int]
    d = cfg.obs_dim
    k = cfg.latent_dim
    m = cfg.num_visibles
    T = cfg.horizon
    SEED = cfg.seed
    
    if "T" in cfg.context_std:
        power = cfg.context_std.split("T")[-1]
        if cfg.context_std_variant:
            context_std = [(t+1) ** float(power) for t in range(T)]
        else:
            context_std = T ** float(power)
        context_label = f"$T^{{{power}}}$"
    else:
        context_std = float(cfg.context_std)
        context_label = cfg.context_std
    
    RESULT_PATH = f"{MOTHER_PATH}/results"
    FIGURE_PATH = f"{MOTHER_PATH}/figures"

    regret_results = dict()

    ## generate the latent feature
    Z = feature_sampler(dimension=k, feat_dist=cfg.feat_dist, size=action_space_size, disjoint=cfg.feat_disjoint, cov_dist=cfg.feat_cov_dist, 
                        bound=cfg.latent_feature_bound, bound_method=cfg.latent_bound_method, uniform_rng=cfg.feat_uniform_rng, random_state=SEED)
    
    ## generate the decoder mapping
    A = mapping_generator(latent_dim=k, obs_dim=d, distribution=cfg.map_dist, lower_bound=cfg.map_lower_bound, 
                          upper_bound=cfg.map_upper_bound, uniform_rng=cfg.map_uniform_rng, random_state=SEED+1)
    
    for m in cfg.num_visibles:
        ## generate the true parameter -> corresponds to "mu"
        Z_ = Z[:, :m]  # (M, k) -> (M, m)
        A_ = A[:, :m]  # (d, k) -> (d, m)
        true_mu = param_generator(dimension=m, distribution=cfg.param_dist, disjoint=cfg.param_disjoint, 
                                  bound=cfg.param_bound, uniform_rng=cfg.param_uniform_rng, random_state=SEED-1)
        for arms in num_actions:
            if cfg.check_specs:
                print(f"Arms : {arms}, Feature : {cfg.feat_dist}, Bias : {cfg.bias_dist}, Parameter : {cfg.param_dist}")
                print(f"Context std : {cfg.context_std}, Original seed : {SEED}, Number of influential variables : {m}")
                print(f"The maximum norm of the latent features : {np.amax([l2norm(feat) for feat in Z]):.4f}")
                print(f"Shape of the decoder mapping : {A.shape},\tNumber of reward parameters : {true_mu.shape[0]}")
                print(f"Lambda : {cfg.lbda}\tL2 norm of the true mu : {l2norm(true_mu):.4f}")

            if len(num_actions) > 1:
                assert len(cfg.num_visibles) == 1
                key = arms
            elif len(cfg.num_visibles) > 1:
                assert len(num_actions) == 1
                key = m

            ## run an experiment
            regrets = run_trials(trials=cfg.trials, arms=arms, lbda=cfg.lbda, horizon=T, latent=Z_, decoder=A_, 
                                reward_params=true_mu, noise_dist=("gaussian", "gaussian"), noise_std=[context_std, cfg.reward_std], 
                                feat_bound=cfg.obs_feature_bound, feat_bound_method=cfg.obs_bound_method, random_state=SEED)
            
            regret_results[key] = regrets
            # error_results[arms] = errors

    ## save the results
    if len(num_actions) > 1:
        assert len(cfg.num_visibles) == 1
        label_name = r"$\vert \mathcal{A}_t\vert$"
    elif len(cfg.num_visibles) > 1:
        assert len(num_actions) == 1
        label_name = r"$m$"
    fname = (f"{SEED}_noise_{cfg.context_std}_nvisibles_{cfg.num_visibles}_{METHOD_DICT[cfg.latent_bound_method]}_" 
             f"feat_{DIST_DICT[cfg.feat_dist]}_{DEP_DICT[cfg.feat_disjoint]}_map_{DIST_DICT[cfg.map_dist]}_"
             f"bias_{DIST_DICT[cfg.bias_dist]}_param_{DIST_DICT[cfg.param_dist]}_{DEP_DICT[cfg.param_disjoint]}")
    fig = show_result(regrets=regret_results, horizon=T, label_name=label_name)
    save_plot(fig, path=FIGURE_PATH, fname=fname)
    save_result(result=vars(cfg), path=RESULT_PATH, fname=fname, filetype=cfg.filetype)
