from util import *
from models import *
from prime import get_primes
from cfg import get_cfg
import warnings

FEAT_DICT = {
    ("gaussian", True): r"$\sim N(0, I_k)$",
    ("gaussian", False): r"$\sim N(0, \Sigma_k)$",
    ("uniform", True): r"$\sim Unif_{I_k}$",
    ("uniform", False): r"$\sim Unif_{\Sigma_k}$"
}

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    
    d = 12
    k = 7
    M = 20
    N = 20
    T = 30000
    feature_bound = 1.
    mapping_bound = 1.
    param_bound = 1.
    
    # seeds = get_primes(start=1100, end=2000)
    seeds = [241, 251, 257, 263, 349, 379, 383, 397, 457, 463, 503, 541, 547, 463, 457, 397, 383, 643, 691, 
             719, 751, 773, 787, 829, 881, 907, 977, 1009, 1021, 1033, 1039, 1049, 1117, 1123, 1433, 1471, 
             1481, 1483, 1487, 1499, 1567, 1583, 1609, 1663, 1693, 1699, 1777, 1801, 1871, 1873, 1879, 1889, 1973, 1987, 1997]

    feat_dist = "gaussian"
    disjoint = True
    reward_std = 0.1
    context_std = [0, 1 / np.sqrt(T)]
    alphas = [0.3, 0.5, 0.7, 0.9]
    trials = 1
    bound_method = "scaling"
    if M == N:
        fixed_flag = "fixed"
    else:
        fixed_flag = "unfixed"

    for seed in seeds:
        print(f"seed = {seed}")
        Z = feature_sampler(dimension=k, feat_dist=feat_dist, size=M, disjoint=disjoint, cov_dist="gaussian",
                            bound=feature_bound, bound_method=bound_method, random_state=seed)
        A = mapping_generator(latent_dim=k, obs_dim=d, distribution="uniform", 
                              upper_bound=mapping_bound, random_state=(seed*11)//3)
        true_mu = param_generator(dimension=k, distribution="uniform", disjoint=True, 
                                  bound=param_bound, random_state=((seed*11)//3)+2)
        B = left_pseudo_inverse(A)
        true_theta = B.T @ true_mu
        
        result_container = []
        for ctx_std in context_std:
            result = dict()
            for alpha in alphas:
                print(f"alpha = {alpha}\tctx_std = {ctx_std:.5f}\tArms : {fixed_flag}")
                regret_container = np.zeros(trials, dtype=object)
                error_container = np.zeros(trials, dtype=object)
                for trial in range(trials):
                    regrets = np.zeros(T)
                    errors = np.zeros(T)
                    agent = LinUCB(d=d, alpha=alpha, lbda=1.)
                    for t in tqdm(range(T)):
                        seed_ = seed + (100000 * trial) + t + int(1000000*alpha)
                        if M == N:
                            idx = np.arange(M)
                        else:
                            idx = np.random.choice(np.arange(M), size=N, replace=False)
                        latent_set = Z[idx, :]

                        ## sample the context noise and generate the observable feature
                        context_noise = subgaussian_noise(distribution="gaussian", size=(N*d), std=ctx_std, random_state=seed_).reshape(N, d)
                        action_set = latent_set @ A.T + context_noise

                        ## sample the reward noise and compute the reward
                        reward_noise = subgaussian_noise(distribution="gaussian", size=N, std=reward_std, random_state=seed_+1)
                        expected_reward = latent_set @ true_mu
                        if t == 0:
                            print(f"Reward range : [{np.min(expected_reward):.5f}, {np.max(expected_reward):.5f}]")
                        true_reward = expected_reward + reward_noise
                        optimal_arm = np.argmax(expected_reward)
                        optimal_reward = expected_reward[optimal_arm]

                        ## choose the best action
                        chosen_arm = agent.choose(action_set)
                        chosen_reward = true_reward[chosen_arm]
                        chosen_context = action_set[chosen_arm]

                        ## compute the regret and the theta distances
                        regrets[t] = optimal_reward - expected_reward[chosen_arm]
                        errors[t] = l2norm(true_theta - agent.theta_hat)

                        ## update the agent
                        agent.update(x=chosen_context, r=chosen_reward)

                    regret_container[trial] = np.cumsum(regrets)
                    error_container[trial] = errors
                result[alpha] = (regret_container, error_container)
            result_container.append(result)
        
        NROWS, NCOLS = 2, 2
        title = r"$\sigma_\eta$"
        fig, ax = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=(13, 10))
        
        zero_result, noisy_result = result_container
        for key in zero_result:
            regrets, errors = zero_result[key]
            ax[0][0].plot(np.mean(regrets, axis=0), label=f"\u03B1={key}")
            ax[0][0].grid(True)
            ax[0][0].set_xlabel("Round")
            ax[0][0].set_ylabel(r"$R_t$")
            ax[0][0].set_title(f"Regret when {title} = 0")
            ax[0][0].legend()

            ax[0][1].plot(np.mean(errors, axis=0), label=f"\u03B1={key}")
            ax[0][1].grid(True)
            ax[0][1].set_xlabel("Round")
            ax[0][1].set_ylabel(r"$\Vert \hat{\theta}_t - \theta_*\Vert$")
            ax[0][1].set_ylim(-0.1, None)
            ax[0][1].set_title(f"Parameter empirical error when {title} = 0")
            ax[0][1].legend()

        for key in noisy_result:
            noisy_regrets, noisy_errors = noisy_result[key]
            ax[1][0].plot(np.mean(noisy_regrets, axis=0), label=f"\u03B1={key}")
            ax[1][0].grid(True)
            ax[1][0].set_xlabel("Round")
            ax[1][0].set_ylabel(r"$R_t$")
            ax[1][0].set_title(f"Regret when {title} = {context_std[-1]:.5f}")
            ax[1][0].legend()

            ax[1][1].plot(np.mean(noisy_errors, axis=0), label=f"\u03B1={key}")
            ax[1][1].grid(True)
            ax[1][1].set_xlabel("Round")
            ax[1][1].set_ylabel(r"$\Vert \hat{\theta}_t - \theta_*\Vert$")
            ax[1][1].set_ylim(-0.1, None)
            ax[1][1].set_title(f"Parameter empirical error when {title} = {context_std[-1]:.5f}")
            ax[1][1].legend()

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.suptitle(f"$Z${FEAT_DICT[(feat_dist, disjoint)]} seed = {seed}, $Z$ bound method = {bound_method}")
        fname = f"feat_{feat_dist}_disjoint_{disjoint}_seed_{seed}_bound_{bound_method}_arms_{fixed_flag}_again"
        save_plot(fig, path='seed_comparison', fname=fname)
        plt.close('all')
