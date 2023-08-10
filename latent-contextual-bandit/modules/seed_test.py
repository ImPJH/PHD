from util import *
from models import *
from prime import get_primes
from cfg import get_cfg
import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    
    d = 12
    k = 7
    M = 50000
    N = 20
    T = 30000
    feature_bound = 1.
    mapping_bound = 1.
    param_bound = 1.
    
    seeds = get_primes(start=600, end=2000)

    reward_std = 0.1
    context_std = [0, 1 / np.sqrt(T)]
    alphas = [0., 0.001, 0.01, 0.05, 0.1]
    trials = 1
    bound_method = "scaling"
    if M == N:
        fixed_flag = "fixed"
    else:
        fixed_flag = "unfixed"

    for seed in seeds:
        print(f"seed = {seed}")
        Z = feature_sampler(dimension=k, feat_dist="gaussian", size=M, disjoint=True, 
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
                print(f"alpha={alpha}\tctx_std={ctx_std:.5f}")
                regret_container = np.zeros(trials, dtype=object)
                error_container = np.zeros(trials, dtype=object)
                for trial in range(trials):
                    regrets = np.zeros(T)
                    errors = np.zeros(T)
                    agent = LinUCB(d=d, alpha=alpha, lbda=1.)
                    for t in tqdm(range(T)):
                        seed_ = seed + (100000 * trial) + t + int(1000000*alpha)
                        idx = np.random.choice(np.arange(M), size=N, replace=False)
                        latent_set = Z[idx, :]

                        ## sample the context noise and generate the observable feature
                        context_noise = subgaussian_noise(distribution="gaussian", size=(N*d), std=ctx_std, random_state=seed_).reshape(N, d)
                        action_set = latent_set @ A.T + context_noise

                        ## sample the reward noise and compute the reward
                        reward_noise = subgaussian_noise(distribution="gaussian", size=N, std=reward_std, random_state=seed_+1)
                        expected_reward = latent_set @ true_mu
                        if t == 0:
                            print(f"Reward range: [{np.min(expected_reward):.5f}, {np.max(expected_reward):.5f}]")
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
        fig.suptitle(f"seed = {seed}, $Z$ bound method = {bound_method}")
        fname = f"seed_{seed}_bound_{bound_method}_arms_{fixed_flag}"
        save_plot(fig, path='seed_comparison', fname=fname)
