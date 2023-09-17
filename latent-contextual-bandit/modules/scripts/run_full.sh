#!/bin/bash

seeds=$(python3 get_primes.py --nums 4000 5000)
# seeds=(109 151 157 163 167 173 179 191 197 199 211 229 233 239 241 257 263 269 271 277 281 283 307 311 313 331 337 347 349 353 383 389 397 401 409 419 421 433 443 449 457 463 467 491 499 509 521 541 563 569 571 587 593 599 601 607 613 641 643 647 659 661 673 691 701 709 719 727 733 739 743 751 757 769 787 797 809 811 821 907 1583 1609)
# echo $seeds | xargs -n 1 -P 4 ./run_seeds_main.sh

# export -f run_trial

# run_trial() {
#     seed=$1
#     echo "Full Mode"
#     echo "When an arm set is not fixed"
#     python main.py --mode full --alphas 0.0 0.01 0.1 0.3 0.5 --trials 3 --action_spaces 50000 --num_actions 20 --obs_dim 12 --latent_dim 10 --horizon 30000 --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound 1 --obs_feature_bound 1 --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std T-0.5 --reward_std 0.1 --param_dist uniform --param_bound 1 --param_uniform_rng -1 1 --param_disjoint --check_specs --seed_mode
#     echo "When an arm set is fixed"
#     python main.py --mode full --alphas 0.5 0.55 0.6 0.8 1.0 --trials 3 --action_spaces 20 --num_actions 20 --obs_dim 12 --latent_dim 10 --horizon 30000 --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound 1 --obs_feature_bound 1 --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std T-0.5 --reward_std 0.1 --param_dist uniform --param_bound 1 --param_uniform_rng -1 1 --param_disjoint --check_specs --seed_mode --fixed
# }

# export -f run_trial
# echo $seeds | parallel -j 10 run_trial {}

IFS=','
echo "Context Noise is 1 over sqrt(T)"
for seed in $seeds; do
    echo "Full Mode"
    echo "When an arm set is not fixed"
    python main.py --mode full --alphas 0.0 0.01 0.1 0.3 0.5 --trials 8 --action_spaces 50000 --num_actions 20 --obs_dim 10 --latent_dim 8 --horizon 30000 --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound 1 --obs_feature_bound 1 --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std T-0.5 --reward_std 0.1 --param_dist uniform --param_bound 1 --param_uniform_rng -1 1 --param_disjoint --check_specs --seed_mode
    echo "When an arm set is fixed"
    python main.py --mode full --alphas 0.4 0.45 0.5 0.55 0.6 --trials 8 --action_spaces 20 --num_actions 20 --obs_dim 10 --latent_dim 8 --horizon 30000 --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound 1 --obs_feature_bound 1 --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std T-0.5 --reward_std 0.1 --param_dist uniform --param_bound 1 --param_uniform_rng -1 1 --param_disjoint --check_specs --seed_mode --fixed
done

IFS=','
echo "Context Noise is 0"
for seed in $seeds; do
    echo "Full Mode"
    echo "When an arm set is not fixed"
    python main.py --mode full --alphas 0.0 0.01 0.1 0.3 0.5 --trials 8 --action_spaces 50000 --num_actions 20 --obs_dim 10 --latent_dim 8 --horizon 30000 --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound 1 --obs_feature_bound 1 --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std 0 --reward_std 0.1 --param_dist uniform --param_bound 1 --param_uniform_rng -1 1 --param_disjoint --check_specs --seed_mode
    echo "When an arm set is fixed"
    python main.py --mode full --alphas 0.4 0.45 0.5 0.55 0.6 --trials 8 --action_spaces 20 --num_actions 20 --obs_dim 10 --latent_dim 8 --horizon 30000 --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound 1 --obs_feature_bound 1 --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std 0 --reward_std 0.1 --param_dist uniform --param_bound 1 --param_uniform_rng -1 1 --param_disjoint --check_specs --seed_mode --fixed
done
