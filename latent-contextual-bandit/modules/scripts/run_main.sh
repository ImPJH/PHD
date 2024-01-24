#!/bin/bash

seeds=$(python3 get_primes.py --nums 5 200)

M=100000 # action space size
d=10 # dimension of "mapped" features
k=8 # dimension of "latent" features
half=$((k / 2))
almost=$((k - 1))
feat_bound=1
sigma=0.1
param_bound=1
T=50000 # total horizon
# K=15
delta=0.000001
TRIALS=7 # number of trials
lambda=10

echo "Partial Mode"
IFS=','
# for seed in "${seeds[@]}"; do
# for seed in {350..399}; do
for seed in $seeds; do
    # echo "Context Noise is 0"
    # python main.py --mode partial --trials $TRIALS --action_spaces $M --num_actions 10 15 20 25 30 --obs_dim $d --latent_dim $k --num_visibles 1 --horizon $T --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound 1 --obs_feature_bound 1 --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std 0 --reward_std 0.1 --param_dist gaussian --param_bound 1 --param_disjoint --bias_dist gaussian --check_specs --seed_mode
    # python main.py --mode partial --trials $TRIALS --action_spaces $M --num_actions 10 15 20 25 30 --obs_dim $d --latent_dim $k --num_visibles $half --horizon $T --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound 1 --obs_feature_bound 1 --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std 0 --reward_std 0.1 --param_dist gaussian --param_bound 1 --param_disjoint --bias_dist gaussian --check_specs --seed_mode
    # python main.py --mode partial --trials $TRIALS --action_spaces $M --num_actions 10 15 20 25 30 --obs_dim $d --latent_dim $k --num_visibles $almost --horizon $T --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound 1 --obs_feature_bound 1 --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std 0 --reward_std 0.1 --param_dist gaussian --param_bound 1 --param_disjoint --bias_dist gaussian --check_specs --seed_mode

    # python main.py --mode partial --trials $TRIALS --action_spaces $M --num_actions 10 15 20 25 30 --obs_dim $d --latent_dim $k --num_visibles 1 --horizon $T --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound 1 --obs_feature_bound 1 --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std 0 --reward_std 0.1 --param_dist gaussian --param_bound 1 --param_disjoint --bias_dist uniform --check_specs --seed_mode
    # python main.py --mode partial --trials $TRIALS --action_spaces $M --num_actions 10 15 20 25 30 --obs_dim $d --latent_dim $k --num_visibles $half --horizon $T --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound 1 --obs_feature_bound 1 --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std 0 --reward_std 0.1 --param_dist gaussian --param_bound 1 --param_disjoint --bias_dist uniform --check_specs --seed_mode
    # python main.py --mode partial --trials $TRIALS --action_spaces $M --num_actions 10 15 20 25 30 --obs_dim $d --latent_dim $k --num_visibles $almost --horizon $T --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound 1 --obs_feature_bound 1 --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std 0 --reward_std 0.1 --param_dist gaussian --param_bound 1 --param_disjoint --bias_dist uniform --check_specs --seed_mode
    
    echo "Context Noise is 1 over sqrt(T)"
    python main.py --mode partial --trials $TRIALS --action_spaces $M --num_actions 10 15 20 25 30 --obs_dim $d --latent_dim $k --lbda $lambda --num_visibles 1 --horizon $T --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound 1 --obs_feature_bound 1 --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std T-0.5 --reward_std 0.1 --param_dist gaussian --param_bound 1 --param_disjoint --bias_dist gaussian --check_specs --seed_mode
    python main.py --mode partial --trials $TRIALS --action_spaces $M --num_actions 10 15 20 25 30 --obs_dim $d --latent_dim $k --lbda $lambda --num_visibles $half --horizon $T --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound 1 --obs_feature_bound 1 --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std T-0.5 --reward_std 0.1 --param_dist gaussian --param_bound 1 --param_disjoint --bias_dist gaussian --check_specs --seed_mode
    python main.py --mode partial --trials $TRIALS --action_spaces $M --num_actions 10 15 20 25 30 --obs_dim $d --latent_dim $k --lbda $lambda --num_visibles $almost --horizon $T --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound 1 --obs_feature_bound 1 --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std T-0.5 --reward_std 0.1 --param_dist gaussian --param_bound 1 --param_disjoint --bias_dist gaussian --check_specs --seed_mode

    python main.py --mode partial --trials $TRIALS --action_spaces $M --num_actions 10 15 20 25 30 --obs_dim $d --latent_dim $k --lbda $lambda --num_visibles 1 --horizon $T --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound 1 --obs_feature_bound 1 --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std T-0.5 --reward_std 0.1 --param_dist uniform --param_bound 1 --param_disjoint --bias_dist gaussian --check_specs --seed_mode
    python main.py --mode partial --trials $TRIALS --action_spaces $M --num_actions 10 15 20 25 30 --obs_dim $d --latent_dim $k --lbda $lambda --num_visibles $half --horizon $T --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound 1 --obs_feature_bound 1 --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std T-0.5 --reward_std 0.1 --param_dist uniform --param_bound 1 --param_disjoint --bias_dist gaussian --check_specs --seed_mode
    python main.py --mode partial --trials $TRIALS --action_spaces $M --num_actions 10 15 20 25 30 --obs_dim $d --latent_dim $k --lbda $lambda --num_visibles $almost --horizon $T --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound 1 --obs_feature_bound 1 --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std T-0.5 --reward_std 0.1 --param_dist uniform --param_bound 1 --param_disjoint --bias_dist gaussian --check_specs --seed_mode

    python main.py --mode partial --trials $TRIALS --action_spaces $M --num_actions 10 15 20 25 30 --obs_dim $d --latent_dim $k --lbda $lambda --num_visibles 1 --horizon $T --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound 1 --obs_feature_bound 1 --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std T-0.5 --reward_std 0.1 --param_dist gaussian --param_bound 1 --param_disjoint --bias_dist uniform --check_specs --seed_mode
    python main.py --mode partial --trials $TRIALS --action_spaces $M --num_actions 10 15 20 25 30 --obs_dim $d --latent_dim $k --lbda $lambda --num_visibles $half --horizon $T --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound 1 --obs_feature_bound 1 --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std T-0.5 --reward_std 0.1 --param_dist gaussian --param_bound 1 --param_disjoint --bias_dist uniform --check_specs --seed_mode
    python main.py --mode partial --trials $TRIALS --action_spaces $M --num_actions 10 15 20 25 30 --obs_dim $d --latent_dim $k --lbda $lambda --num_visibles $almost --horizon $T --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound 1 --obs_feature_bound 1 --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std T-0.5 --reward_std 0.1 --param_dist gaussian --param_bound 1 --param_disjoint --bias_dist uniform --check_specs --seed_mode

    python main.py --mode partial --trials $TRIALS --action_spaces $M --num_actions 10 15 20 25 30 --obs_dim $d --latent_dim $k --lbda $lambda --num_visibles 1 --horizon $T --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound 1 --obs_feature_bound 1 --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std T-0.5 --reward_std 0.1 --param_dist uniform --param_bound 1 --param_disjoint --bias_dist uniform --check_specs --seed_mode
    python main.py --mode partial --trials $TRIALS --action_spaces $M --num_actions 10 15 20 25 30 --obs_dim $d --latent_dim $k --lbda $lambda --num_visibles $half --horizon $T --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound 1 --obs_feature_bound 1 --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std T-0.5 --reward_std 0.1 --param_dist uniform --param_bound 1 --param_disjoint --bias_dist uniform --check_specs --seed_mode
    python main.py --mode partial --trials $TRIALS --action_spaces $M --num_actions 10 15 20 25 30 --obs_dim $d --latent_dim $k --lbda $lambda --num_visibles $almost --horizon $T --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound 1 --obs_feature_bound 1 --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std T-0.5 --reward_std 0.1 --param_dist uniform --param_bound 1 --param_disjoint --bias_dist uniform --check_specs --seed_mode
done
