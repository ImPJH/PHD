#!/bin/bash

seeds=$(python3 get_primes.py --nums 6070 6080)
# seeds=(3499 3491 3467 3463 3457 3391 3389 3373 3343 3331 3329 3323 3319 3301 3259 3203 3163 3121 3109 3079 3037 2999 2927 2917 2903 2887 2879 2861 2851 2843 2833 2797 2777 2749 2729 2713 2707 2677 2659 2647 2591 2579 2551 2549 2543 2477 2467 2441 2437 2417 2389 2381 2377 2357 2341 2281 2269 2251 2161 2137 2131 2113 2089 2087 2069 2039 2027 2011 1997 1993 1973 1933 1879 1871 1811 1787 1753 1721 1699 1669 1663 1627 1619 1613 1559 1549 1531 1499 1489 1487 1483 1453 1451 1423 1381 1327 1321 1303 1301  1283 1231 1201 1109 1093 1091 1049 1039 1021 1009 991 977 971 929 919 863 859 857 823 811 809 761 757 751 709 701 677 659 631 607 601 593 577 557 541 523 521 503 499 487 479 467 461 457 449 443 419 409 349 347 337 331 313 283 277 257 241 233 193 173 167 163 157 151 137 83 71 67 59 47 43 29 23 7 5)

M=100000 # action space size
d=12 # dimension of "mapped" features
k=8 # dimension of "latent" features
half=$((k / 2))
almost=$((k - 1))
feat_bound=1
sigma=0.1
param_bound=1
T=20000 # total horizon
K=35
delta=0.1
TRIALS=5 # number of trials
ALPHA1=$(bc -l <<< "scale=2; sqrt(0.5 * l($T * $K / $delta))")
ALPHA2=$(bc -l <<< "scale=2; 1 + sqrt(l(2/$delta)/2)")
ALPHA3_A=$(bc -l <<< "scale=2; $sigma * sqrt($d * l(1 + ($T * $feat_bound^2) / 1) / $delta)")
ALPHA3_B=$param_bound
ALPHA3=$(bc -l <<< "scale=2; 1 + $ALPHA3_B / $ALPHA3_A")

# ALPHA3 값이 올바르게 계산되었는지 출력하여 확인
echo "ALPHA1 calculated value: $ALPHA1"
echo "ALPHA2 calculated value: $ALPHA2"
echo "ALPHA3 calculated value: $ALPHA3"

echo "Partial Mode"
IFS=','
# for seed in "${seeds[@]}"; do
# for seed in {350..399}; do
for seed in $seeds; do
    # echo "Context Noise is 0"
    # python main_vary.py --mode partial --alphas 0.3 0.5 1.0 $ALPHA1 $ALPHA2 $ALPHA3 --trials $TRIALS --action_spaces $M --num_actions $K --obs_dim $d --latent_dim $k --num_visibles 1 --horizon $T --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound $feat_bound --obs_feature_bound $feat_bound --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std 0 --reward_std $sigma --param_dist uniform --param_bound $param_bound --param_disjoint --bias_dist gaussian --check_specs --seed_mode
    # python main_vary.py --mode partial --alphas 0.3 0.5 1.0 $ALPHA1 $ALPHA2 $ALPHA3 --trials $TRIALS --action_spaces $M --num_actions $K --obs_dim $d --latent_dim $k --num_visibles $half --horizon $T --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound $feat_bound --obs_feature_bound $feat_bound --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std 0 --reward_std $sigma --param_dist uniform --param_bound $param_bound --param_disjoint --bias_dist gaussian --check_specs --seed_mode
    # python main_vary.py --mode partial --alphas 0.3 0.5 1.0 $ALPHA1 $ALPHA2 $ALPHA3 --trials $TRIALS --action_spaces $M --num_actions $K --obs_dim $d --latent_dim $k --num_visibles $almost --horizon $T --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound $feat_bound --obs_feature_bound $feat_bound --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std 0 --reward_std $sigma --param_dist uniform --param_bound $param_bound --param_disjoint --bias_dist gaussian --check_specs --seed_mode

    # python main_vary.py --mode partial --alphas 0.3 0.5 1.0 $ALPHA1 $ALPHA2 $ALPHA3 --trials $TRIALS --action_spaces $M --num_actions $K --obs_dim $d --latent_dim $k --num_visibles 1 --horizon $T --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound $feat_bound --obs_feature_bound $feat_bound --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std 0 --reward_std $sigma --param_dist uniform --param_bound $param_bound --param_disjoint --bias_dist uniform --check_specs --seed_mode
    # python main_vary.py --mode partial --alphas 0.3 0.5 1.0 $ALPHA1 $ALPHA2 $ALPHA3 --trials $TRIALS --action_spaces $M --num_actions $K --obs_dim $d --latent_dim $k --num_visibles $half --horizon $T --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound $feat_bound --obs_feature_bound $feat_bound --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std 0 --reward_std $sigma --param_dist uniform --param_bound $param_bound --param_disjoint --bias_dist uniform --check_specs --seed_mode
    # python main_vary.py --mode partial --alphas 0.3 0.5 1.0 $ALPHA1 $ALPHA2 $ALPHA3 --trials $TRIALS --action_spaces $M --num_actions $K --obs_dim $d --latent_dim $k --num_visibles $almost --horizon $T --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound $feat_bound --obs_feature_bound $feat_bound --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std 0 --reward_std $sigma --param_dist uniform --param_bound $param_bound --param_disjoint --bias_dist uniform --check_specs --seed_mode
    
    echo "Context Noise is 1 over sqrt(T)"
    python main_vary.py --mode partial --alphas 0.3 0.5 1.0 $ALPHA1 $ALPHA2 $ALPHA3 --trials $TRIALS --action_spaces $M --num_actions $K --obs_dim $d --latent_dim $k --num_visibles 1 --horizon $T --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound $feat_bound --obs_feature_bound $feat_bound --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std T-0.5 --reward_std $sigma --param_dist uniform --param_bound $param_bound --param_disjoint --bias_dist gaussian --check_specs --seed_mode
    python main_vary.py --mode partial --alphas 0.3 0.5 1.0 $ALPHA1 $ALPHA2 $ALPHA3 --trials $TRIALS --action_spaces $M --num_actions $K --obs_dim $d --latent_dim $k --num_visibles $half --horizon $T --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound $feat_bound --obs_feature_bound $feat_bound --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std T-0.5 --reward_std $sigma --param_dist uniform --param_bound $param_bound --param_disjoint --bias_dist gaussian --check_specs --seed_mode
    python main_vary.py --mode partial --alphas 0.3 0.5 1.0 $ALPHA1 $ALPHA2 $ALPHA3 --trials $TRIALS --action_spaces $M --num_actions $K --obs_dim $d --latent_dim $k --num_visibles $almost --horizon $T --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound $feat_bound --obs_feature_bound $feat_bound --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std T-0.5 --reward_std $sigma --param_dist uniform --param_bound $param_bound --param_disjoint --bias_dist gaussian --check_specs --seed_mode

    python main_vary.py --mode partial --alphas 0.3 0.5 1.0 $ALPHA1 $ALPHA2 $ALPHA3 --trials $TRIALS --action_spaces $M --num_actions $K --obs_dim $d --latent_dim $k --num_visibles 1 --horizon $T --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound $feat_bound --obs_feature_bound $feat_bound --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std T-0.5 --reward_std $sigma --param_dist uniform --param_bound $param_bound --param_disjoint --bias_dist uniform --check_specs --seed_mode
    python main_vary.py --mode partial --alphas 0.3 0.5 1.0 $ALPHA1 $ALPHA2 $ALPHA3 --trials $TRIALS --action_spaces $M --num_actions $K --obs_dim $d --latent_dim $k --num_visibles $half --horizon $T --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound $feat_bound --obs_feature_bound $feat_bound --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std T-0.5 --reward_std $sigma --param_dist uniform --param_bound $param_bound --param_disjoint --bias_dist uniform --check_specs --seed_mode
    python main_vary.py --mode partial --alphas 0.3 0.5 1.0 $ALPHA1 $ALPHA2 $ALPHA3 --trials $TRIALS --action_spaces $M --num_actions $K --obs_dim $d --latent_dim $k --num_visibles $almost --horizon $T --seed "$seed" --feat_dist gaussian --feat_disjoint --latent_feature_bound $feat_bound --obs_feature_bound $feat_bound --latent_bound_method scaling --map_dist uniform --map_upper_bound 1 --context_std T-0.5 --reward_std $sigma --param_dist uniform --param_bound $param_bound --param_disjoint --bias_dist uniform --check_specs --seed_mode
done