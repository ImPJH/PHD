#!/bin/bash

seeds=$(python3 get_primes.py --nums 200 300)
K=15
k=12
d_values=(1 $(($k/2)) $(($k-1)))  # Define the d_values array with 1, (k/2), and (k-1)
feat_bound=1
sigma=0.05
param_bound=1
T=2000
delta_values=(0.001 0.0001)
trials=5
p_values=(0.1 0.2 0.3 0.4)

# Iterate over each value in seeds, p_values, d_values, and delta_values and run the command
IFS=','
for seed in $seeds;
do
    for p in "${p_values[@]}"
    do
        for d in "${d_values[@]}"
        do
            for delta in "${delta_values[@]}"
            do
                python main_v3.py --trials $trials --horizon $T --arms $K --latent_dim $k --dim $d --seed "$seed" --feat_dist gaussian --feat_feature_bound 1 --feat_bound_method clipping --feat_bound_type lsup --feat_disjoint --reward_std $sigma --param_dist uniform --param_uniform_rng -0.5 0.5 --param_bound 1 --param_disjoint --param_bound_type l1 --p $p --delta $delta --explore --init_explore quad
            done
        done
    done
done