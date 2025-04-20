#!/bin/bash

seeds=$(python3 get_primes.py --nums 4349 4600)
K_values=(20)
k=30
d=10  # Define the d_values array with 1, (k/2), and (k-1)
feat_bound=1
sigma=0.03
param_bound=1
T=1200
delta_values=(0.0001)
trials=5
# p_values=(0.6)
p=0.6
case=1
explores=("double" "triple" "quad")
today=$(date +"%Y-%m-%d")

# Iterate over each value in seeds, p_values, d_values, and delta_values and run the command
IFS=','
for seed in $seeds;
do
    for explore in "${explores[@]}"
    do
        for K in "${K_values[@]}"
        do
            for delta in "${delta_values[@]}"
            do
                python main.py --trials $trials --horizon $T --arms $K \
                --latent_dim $k --dim $d --seed "$seed" --feat_dist gaussian \
                --feat_feature_bound 1 --feat_bound_method clipping \
                --feat_bound_type lsup --feat_disjoint --reward_std $sigma \
                --param_dist uniform --param_uniform_rng -0.5 0.5 \
                --param_bound 1 --param_disjoint --param_bound_type l1 \
                --p $p --delta $delta --explore --init_explore $explore \
                --case $case --date $today
            done
        done
    done
done