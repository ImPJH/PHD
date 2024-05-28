#!/bin/bash

K=10
k=8
d_values=(1 $(($k/2)) $(($k-1)))  # Define the d_values array with 1, (k/2), and (k-1)
feat_bound=1
sigma=0.1
param_bound=1
T=1500
delta=0.001
trials=5
p=0.2
SEED=654654

# Iterate over each value in d_values and run the command
for d in "${d_values[@]}"
do
    python main_v2.py --trials $trials --horizon $T --arms $K --latent_dim $k --dim $d --seed $SEED --feat_dist gaussian --feat_feature_bound 1 --feat_bound_method clipping --feat_bound_type lsup --feat_disjoint --reward_std $sigma --param_dist uniform --param_uniform_rng -0.5 0.5 --param_bound 1 --param_disjoint --param_bound_type l1 --p $p --delta $delta --explore --init_explore double
done