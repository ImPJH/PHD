#!/bin/bash

K_values=(5 10 20 30)
d=5 # dimension of observable features
feat_bound=1
sigma=0.1
param_bound=1
T=1500
delta=0.001
trials=5
p=0.2
SEED=911

# Iterate over each K value and run the command
for K in "${K_values[@]}"
do
    python main.py --trials $trials --horizon $T --arms $K --dim $d --seed $SEED --feat_dist gaussian --feat_feature_bound 1 --feat_bound_method clipping --feat_bound_type lsup --feat_disjoint --reward_std $sigma --param_dist gaussian --param_bound 1 --param_disjoint --param_bound_type l1 --p $p --delta $delta --explore --init_explore K
done