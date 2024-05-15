#!/bin/bash
#!/Users/sungwoo/miniconda3/bin/python

seeds=$(python get_primes.py --nums 101 102)

K=20 # action space size
d=10 # dimension of "mapped" features
feat_bound=1
sigma=0.1
param_bound=1
T=1000
delta=0.001
trials=10
epsilon=0.01
p=0.5

IFS=','
# for seed in "${seeds[@]}"; do
# for seed in {350..399}; do
for seed in $seeds; do
    echo "Seed: $seed"
    python main.py --trials $trials --horizon $T --num_actions $K --obs_dim $d --seed "$seed" --feat_dist gaussian --feat_feature_bound 1 --feat_bound_method clipping --feat_bound_type lsup --feat_disjoint --reward_std $sigma --param_dist gaussian --param_bound 1 --param_disjoint --param_bound_type l1 --p $p --epsilon $epsilon --delta $delta
done
# python main.py --trials $trials --horizon $T --num_actions $K --obs_dim $d --feat_dist gaussian --feat_feature_bound 1 --feat_bound_method clipping --feat_bound_type lsup --feat_disjoint --reward_std $sigma --param_dist gaussian --param_bound 1 --param_disjoint --param_bound_type l1 --p $p --epsilon $epsilon --delta $delta