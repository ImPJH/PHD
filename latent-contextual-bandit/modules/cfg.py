import argparse
import multiprocessing

def tuple_type(inputs):
    return tuple(map(float, inputs.split(',')))


def get_cfg():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed_mode", action="store_true")
    parser.add_argument("--mode", type=str, choices=['full', 'partial'])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--alphas", type=float, nargs="+")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--agent_type", type=str)
    parser.add_argument("--check_specs", action="store_true")
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument("--num_cpus", type=int, default=multiprocessing.cpu_count()//2)
        
    parser.add_argument("--action_spaces", "-A", type=int, default=10000)
    parser.add_argument("--num_actions", "-N", type=int, nargs="+")
    parser.add_argument("--obs_dim", "-d", type=int, default=10)
    parser.add_argument("--latent_dim", "-k", type=int, default=7)
    parser.add_argument("--num_visibles", "-m", type=int, default=None)
    parser.add_argument("--horizon", "-T", type=int, default=10000)
    parser.add_argument("--seed", "-S", type=int, default=None)
    parser.add_argument("--lbda", "-L", type=float, default=1.)
    parser.add_argument("--epsilon", "-E", type=float, default=0.1)
    parser.add_argument("--egreedy", action="store_true")
    
    parser.add_argument("--feat_dist", "-FD", type=str, default="gaussian")
    parser.add_argument("--feat_disjoint", action="store_true")
    parser.add_argument("--feat_cov_dist", "-FCD", type=str, default=None)
    parser.add_argument("--feat_uniform_rng", "-FUR", type=float, default=None, nargs=2)
    
    parser.add_argument("--obs_feature_bound", "-OFB", type=float, default=None)
    parser.add_argument("--latent_feature_bound", "-LFB", type=float, default=None)
    parser.add_argument("--obs_bound_method", "-OBM", type=str, choices=["scaling", "clipping"], default=None)
    parser.add_argument("--latent_bound_method", "-LBM", type=str, choices=["scaling", "clipping"], default=None)
    
    parser.add_argument("--map_dist", "-MD", type=str, default="uniform")
    parser.add_argument("--map_lower_bound", "-MLB", type=float, default=None)
    parser.add_argument("--map_upper_bound", "-MUB", type=float, default=None)
    parser.add_argument("--map_uniform_rng", "-MUR", type=float, default=None, nargs=2)
    
    parser.add_argument("--context_std", "-CS", type=str, default=None)
    parser.add_argument("--reward_std", "-RNS", type=float, default=0.1)
    
    parser.add_argument("--param_dist", "-PD", type=str, default="uniform")
    parser.add_argument("--param_bound", "-PB", type=float, default=1.)
    parser.add_argument("--param_uniform_rng", "-PUR", type=float, default=None, nargs=2)
    parser.add_argument("--param_disjoint", action="store_true")
    
    parser.add_argument("--filetype", type=str, choices=["pickle", "json"], default="pickle")
    parser.add_argument("--is_control", action="store_true")
    
    return parser.parse_args()
