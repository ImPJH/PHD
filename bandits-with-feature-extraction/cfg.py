from functools import partial
import argparse

def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=128, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of threads")
    parser.add_argument("--name", type=str, default="Model", help="The name of the model")
    parser.add_argument("--save_plot", action="store_true")
    parser.add_argument("--tqdm", action="store_true")
    parser.add_argument("--aux", action="store_false")
    parser.add_argument("--google_modified", type=bool, default=True)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--shortcut", type=bool, default=True)
    parser.add_argument("--multihead", action='store_true')
    parser.add_argument("--save_model", action='store_true')
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--decompose", action='store_true')
    parser.add_argument("--cpu", action="store_true")
    
    ## arguments for data augmentation
    augarg = partial(parser.add_argument, type=float)
    augarg("--gs_f", default=3, help="Gaussian blur filter size")
    augarg("--gs_s", default=1.0, help="Gaussian blur sigma")
    augarg("--cj_a", default=1.5, help="Color jitter alpha")
    augarg("--cj_b", default=20, help="Color jitter beta")
    augarg("--ro_a", default=15, help="Rotation angle")
    
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--optim", choices=['sgd', 'rmsprop', 'adagrad', 'adam', 'adamax', 'adadelta'], 
                        default='adam', help="Optimizers")
    parser.add_argument("--factor", type=float, default=0.1, help="Scheduler factor")
    
    ## arguments for feature map
    parser.add_argument("--aug_p", type=float, default=0.5, help="Random augmentation probability")
    parser.add_argument("--img_size", type=int, default=None, help="Image size")
    
    parser.add_argument("--drop_cnn", type=float, default=0.1, help="Dropout rate for CNN network")
    parser.add_argument("--drop_fc", type=float, default=0.5, help="Dropout rate for FC network")
    parser.add_argument("--epoch", type=int, default=200, help="Training epochs")
    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--head_size1", type=int, default=4096)
    parser.add_argument("--head_size2", type=int, default=1000)
    
    parser.add_argument(
        "--wandb",
        type=str,
        default="self_supervised",
        help="name of the project for logging at https://wandb.ai",
    )
    
    return parser.parse_args()
