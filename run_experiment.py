from pathlib import Path
import sys
import argparse
import subprocess
import os
import ray
from functools import partial

import numpy as np
import torch
import pickle
import shutil

# Auto-install dependencies if needed
def install_dependencies():
    """Install required dependencies automatically."""
    required_packages = [
        'ray[default]', 'torch', 'numpy', 'mujoco', 'gymnasium', 
        'stable-baselines3', 'matplotlib', 'opencv-python', 'imageio',
        'tensorboard', 'wandb', 'neptune-client'
    ]
    
    print("Checking and installing dependencies...")
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'ray[default]':
                import ray
            elif package == 'opencv-python':
                import cv2
            elif package == 'neptune-client':
                import neptune
            else:
                __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Installing missing packages: {missing_packages}")
        for package in missing_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("Dependencies installed successfully!")
    else:
        print("All dependencies already installed.")

# Install dependencies at import time
install_dependencies()

# Set up GM platform environment variables
os.environ.setdefault('WANDB_PROJECT', 'humanoid-walking-rl')
os.environ.setdefault('WANDB_MODE', 'online')

print("GM Platform Integration: ENABLED")
print(f"WandB Project: {os.environ['WANDB_PROJECT']}")
if os.environ.get('WANDB_ENTITY'):
    print(f"WandB Entity: {os.environ['WANDB_ENTITY']}")

from rl.algos.ppo import PPO
from rl.algos.sac import SAC
from rl.envs.wrappers import SymmetricEnv
from rl.utils.eval import EvaluateEnv

def import_env(env_name_str):
    if env_name_str=='jvrc_walk':
        from envs.jvrc import JvrcWalkEnv as Env
    elif env_name_str=='jvrc_step':
        from envs.jvrc import JvrcStepEnv as Env
    elif env_name_str=='h1':
        from envs.h1 import H1Env as Env
    else:
        raise Exception("Check env name!")
    return Env

def run_experiment(args):
    # import the correct environment
    Env = import_env(args.env)

    # wrapper function for creating parallelized envs
    env_fn = partial(Env, path_to_yaml=args.yaml)
    _env = env_fn()
    if not args.no_mirror:
        try:
            print("Wrapping in SymmetricEnv.")
            env_fn = partial(SymmetricEnv, env_fn,
                             mirrored_obs=_env.robot.mirrored_obs,
                             mirrored_act=_env.robot.mirrored_acts,
                             clock_inds=_env.robot.clock_inds)
        except AttributeError as e:
            print("Warning! Cannot use SymmetricEnv.", e)

    # Set up Parallelism
    #os.environ['OMP_NUM_THREADS'] = '1'  # [TODO: Is this needed?]
    if not ray.is_initialized():
        ray.init(num_cpus=args.num_procs)

    # dump hyperparameters
    Path.mkdir(args.logdir, parents=True, exist_ok=True)
    pkl_path = Path(args.logdir, "experiment.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(args, f)

    # copy config file
    if args.yaml:
        config_out_path = Path(args.logdir, "config.yaml")
        shutil.copyfile(args.yaml, config_out_path)

    # Choose algorithm
    algorithm = getattr(args, 'algorithm', 'ppo').lower()
    if algorithm == 'ppo':
        algo = PPO(env_fn, args)
    elif algorithm == 'sac':
        algo = SAC(env_fn, args)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    if algorithm == 'sac':
        # For SAC, n_itr will be interpreted as the total number of training steps
        algo.train(num_total_steps=args.n_itr)
    else:
        # For PPO, the original signature is kept
        algo.train(env_fn, args.n_itr)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    if sys.argv[1] == 'train':
        sys.argv.remove(sys.argv[1])

        parser.add_argument("--env", required=True, type=str)
        parser.add_argument("--logdir", default=Path("/tmp/logs"), type=Path, help="Path to save weights and logs")
        parser.add_argument("--input-norm-steps", type=int, default=100000)
        parser.add_argument("--n-itr", type=int, default=20000, help="Number of iterations of the learning algorithm")
        parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate") # Xie
        parser.add_argument("--eps", type=float, default=1e-5, help="Adam epsilon (for numerical stability)")
        parser.add_argument("--lam", type=float, default=0.95, help="Generalized advantage estimate discount")
        parser.add_argument("--gamma", type=float, default=0.99, help="MDP discount")
        parser.add_argument("--std-dev", type=float, default=0.223, help="Action noise for exploration")
        parser.add_argument("--learn-std", action="store_true", help="Exploration noise will be learned")
        parser.add_argument("--entropy-coeff", type=float, default=0.0, help="Coefficient for entropy regularization")
        parser.add_argument("--clip", type=float, default=0.2, help="Clipping parameter for PPO surrogate loss")
        parser.add_argument("--minibatch-size", type=int, default=64, help="Batch size for PPO updates")
        parser.add_argument("--epochs", type=int, default=3, help="Number of optimization epochs per PPO update") #Xie
        parser.add_argument("--use-gae", type=bool, default=True,help="Whether or not to calculate returns using Generalized Advantage Estimation")
        parser.add_argument("--num-procs", type=int, default=12, help="Number of threads to train on")
        parser.add_argument("--max-grad-norm", type=float, default=0.05, help="Value to clip gradients at")
        parser.add_argument("--max-traj-len", type=int, default=400, help="Max episode horizon")
        parser.add_argument("--no-mirror", required=False, action="store_true", help="to use SymmetricEnv")
        parser.add_argument("--mirror-coeff", required=False, default=0.4, type=float, help="weight for mirror loss")
        parser.add_argument("--eval-freq", required=False, default=100, type=int, help="Frequency of performing evaluation")
        parser.add_argument("--continued", required=False, type=Path, help="path to pretrained weights")
        parser.add_argument("--recurrent", required=False, action="store_true", help="use LSTM instead of FF")
        parser.add_argument("--imitate", required=False, type=str, default=None, help="Policy to imitate")
        parser.add_argument("--imitate-coeff", required=False, type=float, default=0.3, help="Coefficient for imitation loss")
        parser.add_argument("--yaml", required=False, type=str, default=None, help="Path to config file passed to Env class")
        
        # Algorithm selection
        parser.add_argument("--algorithm", required=False, type=str, default="ppo", choices=["ppo", "sac"], 
                           help="RL algorithm to use")
        
        # SAC specific parameters
        parser.add_argument("--tau", required=False, type=float, default=0.005, help="SAC: Soft update rate")
        parser.add_argument("--alpha", required=False, type=float, default=0.2, help="SAC: Entropy regularization coefficient")
        parser.add_argument("--auto-alpha", required=False, action="store_true", help="SAC: Automatic alpha tuning")
        parser.add_argument("--target-entropy", required=False, type=float, default=None, help="SAC: Target entropy")
        parser.add_argument("--batch-size", required=False, type=int, default=256, help="SAC: Batch size for updates")
        parser.add_argument("--buffer-size", required=False, type=int, default=1000000, help="SAC: Replay buffer size")
        parser.add_argument("--learning-starts", required=False, type=int, default=1000, help="SAC: Steps before learning starts")
        parser.add_argument("--update-freq", required=False, type=int, default=1, help="SAC: Update frequency")
        parser.add_argument("--gradient-steps", required=False, type=int, default=1, help="SAC: Gradient steps per update")
        parser.add_argument("--use-lstm", required=False, action="store_true", help="SAC: Use LSTM networks")
        parser.add_argument("--use-wandb", required=False, action="store_true", default=True, help="Enable WandB/GM platform logging")
        
        args = parser.parse_args()

        run_experiment(args)

    elif sys.argv[1] == 'eval':
        sys.argv.remove(sys.argv[1])

        parser.add_argument("--path", required=False, type=Path, default=Path("/tmp/logs"),
                            help="Path to trained model dir")
        parser.add_argument("--out-dir", required=False, type=Path, default=None,
                            help="Path to directory to save videos")
        parser.add_argument("--ep-len", required=False, type=int, default=10,
                            help="Episode length to play (in seconds)")
        args = parser.parse_args()

        path_to_actor = ""
        if args.path.is_file() and args.path.suffix==".pt":
            path_to_actor = args.path
        elif args.path.is_dir():
            path_to_actor = Path(args.path, "actor.pt")
        else:
            raise Exception("Invalid path to actor module: ", args.path)

        path_to_pkl = Path(path_to_actor.parent, "experiment.pkl")

        # load experiment args
        run_args = pickle.load(open(path_to_pkl, "rb"))
        # load trained policy
        policy = torch.load(path_to_actor, weights_only=False)
        policy.eval()

        # Only load critic for PPO, as it's not needed for SAC eval and filenames differ.
        if getattr(run_args, 'algorithm', 'ppo') == 'ppo':
            path_to_critic = Path(path_to_actor.parent, "critic" + str(path_to_actor).split('actor')[1])
            critic = torch.load(path_to_critic, weights_only=False)
            critic.eval()

        # load experiment args
        run_args = pickle.load(open(path_to_pkl, "rb"))

        # import the correct environment
        Env = import_env(run_args.env)
        if "yaml" in run_args and run_args.yaml is not None:
            yaml_path = Path(run_args.yaml)
        else:
            yaml_path = None
        env = partial(Env, yaml_path)()

        # run
        e = EvaluateEnv(env, policy, args)
        e.run()
