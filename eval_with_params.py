#!/usr/bin/env python3

"""
Enhanced evaluation script with parameter control
Based on run_experiment.py but with walking parameter control
"""

import os
import sys
import argparse
import pickle
import torch
from pathlib import Path
from functools import partial

from rl.utils.eval import EvaluateEnv

def import_env(env_name_str):
    if env_name_str == 'jvrc_walk':
        from envs.jvrc import JvrcWalkEnv as Env
    elif env_name_str == 'jvrc_step':
        from envs.jvrc import JvrcStepEnv as Env
    elif env_name_str == 'h1':
        from envs.h1 import H1Env as Env
    else:
        raise Exception(f"Unknown environment: {env_name_str}")
    return Env

def main():
    parser = argparse.ArgumentParser(description='Evaluate model with custom parameters')
    parser.add_argument("--path", required=False, type=Path, 
                       default=Path("experiments/jvrc_walk_fast"),
                       help="Path to trained model dir")
    parser.add_argument("--out-dir", required=False, type=Path, default=None,
                       help="Path to directory to save videos")
    parser.add_argument("--ep-len", required=False, type=int, default=10,
                       help="Episode length to play (in seconds)")
    
    # Walking parameter controls
    parser.add_argument("--speed", type=float, default=None,
                       help="Override walking speed (m/s)")
    parser.add_argument("--height", type=float, default=None,
                       help="Override walking height (m)")
    parser.add_argument("--swing-duration", type=float, default=None,
                       help="Override swing duration (s)")
    parser.add_argument("--stance-duration", type=float, default=None,
                       help="Override stance duration (s)")
    
    args = parser.parse_args()
    
    # Find model files
    path_to_actor = ""
    if args.path.is_file() and args.path.suffix == ".pt":
        path_to_actor = args.path
    elif args.path.is_dir():
        path_to_actor = Path(args.path, "actor.pt")
    else:
        raise Exception("Invalid path to actor module: ", args.path)

    path_to_pkl = Path(path_to_actor.parent, "experiment.pkl")

    # Load experiment args
    run_args = pickle.load(open(path_to_pkl, "rb"))
    
    # Load trained policy
    policy = torch.load(path_to_actor, weights_only=False)
    policy.eval()
    
    # Load critic (different for PPO vs SAC)
    critic = None
    path_to_critic = Path(path_to_actor.parent, "critic" + str(path_to_actor).split('actor')[1])
    path_to_critic1 = Path(path_to_actor.parent, "critic1" + str(path_to_actor).split('actor')[1])
    
    if path_to_critic.exists():
        # PPO model
        critic = torch.load(path_to_critic, weights_only=False)
        critic.eval()
    elif path_to_critic1.exists():
        # SAC model - only load critic1 for evaluation (we don't need it anyway)
        critic = torch.load(path_to_critic1, weights_only=False)
        critic.eval()
    else:
        print("Warning: No critic found, evaluation will use actor only")

    # Import and create environment
    Env = import_env(run_args.env)
    if hasattr(run_args, "yaml") and run_args.yaml is not None:
        yaml_path = Path(run_args.yaml)
    else:
        yaml_path = None
    env = partial(Env, yaml_path)()
    
    # Override walking parameters if specified
    if args.speed is not None:
        print(f"Overriding walking speed to {args.speed} m/s")
        env.task._goal_speed_ref = args.speed
        
    if args.height is not None:
        print(f"Overriding walking height to {args.height} m")
        env.task._goal_height_ref = args.height
        
    if args.swing_duration is not None:
        print(f"Overriding swing duration to {args.swing_duration} s")
        env.task._swing_duration = args.swing_duration
        
    if args.stance_duration is not None:
        print(f"Overriding stance duration to {args.stance_duration} s")
        env.task._stance_duration = args.stance_duration
        
    if args.swing_duration is not None or args.stance_duration is not None:
        env.task._total_duration = env.task._swing_duration + env.task._stance_duration
        print(f"Total step duration: {env.task._total_duration} s")

    print(f"Current walking parameters:")
    print(f"  Speed: {getattr(env.task, '_goal_speed_ref', 'default')} m/s")
    print(f"  Height: {getattr(env.task, '_goal_height_ref', 'default')} m")
    print(f"  Swing duration: {getattr(env.task, '_swing_duration', 'default')} s")
    print(f"  Stance duration: {getattr(env.task, '_stance_duration', 'default')} s")

    # Store override parameters for later use
    override_params = {
        'speed': args.speed,
        'height': args.height,
        'swing_duration': args.swing_duration,
        'stance_duration': args.stance_duration
    }
    
    # Run evaluation
    e = EvaluateEnv(env, policy, args)
    e.override_params = override_params
    e.run()

if __name__ == '__main__':
    main()