#!/usr/bin/env python3

"""
Demo script for visualizing trained JVRC walking model in MuJoCo
"""

import os
import sys
import argparse
import pickle
import torch
from pathlib import Path
from functools import partial

import numpy as np
import mujoco
import mujoco.viewer

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
    parser = argparse.ArgumentParser(description='Demo JVRC walking model')
    parser.add_argument('--model-path', type=str, 
                       default='experiments/jvrc_walk_fast',
                       help='Path to trained model directory')
    parser.add_argument('--model-checkpoint', type=str, default='actor.pt',
                       help='Specific model checkpoint to load')
    parser.add_argument('--steps', type=int, default=10000,
                       help='Number of steps to run')
    parser.add_argument('--realtime', action='store_true',
                       help='Run in real-time')
    
    args = parser.parse_args()
    
    # Path setup
    model_dir = Path(args.model_path)
    if not model_dir.exists():
        print(f"Error: Model directory {model_dir} does not exist")
        return
    
    actor_path = model_dir / args.model_checkpoint
    if not actor_path.exists():
        print(f"Error: Actor model {actor_path} does not exist")
        print(f"Available models in {model_dir}:")
        for f in model_dir.glob("actor*.pt"):
            print(f"  {f.name}")
        return
        
    experiment_path = model_dir / "experiment.pkl"
    if not experiment_path.exists():
        print(f"Error: Experiment config {experiment_path} does not exist")
        return
    
    print(f"Loading model from: {actor_path}")
    
    # Load experiment configuration
    with open(experiment_path, 'rb') as f:
        run_args = pickle.load(f)
    
    print(f"Environment: {run_args.env}")
    
    # Load trained policy
    policy = torch.load(actor_path, weights_only=False)
    policy.eval()
    
    # Create environment
    Env = import_env(run_args.env)
    yaml_path = getattr(run_args, 'yaml', None)
    if yaml_path and not os.path.exists(yaml_path):
        yaml_path = None
        
    env = Env(path_to_yaml=yaml_path)
    print(f"Environment created successfully")
    
    # Reset environment
    obs = env.reset_model()
    print(f"Environment reset, observation shape: {obs.shape}")
    
    # Set up viewer
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        print("MuJoCo viewer launched")
        print("Controls:")
        print("  Space: Pause/Resume")
        print("  R: Reset")
        print("  Q: Quit")
        
        step_count = 0
        episode_count = 0
        total_reward = 0
        
        try:
            while step_count < args.steps and viewer.is_running():
                # Get action from policy
                with torch.no_grad():
                    if hasattr(policy, 'predict'):
                        action, _ = policy.predict(obs)
                    else:
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                        action = policy(obs_tensor).cpu().numpy().flatten()
                
                # Step environment
                obs, reward, done, info = env.step(action)
                total_reward += reward
                
                # Update viewer
                viewer.sync()
                
                if args.realtime:
                    import time
                    time.sleep(env.model.opt.timestep)
                
                step_count += 1
                
                # Handle episode end
                if done:
                    episode_count += 1
                    print(f"Episode {episode_count} finished after {step_count} steps")
                    print(f"Total reward: {total_reward:.3f}")
                    print(f"Average reward: {total_reward/step_count:.5f}")
                    
                    # Reset
                    obs = env.reset_model()
                    total_reward = 0
                    step_count = 0
                
                # Print progress
                if step_count % 100 == 0:
                    print(f"Step {step_count}, Reward: {reward:.5f}")
                    
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        
        print(f"Demo finished after {step_count} steps")

if __name__ == '__main__':
    main()