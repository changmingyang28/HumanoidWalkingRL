#!/usr/bin/env python3

"""
Custom demo script with adjustable walking parameters
Usage: python demo_custom_speed.py --speed 1.2 --height 0.85
"""

import os
import sys
import argparse
import pickle
import torch
from pathlib import Path

import numpy as np
import mujoco.viewer
from envs.jvrc import JvrcWalkEnv

def main():
    parser = argparse.ArgumentParser(description='Demo with custom walking parameters')
    parser.add_argument('--model-path', type=str, 
                       default='experiments/jvrc_walk_fast/actor.pt',
                       help='Path to trained model')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Target walking speed (m/s)')
    parser.add_argument('--height', type=float, default=0.80,
                       help='Target walking height (m)')
    parser.add_argument('--swing-duration', type=float, default=0.75,
                       help='Swing phase duration (s)')
    parser.add_argument('--stance-duration', type=float, default=0.35,
                       help='Stance phase duration (s)')
    parser.add_argument('--steps', type=int, default=5000,
                       help='Number of steps to run')
    
    args = parser.parse_args()
    
    # Load model
    if os.path.exists(args.model_path):
        model_path = args.model_path
    else:
        model_path = f"experiments/jvrc_walk_fast/{args.model_path}"
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            return
    
    print(f"Loading model: {model_path}")
    policy = torch.load(model_path, weights_only=False)
    policy.eval()
    
    # Create environment
    env = JvrcWalkEnv()
    
    # Override walking parameters
    print(f"Setting walking parameters:")
    print(f"  Speed: {args.speed} m/s")
    print(f"  Height: {args.height} m")
    print(f"  Swing duration: {args.swing_duration} s")
    print(f"  Stance duration: {args.stance_duration} s")
    
    # Modify task parameters
    env.task._goal_speed_ref = args.speed
    env.task._goal_height_ref = args.height
    env.task._swing_duration = args.swing_duration
    env.task._stance_duration = args.stance_duration
    env.task._total_duration = args.swing_duration + args.stance_duration
    
    # Reset environment with new parameters
    obs = env.reset_model()
    
    print("\nStarting demo...")
    print("Controls: Space=pause, R=reset, Q=quit")
    print(f"Running for {args.steps} steps")
    
    # Run demo
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        step_count = 0
        total_reward = 0
        episode_count = 0
        
        try:
            while step_count < args.steps and viewer.is_running():
                # Get action
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    action = policy(obs_tensor).cpu().numpy().flatten()
                
                # Step environment
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                step_count += 1
                
                # Update viewer
                viewer.sync()
                
                if step_count % 500 == 0:
                    avg_reward = total_reward / step_count
                    print(f"Step {step_count}, Avg reward: {avg_reward:.4f}, Current: {reward:.4f}")
                
                if done:
                    episode_count += 1
                    print(f"Episode {episode_count} ended at step {step_count}")
                    obs = env.reset_model()
                    
        except KeyboardInterrupt:
            print("\nDemo interrupted")
        
        print(f"Demo finished. Total steps: {step_count}, Episodes: {episode_count}")
        if step_count > 0:
            print(f"Average reward: {total_reward/step_count:.4f}")

if __name__ == '__main__':
    main()