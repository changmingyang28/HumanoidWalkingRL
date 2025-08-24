#!/usr/bin/env python3

"""
Quick demo script for JVRC walking model
Usage: python quick_demo.py [checkpoint_number]
Example: python quick_demo.py 4999  # loads actor_4999.pt
"""

import sys
import os
sys.path.insert(0, os.getcwd())

import torch
import pickle
from pathlib import Path
from envs.jvrc import JvrcWalkEnv
import mujoco.viewer

def main():
    # Default to latest checkpoint
    checkpoint = "actor.pt"  
    
    if len(sys.argv) > 1:
        checkpoint_num = sys.argv[1]
        checkpoint = f"actor_{checkpoint_num}.pt"
    
    model_path = Path("experiments/jvrc_walk_fast") / checkpoint
    
    if not model_path.exists():
        print(f"Model {model_path} not found!")
        print("Available models:")
        for f in Path("experiments/jvrc_walk_fast").glob("actor*.pt"):
            print(f"  {f.name}")
        return
    
    print(f"Loading: {model_path}")
    
    # Load model and config
    policy = torch.load(model_path, weights_only=False)
    policy.eval()
    
    # Create environment
    env = JvrcWalkEnv()
    obs = env.reset_model()
    
    print("Starting MuJoCo viewer...")
    print("Press Space to pause, R to reset, Q to quit")
    
    # Run demo
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        step = 0
        total_reward = 0
        
        while viewer.is_running() and step < 10000:
            # Get action
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action = policy(obs_tensor).cpu().numpy().flatten()
            
            # Step
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            
            viewer.sync()
            step += 1
            
            if step % 500 == 0:
                print(f"Step {step}, Avg reward: {total_reward/step:.4f}")
            
            if done:
                print(f"Episode done at step {step}, resetting...")
                obs = env.reset_model()
                total_reward = 0
                step = 0
    
    print("Demo finished!")

if __name__ == "__main__":
    main()