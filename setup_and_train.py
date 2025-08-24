#!/usr/bin/env python3
"""
Setup script that installs dependencies and runs training
Usage: gm-run setup_and_train.py
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, check=True):
    """Run a command and print output"""
    print(f"Running: {cmd}")
    if isinstance(cmd, str):
        cmd = cmd.split()
    result = subprocess.run(cmd, check=check, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result

def main():
    # Change to the project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    print("Installing dependencies...")
    try:
        # Install requirements
        run_command("pip install -r requirements.txt")
        print("Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        sys.exit(1)
    
    print("Starting training...")
    try:
        # Run the training command
        train_cmd = [
            sys.executable, "HumanoidWalkingRL/run_experiment.py", 
            "train", 
            "--env", "h1", 
            "--yaml", "envs/h1/configs/walking.yaml", 
            "--algorithm", "sac", 
            "--logdir", "./sac_h1_walk_straight_fast", 
            "--n-itr", "1000000"
        ]
        run_command(train_cmd)
        print("Training completed!")
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()