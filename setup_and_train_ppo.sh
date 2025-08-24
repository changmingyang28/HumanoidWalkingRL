#!/bin/bash
# Enhanced setup and training script for GM platform deployment - PPO Version
# Usage: gm-run setup_and_train_ppo.sh

set -e  # Exit on any error

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Dependencies installed successfully!"

# Configure GM platform integration
echo "Configuring GradMotion platform integration..."
export WANDB_PROJECT="${WANDB_PROJECT:-humanoid-walking-rl}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"  # Set your GM username if needed
export WANDB_MODE="${WANDB_MODE:-online}"

# Create logs directory
mkdir -p logs

echo "Starting PPO training with GM platform integration..."
echo "WandB Project: $WANDB_PROJECT"
echo "Algorithm: PPO (saves every 1000 iterations)"
echo "Expected checkpoints: ppo_checkpoint_1000.pt, ppo_checkpoint_2000.pt, etc."

python HumanoidWalkingRL/run_experiment.py train \
  --env h1 \
  --yaml envs/h1/configs/walking.yaml \
  --algorithm ppo \
  --logdir ./ppo_h1_walk_straight_fast \
  --n-itr 50000 \
  --use-wandb

echo "PPO Training completed!"
echo "Check the GM platform for uploaded models and training metrics."
echo "Local logs saved in: ./ppo_h1_walk_straight_fast/models/"