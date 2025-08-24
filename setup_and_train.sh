#!/bin/bash
# Setup and training script for platform deployment
# Usage: gm-run setup_and_train.sh

set -e  # Exit on any error

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Dependencies installed successfully!"
echo "Starting training..."

python HumanoidWalkingRL/run_experiment.py train \
  --env h1 \
  --yaml envs/h1/configs/walking.yaml \
  --algorithm sac \
  --logdir ./sac_h1_walk_straight_fast \
  --n-itr 1000000

echo "Training completed!"