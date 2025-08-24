"""Soft Actor-Critic (SAC) Algorithm Implementation."""
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
import time
import datetime
import sys
from copy import deepcopy

from rl.storage.replay_buffer import ReplayBuffer
from rl.policies.sac_actor_critic import SAC_LSTM_Actor, SAC_LSTM_Critic, SAC_FF_Actor, SAC_FF_Critic
from rl.envs.normalize import get_normalization_params


class SAC:
    def __init__(self, env_fn, args):
        # SAC hyperparameters
        self.gamma = args.gamma
        self.lr = args.lr
        self.tau = getattr(args, 'tau', 0.005)  # soft update rate
        self.alpha = getattr(args, 'alpha', 0.2)  # entropy regularization
        self.auto_alpha = getattr(args, 'auto_alpha', True)  # automatic alpha tuning
        self.target_entropy = getattr(args, 'target_entropy', None)
        
        self.batch_size = getattr(args, 'batch_size', 256)
        self.buffer_size = getattr(args, 'buffer_size', 1000000)
        self.learning_starts = getattr(args, 'learning_starts', 5000)
        self.update_freq = getattr(args, 'update_freq', 1)
        self.gradient_steps = getattr(args, 'gradient_steps', 1)
        
        self.max_traj_len = args.max_traj_len
        self.eval_freq = args.eval_freq
        
        self.total_steps = 0
        self.highest_reward = -np.inf
        
        # Logging
        self.save_path = Path(args.logdir)
        Path.mkdir(self.save_path, parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.save_path, flush_secs=10)
        
        # Get environment dimensions
        self.env = env_fn()
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        # Set target entropy
        if self.target_entropy is None:
            self.target_entropy = -action_dim
            
        # Create networks - allow choosing between LSTM and FF
        self.use_lstm = getattr(args, 'use_lstm', False) # Default to FF for simplicity
        if self.use_lstm:
            self.actor = SAC_LSTM_Actor(obs_dim, action_dim)
            self.critic1 = SAC_LSTM_Critic(obs_dim, action_dim)
            self.critic2 = SAC_LSTM_Critic(obs_dim, action_dim)
        else:
            # CRITICAL FIX: bounded=True for standard SAC
            self.actor = SAC_FF_Actor(obs_dim, action_dim, bounded=True)
            self.critic1 = SAC_FF_Critic(obs_dim, action_dim)
            self.critic2 = SAC_FF_Critic(obs_dim, action_dim)
        
        # Target critics
        self.target_critic1 = deepcopy(self.critic1)
        self.target_critic2 = deepcopy(self.critic2)
        
        # Freeze target networks
        for param in self.target_critic1.parameters():
            param.requires_grad = False
        for param in self.target_critic2.parameters():
            param.requires_grad = False
            
        # Automatic alpha tuning
        if self.auto_alpha:
            self.log_alpha = torch.log(torch.tensor(self.alpha)).requires_grad_(True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)
        else:
            self.log_alpha = torch.log(torch.tensor(self.alpha))
            
        # Set up observation normalization
        if hasattr(self.env, 'obs_mean') and hasattr(self.env, 'obs_std'):
            obs_mean, obs_std = self.env.obs_mean, self.env.obs_std
        else:
            # NOTE: PPO-style normalization might not be ideal for off-policy.
            # A running mean/std is often preferred, but we keep this for consistency.
            obs_mean, obs_std = get_normalization_params(
                iter=getattr(args, 'input_norm_steps', 1000),
                noise_std=1,
                policy=self.actor,
                env_fn=env_fn,
                procs=args.num_procs
            )
        
        with torch.no_grad():
            obs_mean, obs_std = map(torch.Tensor, (obs_mean, obs_std))
            for net in [self.actor, self.critic1, self.critic2, 
                       self.target_critic1, self.target_critic2]:
                net.obs_mean = obs_mean
                net.obs_std = obs_std
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size, obs_dim, action_dim)
        
    def soft_update(self, target, source, tau):
        """Soft update of target network parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def update_networks(self, gradient_steps):
        """Update actor and critic networks."""
        if len(self.replay_buffer) < self.batch_size:
            return {}
            
        losses = {
            'critic1_loss': [], 'critic2_loss': [], 'actor_loss': [],
            'alpha_loss': [], 'alpha': [], 'entropy': []
        }
        
        for _ in range(gradient_steps):
            # Sample from replay buffer
            batch = self.replay_buffer.sample(self.batch_size)
            states, actions, rewards, next_states, dones = (
                batch['states'], batch['actions'], batch['rewards'],
                batch['next_states'], batch['dones']
            )
            
            # Update critics
            with torch.no_grad():
                next_actions, next_log_probs = self.actor.sample_with_log_prob(next_states)
                target_q1 = self.target_critic1(next_states, next_actions)
                target_q2 = self.target_critic2(next_states, next_actions)
                target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
                target_q = rewards + self.gamma * (1 - dones) * target_q
            
            current_q1 = self.critic1(states, actions)
            current_q2 = self.critic2(states, actions)
            
            critic1_loss = F.mse_loss(current_q1, target_q)
            critic2_loss = F.mse_loss(current_q2, target_q)
            
            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 0.5)
            self.critic1_optimizer.step()
            
            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 0.5)
            self.critic2_optimizer.step()
            
            # Update actor
            sampled_actions, log_probs = self.actor.sample_with_log_prob(states)
            q1_pi = self.critic1(states, sampled_actions)
            q2_pi = self.critic2(states, sampled_actions)
            q_pi = torch.min(q1_pi, q2_pi)
            
            actor_loss = (self.alpha * log_probs - q_pi).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            # Update alpha
            if self.auto_alpha:
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp()
                losses['alpha_loss'].append(alpha_loss.item())
            
            losses['entropy'].append(-log_probs.mean().item())
            losses['critic1_loss'].append(critic1_loss.item())
            losses['critic2_loss'].append(critic2_loss.item())
            losses['actor_loss'].append(actor_loss.item())
            losses['alpha'].append(self.alpha.item() if isinstance(self.alpha, torch.Tensor) else self.alpha)
            
            # Soft update target networks
            self.soft_update(self.target_critic1, self.critic1, self.tau)
            self.soft_update(self.target_critic2, self.critic2, self.tau)
        
        return {k: np.mean(v) for k, v in losses.items()}
    
    @staticmethod
    def save(nets, save_path, suffix=""):
        """Save networks."""
        filetype = ".pt"
        for name, net in nets.items():
            path = Path(save_path, name + suffix + filetype)
            torch.save(net, path)
            print(f"Saved {name} at {path}")
    
    def evaluate(self, eval_env, num_episodes=5):
        """Evaluate current policy."""
        self.actor.eval()
        
        eval_rewards = []
        eval_lengths = []
        
        for _ in range(num_episodes):
            state = eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            if hasattr(self.actor, 'init_hidden_state'):
                self.actor.init_hidden_state()
            
            while not done and episode_length < self.max_traj_len:
                with torch.no_grad():
                    action = self.actor(torch.tensor(state, dtype=torch.float), deterministic=True)
                state, reward, done, _ = eval_env.step(action.detach().numpy())
                episode_reward += reward
                episode_length += 1
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
        
        avg_reward = np.mean(eval_rewards)
        avg_length = np.mean(eval_lengths)
        
        self.actor.train()
        return avg_reward, avg_length
    
    def train(self, num_total_steps):
        """Main off-policy training loop."""
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.lr)
        
        train_start_time = time.time()
        
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_count = 0
        
        for step in range(num_total_steps):
            self.total_steps += 1
            
            if self.total_steps < self.learning_starts:
                # The environment does not have a standard gym action space, so we sample manually.
                action_dim = self.env.action_space.shape[0]
                action = np.random.uniform(low=-1.0, high=1.0, size=action_dim)
            else:
                with torch.no_grad():
                    action = self.actor(torch.tensor(state, dtype=torch.float)).numpy()

            next_state, reward, done, _ = self.env.step(action)
            
            self.replay_buffer.add(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            # Trigger network updates
            if self.total_steps >= self.learning_starts and self.total_steps % self.update_freq == 0:
                losses = self.update_networks(self.gradient_steps)
                
                # Log training losses
                if self.total_steps % 1000 == 0: # Log every 1000 steps
                    for key, value in losses.items():
                        self.writer.add_scalar(f"Loss/{key}", value, self.total_steps)

            # Handle episode termination
            if done or episode_length >= self.max_traj_len:
                episode_count += 1
                print(f"Step: {self.total_steps}, Episode: {episode_count}, Length: {episode_length}, Reward: {episode_reward:.2f}")
                
                self.writer.add_scalar("Train/episode_reward", episode_reward, self.total_steps)
                self.writer.add_scalar("Train/episode_length", episode_length, self.total_steps)
                
                state = self.env.reset()
                episode_reward = 0
                episode_length = 0
                if hasattr(self.actor, 'init_hidden_state'):
                    self.actor.init_hidden_state()

            # Evaluation
            if self.total_steps > 0 and self.total_steps % self.eval_freq == 0:
                eval_start_time = time.time()
                avg_reward, avg_length = self.evaluate(self.env)
                eval_time = time.time() - eval_start_time
                
                print("-" * 37)
                print(f"Evaluation at step {self.total_steps}:")
                print(f"Avg Reward: {avg_reward:.3f}, Avg Length: {avg_length:.3f}, Time: {eval_time:.2f}s")
                print("-" * 37)
                
                self.writer.add_scalar("Eval/mean_reward", avg_reward, self.total_steps)
                self.writer.add_scalar("Eval/mean_episode_length", avg_length, self.total_steps)
                
                nets = {"actor": self.actor, "critic1": self.critic1, "critic2": self.critic2}
                if avg_reward > self.highest_reward:
                    self.highest_reward = avg_reward
                    print("New best model found! Saving...")
                    self.save(nets, self.save_path)
                
                # MODIFICATION: Save a periodic checkpoint only every 100,000 steps
                if self.total_steps % 100000 == 0:
                    print(f"Saving periodic checkpoint at step {self.total_steps}...")
                    self.save(nets, self.save_path, f"_{self.total_steps}")

        print("Training finished.")
        self.env.close()
