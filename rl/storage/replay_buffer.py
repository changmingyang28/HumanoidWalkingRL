import torch
import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """
    Experience Replay Buffer for SAC algorithm.
    Stores single-step transitions (s, a, r, s', done) and supports random sampling.
    """
    
    def __init__(self, capacity, obs_dim, action_dim):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            obs_dim: Observation space dimension
            action_dim: Action space dimension
        """
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Use circular buffer for memory efficiency
        self.states = torch.zeros(capacity, obs_dim, dtype=torch.float32)
        self.actions = torch.zeros(capacity, action_dim, dtype=torch.float32)
        self.rewards = torch.zeros(capacity, 1, dtype=torch.float32)
        self.next_states = torch.zeros(capacity, obs_dim, dtype=torch.float32)
        self.dones = torch.zeros(capacity, 1, dtype=torch.float32)
        
        self.ptr = 0  # Current position in buffer
        self.size = 0  # Current size of buffer
        
    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        self.states[self.ptr] = torch.FloatTensor(state.copy() if hasattr(state, 'copy') else state)
        self.actions[self.ptr] = torch.FloatTensor(action.copy() if hasattr(action, 'copy') else action)
        self.rewards[self.ptr] = torch.FloatTensor([reward])
        self.next_states[self.ptr] = torch.FloatTensor(next_state.copy() if hasattr(next_state, 'copy') else next_state)
        self.dones[self.ptr] = torch.FloatTensor([done])
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            dict: Batch of transitions
        """
        if self.size < batch_size:
            raise ValueError(f"Not enough samples in buffer: {self.size} < {batch_size}")
        
        # Random sampling
        indices = np.random.randint(0, self.size, batch_size)
        
        batch = {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_states': self.next_states[indices],
            'dones': self.dones[indices]
        }
        
        return batch
    
    def __len__(self):
        """Return current size of the buffer."""
        return self.size
    
    def is_ready(self, batch_size):
        """Check if buffer has enough samples for training."""
        return self.size >= batch_size


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay Buffer (optional enhancement).
    Samples transitions based on their TD-error priority.
    """
    
    def __init__(self, capacity, obs_dim, action_dim, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        Initialize prioritized replay buffer.
        
        Args:
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent
            beta_increment: Increment for beta annealing
        """
        super().__init__(capacity, obs_dim, action_dim)
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        
        # Priority storage using sum tree for efficiency
        self.priorities = np.zeros(capacity, dtype=np.float32)
    
    def add(self, state, action, reward, next_state, done):
        """Add transition with maximum priority."""
        super().add(state, action, reward, next_state, done)
        
        # Assign maximum priority to new experiences
        priority_idx = (self.ptr - 1) % self.capacity
        self.priorities[priority_idx] = self.max_priority ** self.alpha
    
    def sample(self, batch_size):
        """Sample batch with prioritized sampling."""
        if self.size < batch_size:
            raise ValueError(f"Not enough samples in buffer: {self.size} < {batch_size}")
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities / priorities.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize weights
        
        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_states': self.next_states[indices],
            'dones': self.dones[indices],
            'weights': torch.FloatTensor(weights),
            'indices': indices
        }
        
        return batch
    
    def update_priorities(self, indices, priorities):
        """Update priorities for given transitions."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = (abs(priority) + 1e-6) ** self.alpha
            self.max_priority = max(self.max_priority, self.priorities[idx])


class EpisodicReplayBuffer:
    """
    Episodic Replay Buffer that stores complete episodes.
    Useful for algorithms that need episode-level sampling.
    """
    
    def __init__(self, capacity, obs_dim, action_dim):
        """Initialize episodic replay buffer."""
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.episodes = deque(maxlen=capacity)
        
    def add_episode(self, states, actions, rewards, dones):
        """
        Add a complete episode to the buffer.
        
        Args:
            states: List/array of states in the episode
            actions: List/array of actions in the episode  
            rewards: List/array of rewards in the episode
            dones: List/array of done flags in the episode
        """
        episode = {
            'states': torch.FloatTensor(states),
            'actions': torch.FloatTensor(actions),
            'rewards': torch.FloatTensor(rewards),
            'dones': torch.FloatTensor(dones)
        }
        
        self.episodes.append(episode)
    
    def sample_episodes(self, num_episodes):
        """Sample random episodes."""
        if len(self.episodes) < num_episodes:
            raise ValueError(f"Not enough episodes: {len(self.episodes)} < {num_episodes}")
        
        return random.sample(self.episodes, num_episodes)
    
    def sample_transitions(self, batch_size):
        """Sample random transitions from random episodes."""
        if len(self.episodes) == 0:
            raise ValueError("No episodes in buffer")
        
        transitions = []
        
        while len(transitions) < batch_size:
            # Sample random episode
            episode = random.choice(self.episodes)
            episode_len = len(episode['rewards'])
            
            # Sample random transition from episode
            idx = random.randint(0, episode_len - 1)
            
            transition = {
                'state': episode['states'][idx],
                'action': episode['actions'][idx],
                'reward': episode['rewards'][idx],
                'next_state': episode['states'][idx + 1] if idx < episode_len - 1 else episode['states'][idx],
                'done': episode['dones'][idx]
            }
            
            transitions.append(transition)
        
        # Convert to batch format
        batch = {
            'states': torch.stack([t['state'] for t in transitions]),
            'actions': torch.stack([t['action'] for t in transitions]),
            'rewards': torch.stack([t['reward'] for t in transitions]).unsqueeze(-1),
            'next_states': torch.stack([t['next_state'] for t in transitions]),
            'dones': torch.stack([t['done'] for t in transitions]).unsqueeze(-1)
        }
        
        return batch
    
    def __len__(self):
        """Return number of episodes in buffer."""
        return len(self.episodes)