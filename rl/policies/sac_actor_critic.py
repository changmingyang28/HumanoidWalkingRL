import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rl.policies.base import Net

# Constants for numerical stability
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6


class SAC_LSTM_Actor(Net):
    """SAC Actor using LSTM with reparameterization trick."""
    
    def __init__(self, state_dim, action_dim, layers=(128, 128), nonlinearity=F.relu):
        super(SAC_LSTM_Actor, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.nonlinearity = nonlinearity
        
        # LSTM layers for sequential processing
        self.lstm_layers = nn.ModuleList()
        self.lstm_layers.append(nn.LSTMCell(state_dim, layers[0]))
        for i in range(len(layers) - 1):
            self.lstm_layers.append(nn.LSTMCell(layers[i], layers[i+1]))
        
        # Output layers for mean and log_std
        self.mean_head = nn.Linear(layers[-1], action_dim)
        self.log_std_head = nn.Linear(layers[-1], action_dim)
        
        # Initialize observation normalization (will be set during training)
        self.obs_mean = 0.0
        self.obs_std = 1.0
        
        # Initialize hidden states
        self.init_hidden_state()
        
        self.initialize_parameters()
    
    def initialize_parameters(self):
        """Initialize network parameters."""
        for lstm_cell in self.lstm_layers:
            # Initialize LSTM weights
            for name, param in lstm_cell.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
        
        # Initialize output layers
        nn.init.uniform_(self.mean_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mean_head.bias, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.bias, -3e-3, 3e-3)
    
    def init_hidden_state(self, batch_size=1):
        """Initialize LSTM hidden states."""
        self.hidden = [torch.zeros(batch_size, lstm.hidden_size) 
                      for lstm in self.lstm_layers]
        self.cells = [torch.zeros(batch_size, lstm.hidden_size) 
                     for lstm in self.lstm_layers]
    
    def get_hidden_state(self):
        """Get current hidden state."""
        return self.hidden, self.cells
    
    def set_hidden_state(self, hidden_data):
        """Set hidden state."""
        if len(hidden_data) != 2:
            raise ValueError("Invalid hidden state data")
        self.hidden, self.cells = hidden_data
    
    def _process_lstm(self, x):
        """Process input through LSTM layers."""
        dims = len(x.size())
        
        if dims == 3:  # Batch of sequences: (seq_len, batch_size, features)
            self.init_hidden_state(batch_size=x.size(1))
            outputs = []
            
            for t in range(x.size(0)):  # Iterate over time steps
                x_t = x[t]  # (batch_size, features)
                
                # Pass through LSTM layers
                for i, lstm_cell in enumerate(self.lstm_layers):
                    h, c = self.hidden[i], self.cells[i]
                    self.hidden[i], self.cells[i] = lstm_cell(x_t, (h, c))
                    x_t = self.hidden[i]
                
                outputs.append(x_t)
            
            return torch.stack(outputs)  # (seq_len, batch_size, hidden_size)
        
        else:
            # Single step or batch of single steps
            if dims == 1:  # Single sample: (features,)
                x = x.unsqueeze(0)  # (1, features)
                single_sample = True
            else:
                single_sample = False
            
            # Pass through LSTM layers
            for i, lstm_cell in enumerate(self.lstm_layers):
                h, c = self.hidden[i], self.cells[i]
                self.hidden[i], self.cells[i] = lstm_cell(x, (h, c))
                x = self.hidden[i]
            
            if single_sample:
                x = x.squeeze(0)  # Back to (features,)
            
            return x
    
    def _get_action_dist_params(self, state):
        """Get distribution parameters (mean and log_std) for actions."""
        # Normalize observations
        state = (state - self.obs_mean) / self.obs_std
        
        # Process through LSTM
        x = self._process_lstm(state)
        
        # Get mean and log_std
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        
        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        
        return mean, log_std
    
    def forward(self, state, deterministic=False):
        """Forward pass - returns action."""
        mean, log_std = self._get_action_dist_params(state)
        
        if deterministic:
            action = torch.tanh(mean)
        else:
            std = log_std.exp()
            # Reparameterization trick
            epsilon = torch.randn_like(mean)
            pre_tanh_action = mean + epsilon * std
            action = torch.tanh(pre_tanh_action)
        
        return action
    
    def sample_with_log_prob(self, state):
        """Sample action and return log probability (needed for SAC)."""
        mean, log_std = self._get_action_dist_params(state)
        std = log_std.exp()
        
        # Create normal distribution and sample
        normal = Normal(mean, std)
        # Reparameterization trick
        pre_tanh_action = normal.rsample()
        action = torch.tanh(pre_tanh_action)
        
        # Calculate log probability with tanh correction
        log_prob = normal.log_prob(pre_tanh_action)
        # Correct for tanh squashing
        log_prob -= torch.log(1 - action.pow(2) + EPSILON)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob
    
    def log_prob(self, state, action):
        """Calculate log probability of given action."""
        mean, log_std = self._get_action_dist_params(state)
        std = log_std.exp()
        
        # Inverse tanh to get pre_tanh_action
        # Clamp action to avoid numerical issues
        action = torch.clamp(action, -1 + EPSILON, 1 - EPSILON)
        pre_tanh_action = torch.atanh(action)
        
        # Calculate log probability
        normal = Normal(mean, std)
        log_prob = normal.log_prob(pre_tanh_action)
        
        # Correct for tanh squashing
        log_prob -= torch.log(1 - action.pow(2) + EPSILON)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return log_prob


class SAC_LSTM_Critic(Net):
    """SAC Critic (Q-function) using LSTM."""
    
    def __init__(self, state_dim, action_dim, layers=(128, 128), nonlinearity=F.relu):
        super(SAC_LSTM_Critic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.nonlinearity = nonlinearity
        
        # LSTM layers - input is state + action concatenated
        input_dim = state_dim + action_dim
        self.lstm_layers = nn.ModuleList()
        self.lstm_layers.append(nn.LSTMCell(input_dim, layers[0]))
        for i in range(len(layers) - 1):
            self.lstm_layers.append(nn.LSTMCell(layers[i], layers[i+1]))
        
        # Output layer - single Q-value
        self.q_head = nn.Linear(layers[-1], 1)
        
        # Initialize observation normalization
        self.obs_mean = 0.0
        self.obs_std = 1.0
        
        # Initialize hidden states
        self.init_hidden_state()
        
        self.initialize_parameters()
    
    def initialize_parameters(self):
        """Initialize network parameters."""
        for lstm_cell in self.lstm_layers:
            # Initialize LSTM weights
            for name, param in lstm_cell.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
        
        # Initialize Q-value head
        nn.init.uniform_(self.q_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.q_head.bias, -3e-3, 3e-3)
    
    def init_hidden_state(self, batch_size=1):
        """Initialize LSTM hidden states."""
        self.hidden = [torch.zeros(batch_size, lstm.hidden_size) 
                      for lstm in self.lstm_layers]
        self.cells = [torch.zeros(batch_size, lstm.hidden_size) 
                     for lstm in self.lstm_layers]
    
    def _process_lstm(self, x):
        """Process input through LSTM layers."""
        dims = len(x.size())
        
        if dims == 3:  # Batch of sequences: (seq_len, batch_size, features)
            self.init_hidden_state(batch_size=x.size(1))
            outputs = []
            
            for t in range(x.size(0)):  # Iterate over time steps
                x_t = x[t]  # (batch_size, features)
                
                # Pass through LSTM layers
                for i, lstm_cell in enumerate(self.lstm_layers):
                    h, c = self.hidden[i], self.cells[i]
                    self.hidden[i], self.cells[i] = lstm_cell(x_t, (h, c))
                    x_t = self.nonlinearity(self.hidden[i])
                
                outputs.append(x_t)
            
            return torch.stack(outputs)  # (seq_len, batch_size, hidden_size)
        
        else:
            # Single step or batch of single steps
            if dims == 1:  # Single sample: (features,)
                x = x.unsqueeze(0)  # (1, features)
                single_sample = True
            else:
                single_sample = False
            
            # Pass through LSTM layers
            for i, lstm_cell in enumerate(self.lstm_layers):
                h, c = self.hidden[i], self.cells[i]
                self.hidden[i], self.cells[i] = lstm_cell(x, (h, c))
                x = self.nonlinearity(self.hidden[i])
            
            if single_sample:
                x = x.squeeze(0)  # Back to (features,)
            
            return x
    
    def forward(self, state, action):
        """Forward pass - returns Q-value."""
        # Normalize observations
        state = (state - self.obs_mean) / self.obs_std
        
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        
        # Process through LSTM
        x = self._process_lstm(x)
        
        # Get Q-value
        q_value = self.q_head(x)
        
        return q_value


class SAC_FF_Actor(Net):
    """SAC Actor using Feed-Forward networks - unbounded version like PPO."""
    
    def __init__(self, state_dim, action_dim, layers=(256, 256), nonlinearity=F.relu, bounded=False):
        super(SAC_FF_Actor, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.nonlinearity = nonlinearity
        self.bounded = bounded  # Add bounded option
        
        # Build feed-forward layers - EXACTLY like PPO
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_dim, layers[0]))
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        # Output layers - like PPO's Gaussian_FF_Actor
        self.mean_head = nn.Linear(layers[-1], action_dim)
        self.log_std_head = nn.Linear(layers[-1], action_dim)
        
        # Initialize observation normalization
        self.obs_mean = 0.0
        self.obs_std = 1.0
        
        self.initialize_parameters()
    
    def initialize_parameters(self):
        """Initialize network parameters - match PPO's normc initialization."""
        # Apply normc initialization like PPO to all layers  
        self.apply(self._normc_init)
        
        # DON'T scale down mean_head - PPO only scales after training starts
        # PPO's mul_(0.01) is applied after normc, but we want same initial scale
        
        # Initialize log_std_head separately for SAC (not affected by normc)
        nn.init.uniform_(self.log_std_head.weight, -3e-3, 3e-3)
        # Initialize log_std to match PPO's fixed std=0.223: log(0.223) ≈ -1.5
        nn.init.constant_(self.log_std_head.bias, -1.5)
    
    def _normc_init(self, m):
        """PPO's normc initialization function."""
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            m.weight.data.normal_(0, 1)
            m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
            if m.bias is not None:
                m.bias.data.fill_(0)
    
    def _get_action_dist_params(self, state):
        """Get distribution parameters."""
        # Normalize observations
        state = (state - self.obs_mean) / self.obs_std
        
        # Forward through layers
        x = state
        for layer in self.layers:
            x = self.nonlinearity(layer(x))
        
        # Get mean and log_std
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        
        return mean, log_std
    
    def forward(self, state, deterministic=False):
        """Forward pass - unbounded like PPO."""
        mean, log_std = self._get_action_dist_params(state)
        
        if deterministic:
            if self.bounded:
                action = torch.tanh(mean)
            else:
                action = mean  # No tanh - like PPO!
        else:
            std = log_std.exp()
            epsilon = torch.randn_like(mean)
            if self.bounded:
                pre_tanh_action = mean + epsilon * std
                action = torch.tanh(pre_tanh_action)
            else:
                action = mean + epsilon * std  # No tanh - like PPO!
        
        return action
    
    def sample_with_log_prob(self, state):
        """Sample action with log probability - unbounded version."""
        mean, log_std = self._get_action_dist_params(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        
        if self.bounded:
            # Original SAC with tanh
            pre_tanh_action = normal.rsample()
            action = torch.tanh(pre_tanh_action)
            log_prob = normal.log_prob(pre_tanh_action)
            log_prob -= torch.log(1 - action.pow(2) + EPSILON)
            log_prob = log_prob.sum(-1, keepdim=True)
        else:
            # Unbounded like PPO
            action = normal.rsample()
            log_prob = normal.log_prob(action)
            log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob
    
    def log_prob(self, state, action):
        """Calculate log probability of action - unbounded version."""
        mean, log_std = self._get_action_dist_params(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        
        if self.bounded:
            # Original SAC with tanh correction
            action = torch.clamp(action, -1 + EPSILON, 1 - EPSILON)
            pre_tanh_action = torch.atanh(action)
            log_prob = normal.log_prob(pre_tanh_action)
            log_prob -= torch.log(1 - action.pow(2) + EPSILON)
            log_prob = log_prob.sum(-1, keepdim=True)
        else:
            # Unbounded like PPO
            log_prob = normal.log_prob(action)
            log_prob = log_prob.sum(-1, keepdim=True)
        
        return log_prob


class SAC_FF_Critic(Net):
    """SAC Critic using Feed-Forward networks."""
    
    def __init__(self, state_dim, action_dim, layers=(256, 256), nonlinearity=F.relu):
        super(SAC_FF_Critic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.nonlinearity = nonlinearity
        
        # Build feed-forward layers - input is state + action
        input_dim = state_dim + action_dim
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, layers[0]))
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        # Output layer
        self.q_head = nn.Linear(layers[-1], 1)
        
        # Initialize observation normalization
        self.obs_mean = 0.0
        self.obs_std = 1.0
        
        self.initialize_parameters()
    
    def initialize_parameters(self):
        """Initialize network parameters."""
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        
        # Initialize output layer
        nn.init.uniform_(self.q_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.q_head.bias, -3e-3, 3e-3)
    
    def forward(self, state, action):
        """Forward pass - returns Q-value."""
        # Normalize observations
        state = (state - self.obs_mean) / self.obs_std
        
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        
        # Forward through layers
        for layer in self.layers:
            x = self.nonlinearity(layer(x))
        
        # Get Q-value
        q_value = self.q_head(x)
        
        return q_value