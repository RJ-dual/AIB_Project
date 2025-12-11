"""
Conservative Q-Learning (CQL) for CartPole-v1
--------------------------------------------
Offline RL algorithm with conservative regularization to prevent Q-value overestimation.
Compatible with train.py interface.
"""

from __future__ import annotations
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple, Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# -----------------------------
# Default Hyperparameters
# -----------------------------
@dataclass
class CQLConfig:
    """Configuration for Conservative Q-Learning."""
    gamma: float = 0.99
    lr: float = 3e-4
    alpha: float = 1.0  # CQL regularization weight
    target_update_tau: float = 0.005  # Soft update coefficient
    batch_size: int = 64
    memory_size: int = 50000
    initial_exploration: int = 1000  # Steps before training starts
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.995
    target_update_freq: int = 100  # Hard update frequency (steps)
    hidden_dim: int = 256
    num_cql_samples: int = 10  # Number of actions to sample for CQL loss
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_interval: int = 100  # Log stats every N episodes


class QNetwork(nn.Module):
    """Q-network architecture for CQL."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Experience replay buffer for offline/online data collection."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Add transition to buffer."""
        # Ensure states are 1D arrays for storage
        state = np.asarray(state).reshape(-1)
        next_state = np.asarray(next_state).reshape(-1)
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Sample batch of transitions."""
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer has only {len(self.buffer)} samples, requested {batch_size}")
        
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.stack(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.stack(next_states)),
            torch.FloatTensor(dones),
        )
    
    def __len__(self) -> int:
        return len(self.buffer)


class RunningNormalizer:
    """Online state normalizer with incremental updates."""
    
    def __init__(self, shape: Tuple[int, ...], eps: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = eps
        self.eps = eps
    
    def update(self, x: np.ndarray):
        """Update statistics with new observation(s)."""
        if x.ndim == 1:
            x = x[np.newaxis, :]
        
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        # Incremental update formulas
        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        
        # Update mean
        self.mean += delta * batch_count / total_count
        
        # Update variance
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize observation(s)."""
        if x.ndim == 1:
            return (x - self.mean) / (np.sqrt(self.var) + self.eps)
        return (x - self.mean) / (np.sqrt(self.var) + self.eps)


class CQLSolver:
    """
    Conservative Q-Learning agent for CartPole.
    
    Implements the standard API for train.py:
    - act(state, evaluation_mode=False) -> action
    - step(state, action, reward, next_state, done)
    - save(path), load(path), get_stats()
    """
    
    def __init__(self, observation_space: int, action_space: int, cfg: Optional[CQLConfig] = None):
        self.obs_dim = observation_space
        self.act_dim = action_space
        self.cfg = cfg or CQLConfig()
        self.device = torch.device(self.cfg.device)
        
        # Q-networks
        self.q_net = QNetwork(self.obs_dim, self.act_dim, self.cfg.hidden_dim).to(self.device)
        self.target_q_net = QNetwork(self.obs_dim, self.act_dim, self.cfg.hidden_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.cfg.lr)
        
        # Experience replay
        self.memory = ReplayBuffer(self.cfg.memory_size)
        
        # State normalizer
        self.normalizer = RunningNormalizer((self.obs_dim,))
        
        # Training statistics
        self.steps = 0
        self.episodes = 0
        self.exploration_rate = self.cfg.eps_start
        
        # For get_stats()
        self.last_loss = None
        self.last_td_loss = None
        self.last_cql_loss = None
        
        print(f"[CQL] Initialized with alpha={self.cfg.alpha}, buffer_size={self.cfg.memory_size}")
    
    def act(self, state_np: np.ndarray, evaluation_mode: bool = False) -> int:
        """
        Select action using ε-greedy policy.
        
        Args:
            state_np: Observation array of shape (1, obs_dim) or (obs_dim,)
            evaluation_mode: If True, act greedily without exploration
        
        Returns:
            Selected action (int)
        """
        # Normalize state
        if state_np.ndim == 1:
            state_np = state_np[np.newaxis, :]
        state_norm = self.normalizer.normalize(state_np)
        
        # Exploration
        if not evaluation_mode and np.random.rand() < self.exploration_rate:
            return random.randrange(self.act_dim)
        
        # Exploitation: greedy action from Q-network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_norm).to(self.device)
            q_values = self.q_net(state_tensor)
            action = torch.argmax(q_values, dim=1).item()
        
        return action
    
    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        Store transition and trigger learning.
        
        This method follows the train.py interface:
        - Store (s, a, r, s', done) in replay buffer
        - Update normalizer
        - Perform training step if enough samples
        """
        # Flatten states for storage
        state_flat = np.asarray(state).reshape(-1)
        next_state_flat = np.asarray(next_state).reshape(-1)
        
        # Store in replay buffer
        self.memory.push(state_flat, action, reward, next_state_flat, done)
        
        # Update normalizer with raw states
        self.normalizer.update(state_flat)
        if not done:  # Don't update with terminal state
            self.normalizer.update(next_state_flat)
        
        # Train if enough samples
        if len(self.memory) >= max(self.cfg.batch_size, self.cfg.initial_exploration):
            self._train_step()
        
        # Decay exploration rate
        if not done:
            self._decay_exploration()
        
        self.steps += 1
        if done:
            self.episodes += 1
    
    def _train_step(self):
        """Perform one training step with CQL loss."""
        try:
            # Sample batch from replay buffer
            states, actions, rewards, next_states, dones = self.memory.sample(self.cfg.batch_size)
            
            # Move to device
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)
            
            # Normalize states
            states_norm = self.normalizer.normalize(states.cpu().numpy())
            next_states_norm = self.normalizer.normalize(next_states.cpu().numpy())
            states_norm = torch.FloatTensor(states_norm).to(self.device)
            next_states_norm = torch.FloatTensor(next_states_norm).to(self.device)
            
            # Compute target Q-values
            with torch.no_grad():
                next_q_values = self.target_q_net(next_states_norm)
                max_next_q = torch.max(next_q_values, dim=1)[0]
                target_q = rewards + (1 - dones) * self.cfg.gamma * max_next_q
            
            # Compute current Q-values for taken actions
            current_q_values = self.q_net(states_norm)
            current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # TD loss (MSE between current Q and target)
            td_loss = F.mse_loss(current_q, target_q)
            
            # CQL conservative loss
            # 1. Compute logsumexp over actions
            random_actions = torch.randint(0, self.act_dim, (self.cfg.num_cql_samples, states.shape[0]),
                                          device=self.device)
            q_random = current_q_values.gather(1, random_actions.T).mean(dim=1)
            
            # 2. Compute Q-values for current policy (greedy)
            current_policy_q = torch.max(current_q_values, dim=1)[0]
            
            # 3. Conservative regularization term
            cql_loss = torch.mean(current_policy_q - q_random)
            
            # Total loss = TD loss + α * CQL loss
            total_loss = td_loss + self.cfg.alpha * cql_loss
            
            # Optimize
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Store loss values for monitoring
            self.last_loss = total_loss.item()
            self.last_td_loss = td_loss.item()
            self.last_cql_loss = cql_loss.item()
            
            # Soft update target network
            self._soft_update_target()
            
            # Periodic hard update (for stability)
            if self.steps % self.cfg.target_update_freq == 0:
                self._hard_update_target()
                
        except ValueError as e:
            # Not enough samples yet
            pass
    
    def _soft_update_target(self):
        """Soft update target network parameters."""
        with torch.no_grad():
            for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(
                    self.cfg.target_update_tau * param.data + 
                    (1 - self.cfg.target_update_tau) * target_param.data
                )
    
    def _hard_update_target(self):
        """Hard copy online network parameters to target network."""
        self.target_q_net.load_state_dict(self.q_net.state_dict())
    
    def _decay_exploration(self):
        """Exponential decay of exploration rate."""
        self.exploration_rate = max(
            self.cfg.eps_end,
            self.exploration_rate * self.cfg.eps_decay
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Return training statistics for monitoring.
        
        Returns:
            Dictionary with training metrics
        """
        return {
            "loss": self.last_loss,
            "td_loss": self.last_td_loss,
            "cql_loss": self.last_cql_loss,
            "epsilon": self.exploration_rate,
            "buffer_size": len(self.memory),
            "episodes": self.episodes,
            "steps": self.steps,
            "alpha": self.cfg.alpha
        }
    
    def save(self, path: str):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'target_q_net_state_dict': self.target_q_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'normalizer_mean': self.normalizer.mean,
            'normalizer_var': self.normalizer.var,
            'normalizer_count': self.normalizer.count,
            'exploration_rate': self.exploration_rate,
            'steps': self.steps,
            'episodes': self.episodes,
            'cfg': self.cfg.__dict__
        }, path)
        print(f"[CQL] Model saved to {path}")
    
    def load(self, path: str):
        """
        Load model checkpoint.
        
        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load networks
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_q_net.load_state_dict(checkpoint['target_q_net_state_dict'])
        
        # Load optimizer
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load normalizer
        if 'normalizer_mean' in checkpoint:
            self.normalizer.mean = checkpoint['normalizer_mean']
            self.normalizer.var = checkpoint['normalizer_var']
            self.normalizer.count = checkpoint['normalizer_count']
        
        # Load training state
        self.exploration_rate = checkpoint.get('exploration_rate', self.cfg.eps_start)
        self.steps = checkpoint.get('steps', 0)
        self.episodes = checkpoint.get('episodes', 0)
        
        print(f"[CQL] Model loaded from {path} (Episodes: {self.episodes}, Steps: {self.steps})")


# -----------------------------
# Convenience functions
# -----------------------------
def create_cql_agent(obs_dim: int, act_dim: int, **kwargs) -> CQLSolver:
    """
    Factory function to create CQL agent.
    
    Args:
        obs_dim: Observation dimension
        act_dim: Action dimension
        **kwargs: Override CQLConfig parameters
    
    Returns:
        CQLSolver instance
    """
    cfg = CQLConfig()
    for key, value in kwargs.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    
    return CQLSolver(obs_dim, act_dim, cfg)


# -----------------------------
# Example usage (for testing)
# -----------------------------
if __name__ == "__main__":
    # Quick test
    import gymnasium as gym
    
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    agent = CQLSolver(obs_dim, act_dim)
    
    # Test act method
    state, _ = env.reset()
    state = np.reshape(state, (1, obs_dim))
    action = agent.act(state)
    print(f"Test action: {action}")
    
    env.close()
