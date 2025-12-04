#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A2C for CartPole-v1 — Stable, debuggable, and fast-converging (≤200 episodes).
Author: Assistant (2025)
修正点：
1. 修复GAE中next_values的最后一步逻辑（区分done/步数终止）
2. 修正状态归一化的数据泄露问题
3. 统一Critic损失权重命名
4. 优化act()的维度兼容性
5. 完善模型加载时的cfg恢复逻辑
6. 优化变量命名（熵项更清晰）
"""

from __future__ import annotations
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
from dataclasses import dataclass
from typing import List, Tuple
import random
import os

# ----------------------------
# Config
# ----------------------------
@dataclass
class ActorCriticConfig:
    env_name: str = "CartPole-v1"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 1e-3  # 提高学习率
    hidden_size: int = 64
    beta_entropy_init: float = 0.01
    beta_entropy_final: float = 0.001
    entropy_decay_episodes: int = 300  # over ~300 episodes
    max_grad_norm: float = 0.5
    steps_per_update: int = 200
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "a2c_cartpole.pth"
    eval_episodes: int = 5
    critic_loss_weight: float = 0.5  # 显式定义Critic损失权重


# ----------------------------
# Network
# ----------------------------
class ActorCriticNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),  # Tanh more stable for small nets
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, act_dim)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(x)
        logits = self.actor(shared)       # [B, A]
        value = self.critic(shared).squeeze(-1)  # [B]
        return logits, value


# ----------------------------
# Running State Normalizer
# ----------------------------
class RunningNormalizer:
    def __init__(self, shape: Tuple[int, ...], eps: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = eps
        self.eps = eps

    def update(self, x: np.ndarray):
        """Update with a batch of observations (2D array: [batch_size, dim])"""
        if x.ndim == 1:
            x = x[np.newaxis, :]
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        self.mean += delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize observations (supports 1D/2D input)"""
        if x.ndim == 1:
            return (x - self.mean) / (np.sqrt(self.var) + self.eps)
        return (x - self.mean) / (np.sqrt(self.var) + self.eps)


# ----------------------------
# A2C Agent
# ----------------------------
class ActorCriticSolver:
    def __init__(self, observation_space: int, action_space: int, cfg: ActorCriticConfig | None = None):
        """
        Compatible with train.py interface:
          - observation_space: int (dim)
          - action_space: int (num actions)
        """
        self.obs_dim = observation_space
        self.act_dim = action_space
        self.cfg = cfg or ActorCriticConfig()
        self.device = torch.device(self.cfg.device)

        # Set seed
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        random.seed(self.cfg.seed)

        # Model & optimizer
        self.net = ActorCriticNet(self.obs_dim, self.act_dim, self.cfg.hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.cfg.lr)

        # State normalizer (online)
        self.state_normalizer = RunningNormalizer((self.obs_dim,))

        # Buffers
        self.reset_buffers()

        # Stats
        self.episode_count = 0
        self.total_steps = 0
        self.current_beta_entropy = self.cfg.beta_entropy_init

        # Last update stats (for get_stats)
        self.last_loss = None
        self.last_actor_loss = None
        self.last_critic_loss = None
        self.last_entropy_term = None

    def reset_buffers(self):
        self.states: List[np.ndarray] = []   # raw states, shape (obs_dim,)
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []  # V(s_t) at time t

    def act(self, state_np: np.ndarray, evaluation_mode: bool = False) -> int:
        """
        Compatible with train.py:
          state_np: np.ndarray of shape (1, obs_dim) or (obs_dim,)
          return: int (action)
        """
        # Ensure state is 2D (batch, dim) for consistency
        if state_np.ndim == 1:
            state_np = state_np[np.newaxis, :]
        
        # Normalize state
        state_norm = self.state_normalizer.normalize(state_np)
        state_t = torch.as_tensor(state_norm, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits, value = self.net(state_t)
            distro = dist.Categorical(logits=logits)

            if evaluation_mode:
                action = torch.argmax(logits, dim=-1).item()
            else:
                action = distro.sample().item()
                # Buffer raw state (for normalizer update later)
                self.states.append(state_np[0].copy())  # 取batch中第一个样本
                self.actions.append(action)
                self.values.append(value[0].item())  # 取第一个样本的价值
        
        return action

    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        train.py calls: agent.step(state, action, reward, next_state, done)
        Record reward/done and trigger update when needed
        """
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.total_steps += 1

        # Trigger update on steps threshold or episode done
        if len(self.rewards) >= self.cfg.steps_per_update or done:
            if done:
                self.episode_count += 1
            self._update()
            self.reset_buffers()

    def _compute_gae(self) -> torch.Tensor:
        """Compute Generalized Advantage Estimation (GAE)"""
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones, dtype=torch.bool, device=self.device)
        values = torch.tensor(self.values, dtype=torch.float32, device=self.device)

        # Calculate next values (critical fix: handle done vs step termination)
        if len(self.dones) > 0 and self.dones[-1]:
            # Episode terminated naturally: next state value is 0
            next_value = torch.tensor(0.0, device=self.device)
        else:
            # Episode truncated by step limit: use V(s_T) as next value
            last_state = self.states[-1]
            last_state_norm = self.state_normalizer.normalize(last_state)
            last_state_t = torch.as_tensor(last_state_norm, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                _, next_value = self.net(last_state_t.unsqueeze(0))
            next_value = next_value.squeeze(0)

        # Build next_values array: [V(s1), V(s2), ..., V(sT), next_value]
        next_values = torch.cat([values[1:], next_value.unsqueeze(0)])
        
        # Compute TD deltas
        deltas = rewards + self.cfg.gamma * next_values * (~dones) - values

        # Compute GAE
        advantages = torch.zeros_like(deltas)
        gae = 0.0
        for t in reversed(range(len(deltas))):
            gae = deltas[t] + self.cfg.gamma * self.cfg.gae_lambda * gae * (~dones[t])
            advantages[t] = gae
        
        return advantages.detach()

    def _update(self):
        """Perform A2C update with current buffer data"""
        if len(self.rewards) == 0:
            return

        # Step 1: Process states (fix data leakage in normalization)
        states_batch = np.stack(self.states)  # (T, obs_dim)
        # Normalize with current parameters BEFORE updating normalizer
        states_norm = self.state_normalizer.normalize(states_batch)
        # Update normalizer with raw states (no leakage)
        self.state_normalizer.update(states_batch)

        # Convert to tensors
        states_t = torch.as_tensor(states_norm, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(self.actions, dtype=torch.long, device=self.device)

        # Forward pass
        logits, values_pred = self.net(states_t)
        distro = dist.Categorical(logits=logits)
        log_probs = distro.log_prob(actions_t)
        entropies = distro.entropy()

        # Compute advantages and returns
        advantages = self._compute_gae()
        returns = advantages + values_pred.detach()
        
        # Normalize advantages (stabilize training)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Calculate losses
        actor_loss = -(log_probs * advantages).mean()  # Maximize policy gradient
        critic_loss = (returns - values_pred).pow(2).mean()  # MSE loss (no internal 0.5)
        entropy_term = -entropies.mean()  # Entropy regularization term (maximize entropy)

        # Entropy coefficient annealing
        if self.episode_count < self.cfg.entropy_decay_episodes:
            frac = 1.0 - self.episode_count / self.cfg.entropy_decay_episodes
            self.current_beta_entropy = (
                self.cfg.beta_entropy_final +
                frac * (self.cfg.beta_entropy_init - self.cfg.beta_entropy_final)
            )
        else:
            self.current_beta_entropy = self.cfg.beta_entropy_final

        # Total loss (weighted combination)
        total_loss = (
            actor_loss +
            self.cfg.critic_loss_weight * critic_loss +
            self.current_beta_entropy * entropy_term
        )

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()

        # Record stats for monitoring
        self.last_loss = total_loss.item()
        self.last_actor_loss = actor_loss.item()
        self.last_critic_loss = critic_loss.item()
        self.last_entropy_term = entropy_term.item()

        # Logging (reduce frequency to avoid spam)
        if self.episode_count % 50 == 0:
            print(
                f"[A2C] Episode {self.episode_count:4d} | "
                f"Steps: {len(self.rewards):3d} | "
                f"Total Loss: {total_loss.item():6.3f} | "
                f"Actor: {actor_loss.item():5.2f} | "
                f"Critic: {critic_loss.item():5.2f} | "
                f"Entropy Term: {entropy_term.item():5.2f} | "
                f"Beta: {self.current_beta_entropy:.4f}"
            )

    def get_stats(self):
        """Required by train.py - return training statistics"""
        return {
            "last_loss": self.last_loss,
            "last_actor_loss": self.last_actor_loss,
            "last_critic_loss": self.last_critic_loss,
            "last_entropy_term": self.last_entropy_term,
            "episodes": self.episode_count,
            "total_steps": self.total_steps,
            "current_beta_entropy": self.current_beta_entropy
        }

    def save(self, path: str | None = None):
        """Save model checkpoint"""
        save_path = path or self.cfg.save_path
        torch.save({
            'net_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'normalizer_mean': self.state_normalizer.mean,
            'normalizer_var': self.state_normalizer.var,
            'normalizer_count': self.state_normalizer.count,
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'cfg_dict': self.cfg.__dict__,
            'current_beta_entropy': self.current_beta_entropy
        }, save_path)
        print(f"Model saved to {save_path}")

    def load(self, path: str | None = None):
        """Load model checkpoint"""
        load_path = path or self.cfg.save_path
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")
        
        ckpt = torch.load(load_path, map_location=self.device)
        
        # Restore model and optimizer
        self.net.load_state_dict(ckpt['net_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        
        # Restore normalizer
        self.state_normalizer.mean = ckpt['normalizer_mean']
        self.state_normalizer.var = ckpt['normalizer_var']
        self.state_normalizer.count = ckpt['normalizer_count']
        
        # Restore training stats
        self.episode_count = ckpt.get('episode_count', 0)
        self.total_steps = ckpt.get('total_steps', 0)
        self.current_beta_entropy = ckpt.get('current_beta_entropy', self.cfg.beta_entropy_init)
        
        # Restore config (overwrite current cfg with saved values)
        cfg_dict = ckpt.get('cfg_dict', {})
        for k, v in cfg_dict.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)
        
        print(f"Model loaded from {load_path} (Episodes: {self.episode_count}, Steps: {self.total_steps})")


