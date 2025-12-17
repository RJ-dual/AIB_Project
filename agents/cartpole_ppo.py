

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PPOConfig:
    policy_lr: float = 3e-4
    value_lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    update_epochs: int = 4
    mini_batch_size: int = 64
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    value_hidden_size: int = 128
    policy_hidden_sizes: tuple = (64, 64)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    target_kl: Optional[float] = 0.01


class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: tuple):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))  # logits
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PPOSolver:
    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig | None = None):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cfg: PPOConfig = cfg or PPOConfig()
        self.device = torch.device(self.cfg.device)

        self.policy_net = PolicyNetwork(self.obs_dim, self.act_dim, self.cfg.policy_hidden_sizes).to(self.device)
        self.value_net = ValueNetwork(self.obs_dim, self.cfg.value_hidden_size).to(self.device)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.cfg.policy_lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.cfg.value_lr)

        self.reset_buffers()
        self.steps = 0
        self.episodes = 0

    def reset_buffers(self):
        self.buf_states: List[np.ndarray] = []
        self.buf_actions: List[int] = []
        self.buf_rewards: List[float] = []
        self.buf_dones: List[bool] = []
        self.buf_log_probs: List[float] = []

    def act(self, state_np: np.ndarray, evaluation_mode: bool = False) -> int:
        # 不在 act 中写入 buffer，训练循环会调用 step 负责记录
        state = torch.as_tensor(state_np, dtype=torch.float32, device=self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            logits = self.policy_net(state)
            dist = torch.distributions.Categorical(logits=logits)
            if evaluation_mode:
                action = torch.argmax(logits, dim=-1).item()
            else:
                action = dist.sample().item()
        return action

    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        与训练循环一致的接口：step(state, action, reward, next_state, done)
        在这里统一记录 state/action/log_prob/reward/done；若 done 则触发 _update_networks。
        """
        # 记录 state/action/reward/done/log_prob
        # state 可能是 shape (1, obs_dim) 或 (obs_dim,), 把它转换为 1D numpy
        s = np.asarray(state).reshape(-1)
        self.buf_states.append(s)
        self.buf_actions.append(int(action))
        self.buf_rewards.append(float(reward))
        self.buf_dones.append(bool(done))

        # 计算 log_prob（无梯度）
        state_tensor = torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.policy_net(state_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(torch.tensor(action, device=self.device)).item()
        self.buf_log_probs.append(logp)

        self.steps += 1

        if done:
            self.episodes += 1
            self._update_networks()
            self.reset_buffers()

    def _compute_gae_and_returns(self, values: np.ndarray, rewards: np.ndarray, dones: np.ndarray):
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_adv = 0.0
        for t in reversed(range(T)):
            next_value = 0.0 if t == T - 1 else values[t + 1]
            not_done = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.cfg.gamma * next_value * not_done - values[t]
            last_adv = delta + self.cfg.gamma * self.cfg.gae_lambda * last_adv * not_done
            advantages[t] = last_adv
        returns = advantages + values
        return advantages, returns

    def _update_networks(self):
        if len(self.buf_states) == 0:
            return

        states = np.vstack(self.buf_states).astype(np.float32)
        actions = np.array(self.buf_actions, dtype=np.int64)
        rewards = np.array(self.buf_rewards, dtype=np.float32)
        dones = np.array(self.buf_dones, dtype=np.bool_)
        old_log_probs = np.array(self.buf_log_probs, dtype=np.float32)

        states_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs_tensor = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)

        # 计算 values（更新前）
        with torch.no_grad():
            values_tensor = self.value_net(states_tensor)
            values = values_tensor.cpu().numpy()

        # GAE + returns
        advantages_np, returns_np = self._compute_gae_and_returns(values, rewards, dones)
        advantages_tensor = torch.as_tensor(advantages_np, dtype=torch.float32, device=self.device)
        returns_tensor = torch.as_tensor(returns_np, dtype=torch.float32, device=self.device)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        N = len(states)
        batch_size = max(1, min(self.cfg.mini_batch_size, N))
        indices = np.arange(N)

        for epoch in range(self.cfg.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, N, batch_size):
                mb_idx = indices[start:start + batch_size]
                mb_states = states_tensor[mb_idx]
                mb_actions = actions_tensor[mb_idx]
                mb_old_log_probs = old_log_probs_tensor[mb_idx]
                mb_advantages = advantages_tensor[mb_idx]
                mb_returns = returns_tensor[mb_idx]

                logits = self.policy_net(mb_states)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean() - self.cfg.entropy_coef * entropy

                value_preds = self.value_net(mb_states)
                value_loss = F.mse_loss(value_preds, mb_returns)

                total_loss = policy_loss + self.cfg.value_coef * value_loss

                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.cfg.max_grad_norm)
                self.policy_optimizer.step()
                self.value_optimizer.step()

            # approx kl 计算（整批）
            with torch.no_grad():
                logits_all = self.policy_net(states_tensor)
                dist_all = torch.distributions.Categorical(logits=logits_all)
                new_log_probs_all = dist_all.log_prob(actions_tensor)
                approx_kl = (old_log_probs_tensor - new_log_probs_all).mean().item()
            if self.cfg.target_kl is not None and approx_kl > self.cfg.target_kl:
                break

        # 打印统计信息
        with torch.no_grad():
            logits_all = self.policy_net(states_tensor)
            dist_all = torch.distributions.Categorical(logits=logits_all)
            entropy_all = dist_all.entropy().mean().item()
            new_log_probs_all = dist_all.log_prob(actions_tensor)
            approx_kl = (old_log_probs_tensor - new_log_probs_all).mean().item()
            clip_frac = ((torch.exp(new_log_probs_all - old_log_probs_tensor) > 1.0 + self.cfg.clip_epsilon) |
                         (torch.exp(new_log_probs_all - old_log_probs_tensor) < 1.0 - self.cfg.clip_epsilon)).float().mean().item()
        print(f"Episode {self.episodes}: steps={len(rewards)}, Return={rewards.sum():.2f}, Entropy={entropy_all:.4f}, KL={approx_kl:.6f}, ClipFrac={clip_frac:.4f}")

    def save(self, path: str):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_opt': self.policy_optimizer.state_dict(),
            'value_opt': self.value_optimizer.state_dict(),
            'cfg': self.cfg.__dict__,
            'steps': self.steps,
            'episodes': self.episodes
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt['policy_state_dict'])
        self.value_net.load_state_dict(ckpt['value_state_dict'])
        self.policy_optimizer.load_state_dict(ckpt['policy_opt'])
        self.value_optimizer.load_state_dict(ckpt['value_opt'])
        self.steps = ckpt.get('steps', 0)
        self.episodes = ckpt.get('episodes', 0)
    # 在 PPOSolver 类中添加
    def get_stats(self):
        """
    返回一个 dict，包含 train.py 期望的监控项（可按需增删）
     """
        return {
        "entropy": getattr(self, "last_entropy", None),
        "kl": getattr(self, "last_kl", None),
        "clip_frac": getattr(self, "last_clip_frac", None),
        "actor_loss": getattr(self, "last_actor_loss", None),
        "critic_loss": getattr(self, "last_critic_loss", None),
        "lr": self.optimizer.param_groups[0]["lr"] if hasattr(self, "optimizer") else None,
        "episode": getattr(self, "episode", None),
    }


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PPOConfig:
    policy_lr: float = 3e-4
    value_lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    update_epochs: int = 4
    mini_batch_size: int = 64
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    value_hidden_size: int = 128
    policy_hidden_sizes: tuple = (64, 64)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    target_kl: Optional[float] = 0.01


class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: tuple):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))  # logits
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PPOSolver:
    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig | None = None):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cfg: PPOConfig = cfg or PPOConfig()
        self.device = torch.device(self.cfg.device)

        self.policy_net = PolicyNetwork(self.obs_dim, self.act_dim, self.cfg.policy_hidden_sizes).to(self.device)
        self.value_net = ValueNetwork(self.obs_dim, self.cfg.value_hidden_size).to(self.device)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.cfg.policy_lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.cfg.value_lr)

        self.reset_buffers()
        self.steps = 0
        self.episodes = 0

    def reset_buffers(self):
        self.buf_states: List[np.ndarray] = []
        self.buf_actions: List[int] = []
        self.buf_rewards: List[float] = []
        self.buf_dones: List[bool] = []
        self.buf_log_probs: List[float] = []

    def act(self, state_np: np.ndarray, evaluation_mode: bool = False) -> int:
        # 不在 act 中写入 buffer，训练循环会调用 step 负责记录
        state = torch.as_tensor(state_np, dtype=torch.float32, device=self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            logits = self.policy_net(state)
            dist = torch.distributions.Categorical(logits=logits)
            if evaluation_mode:
                action = torch.argmax(logits, dim=-1).item()
            else:
                action = dist.sample().item()
        return action

    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        与训练循环一致的接口：step(state, action, reward, next_state, done)
        在这里统一记录 state/action/log_prob/reward/done；若 done 则触发 _update_networks。
        """
        # 记录 state/action/reward/done/log_prob
        # state 可能是 shape (1, obs_dim) 或 (obs_dim,), 把它转换为 1D numpy
        s = np.asarray(state).reshape(-1)
        self.buf_states.append(s)
        self.buf_actions.append(int(action))
        self.buf_rewards.append(float(reward))
        self.buf_dones.append(bool(done))

        # 计算 log_prob（无梯度）
        state_tensor = torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.policy_net(state_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(torch.tensor(action, device=self.device)).item()
        self.buf_log_probs.append(logp)

        self.steps += 1

        if done:
            self.episodes += 1
            self._update_networks()
            self.reset_buffers()

    def _compute_gae_and_returns(self, values: np.ndarray, rewards: np.ndarray, dones: np.ndarray):
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_adv = 0.0
        for t in reversed(range(T)):
            next_value = 0.0 if t == T - 1 else values[t + 1]
            not_done = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.cfg.gamma * next_value * not_done - values[t]
            last_adv = delta + self.cfg.gamma * self.cfg.gae_lambda * last_adv * not_done
            advantages[t] = last_adv
        returns = advantages + values
        return advantages, returns

    def _update_networks(self):
        if len(self.buf_states) == 0:
            return

        states = np.vstack(self.buf_states).astype(np.float32)
        actions = np.array(self.buf_actions, dtype=np.int64)
        rewards = np.array(self.buf_rewards, dtype=np.float32)
        dones = np.array(self.buf_dones, dtype=np.bool_)
        old_log_probs = np.array(self.buf_log_probs, dtype=np.float32)

        states_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs_tensor = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)

        # 计算 values（更新前）
        with torch.no_grad():
            values_tensor = self.value_net(states_tensor)
            values = values_tensor.cpu().numpy()

        # GAE + returns
        advantages_np, returns_np = self._compute_gae_and_returns(values, rewards, dones)
        advantages_tensor = torch.as_tensor(advantages_np, dtype=torch.float32, device=self.device)
        returns_tensor = torch.as_tensor(returns_np, dtype=torch.float32, device=self.device)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        N = len(states)
        batch_size = max(1, min(self.cfg.mini_batch_size, N))
        indices = np.arange(N)

        for epoch in range(self.cfg.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, N, batch_size):
                mb_idx = indices[start:start + batch_size]
                mb_states = states_tensor[mb_idx]
                mb_actions = actions_tensor[mb_idx]
                mb_old_log_probs = old_log_probs_tensor[mb_idx]
                mb_advantages = advantages_tensor[mb_idx]
                mb_returns = returns_tensor[mb_idx]

                logits = self.policy_net(mb_states)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean() - self.cfg.entropy_coef * entropy

                value_preds = self.value_net(mb_states)
                value_loss = F.mse_loss(value_preds, mb_returns)

                total_loss = policy_loss + self.cfg.value_coef * value_loss

                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.cfg.max_grad_norm)
                self.policy_optimizer.step()
                self.value_optimizer.step()

            # approx kl 计算（整批）
            with torch.no_grad():
                logits_all = self.policy_net(states_tensor)
                dist_all = torch.distributions.Categorical(logits=logits_all)
                new_log_probs_all = dist_all.log_prob(actions_tensor)
                approx_kl = (old_log_probs_tensor - new_log_probs_all).mean().item()
            if self.cfg.target_kl is not None and approx_kl > self.cfg.target_kl:
                break

        # 打印统计信息
        with torch.no_grad():
            logits_all = self.policy_net(states_tensor)
            dist_all = torch.distributions.Categorical(logits=logits_all)
            entropy_all = dist_all.entropy().mean().item()
            new_log_probs_all = dist_all.log_prob(actions_tensor)
            approx_kl = (old_log_probs_tensor - new_log_probs_all).mean().item()
            clip_frac = ((torch.exp(new_log_probs_all - old_log_probs_tensor) > 1.0 + self.cfg.clip_epsilon) |
                         (torch.exp(new_log_probs_all - old_log_probs_tensor) < 1.0 - self.cfg.clip_epsilon)).float().mean().item()
        print(f"Episode {self.episodes}: steps={len(rewards)}, Return={rewards.sum():.2f}, Entropy={entropy_all:.4f}, KL={approx_kl:.6f}, ClipFrac={clip_frac:.4f}")

    def save(self, path: str):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_opt': self.policy_optimizer.state_dict(),
            'value_opt': self.value_optimizer.state_dict(),
            'cfg': self.cfg.__dict__,
            'steps': self.steps,
            'episodes': self.episodes
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt['policy_state_dict'])
        self.value_net.load_state_dict(ckpt['value_state_dict'])
        self.policy_optimizer.load_state_dict(ckpt['policy_opt'])
        self.value_optimizer.load_state_dict(ckpt['value_opt'])
        self.steps = ckpt.get('steps', 0)
        self.episodes = ckpt.get('episodes', 0)
    # 在 PPOSolver 类中添加
    def get_stats(self):
        """
    返回一个 dict，包含 train.py 期望的监控项（可按需增删）
     """
        return {
        "entropy": getattr(self, "last_entropy", None),
        "kl": getattr(self, "last_kl", None),
        "clip_frac": getattr(self, "last_clip_frac", None),
        "actor_loss": getattr(self, "last_actor_loss", None),
        "critic_loss": getattr(self, "last_critic_loss", None),
        "lr": self.optimizer.param_groups[0]["lr"] if hasattr(self, "optimizer") else None,
        "episode": getattr(self, "episode", None),
    }
