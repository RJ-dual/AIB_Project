# offline_cql.py
# ------------------------------------------------------
# Offline Conservative Q-Learning (Lagrange version)
# - Double Q
# - Target networks
# - Learnable alpha (Lagrange multiplier)
# ------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict


# =========================
# Config
# =========================
@dataclass
class OfflineCQLConfig:
    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 128
    tau: float = 0.005

    # CQL specific
    cql_temp: float = 1.0
    target_action_gap: float = 5.0  # Lagrange constraint

    hidden_dim: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Q Network
# =========================
class QNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =========================
# Offline CQL Agent
# =========================
class OfflineCQL:
    def __init__(self, obs_dim: int, act_dim: int, cfg: OfflineCQLConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.act_dim = act_dim

        # Q networks
        self.q1 = QNet(obs_dim, act_dim, cfg.hidden_dim).to(self.device)
        self.q2 = QNet(obs_dim, act_dim, cfg.hidden_dim).to(self.device)
        self.q1_target = QNet(obs_dim, act_dim, cfg.hidden_dim).to(self.device)
        self.q2_target = QNet(obs_dim, act_dim, cfg.hidden_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Optimizer
        self.q_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=cfg.lr,
        )

        # Lagrange multiplier alpha
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=cfg.lr)

    # -------------------------
    # Training update
    # -------------------------
    def update(self, batch: Tuple) -> Dict[str, float]:
        states, actions, rewards, next_states, dones = batch

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # -------- TD target --------
        with torch.no_grad():
            q1_next = self.q1_target(next_states)
            q2_next = self.q2_target(next_states)
            q_next = torch.min(q1_next, q2_next).max(dim=1, keepdim=True)[0]
            target_q = rewards + (1.0 - dones) * self.cfg.gamma * q_next

        # -------- Current Q --------
        q1 = self.q1(states).gather(1, actions)
        q2 = self.q2(states).gather(1, actions)

        td_loss = (
            nn.functional.mse_loss(q1, target_q)
            + nn.functional.mse_loss(q2, target_q)
        )

        # -------- CQL loss --------
        q1_all = self.q1(states)
        q2_all = self.q2(states)

        logsumexp_q = (
            torch.logsumexp(q1_all / self.cfg.cql_temp, dim=1).mean()
            + torch.logsumexp(q2_all / self.cfg.cql_temp, dim=1).mean()
        ) * (self.cfg.cql_temp / 2.0)

        q_data = (q1.mean() + q2.mean()) / 2.0
        cql_loss = logsumexp_q - q_data

        # -------- Lagrange update --------
        alpha = self.log_alpha.exp()
        alpha_loss = alpha * (cql_loss.detach() - self.cfg.target_action_gap)

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # -------- Total loss --------
        total_loss = td_loss + alpha.detach() * cql_loss

        self.q_optimizer.zero_grad()
        total_loss.backward()
        self.q_optimizer.step()

        # -------- Target network update --------
        with torch.no_grad():
            for t, s in zip(self.q1_target.parameters(), self.q1.parameters()):
                t.data.copy_(self.cfg.tau * s.data + (1 - self.cfg.tau) * t.data)
            for t, s in zip(self.q2_target.parameters(), self.q2.parameters()):
                t.data.copy_(self.cfg.tau * s.data + (1 - self.cfg.tau) * t.data)

        return {
            "total_loss": total_loss.item(),
            "td_loss": td_loss.item(),
            "cql_loss": cql_loss.item(),
            "alpha": alpha.item(),
        }

    # -------------------------
    # Evaluation action
    # -------------------------
    def act(self, state: np.ndarray) -> int:
        self.q1.eval()
        self.q2.eval()
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = torch.min(self.q1(s), self.q2(s))
            return int(q.argmax(dim=1).item())
