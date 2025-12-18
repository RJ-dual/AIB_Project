# offline_cql.py - 优化版，更高效的实现
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class OfflineCQLConfig:
    # 核心参数
    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 128
    tau: float = 0.005
    
    # CQL参数
    cql_alpha: float = 1.0
    cql_temp: float = 1.0
    
    # 网络参数
    hidden_dim: int = 128
    
    # 训练参数
    steps: int = 50_000
    
    @property
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"


class LightweightQNet(nn.Module):
    """轻量级Q网络"""
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )
        
        # 简单初始化
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FastCQL:
    """快速收敛的CQL实现"""
    def __init__(self, obs_dim: int, act_dim: int, cfg: OfflineCQLConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.act_dim = act_dim
        
        # 使用更小的网络
        self.q1 = LightweightQNet(obs_dim, act_dim, cfg.hidden_dim).to(self.device)
        self.q2 = LightweightQNet(obs_dim, act_dim, cfg.hidden_dim).to(self.device)
        
        # 目标网络
        self.q1_target = LightweightQNet(obs_dim, act_dim, cfg.hidden_dim).to(self.device)
        self.q2_target = LightweightQNet(obs_dim, act_dim, cfg.hidden_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=cfg.lr,
            weight_decay=1e-4
        )
        
        # 学习率调度
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=1000
        )
        
        self.total_steps = 0
    
    def update(self, batch: Tuple) -> Dict[str, float]:
        """简化但高效的更新"""
        states, actions, rewards, next_states, dones = batch
        
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        batch_size = states.shape[0]
        
        # TD目标
        with torch.no_grad():
            next_q1 = self.q1_target(next_states)
            next_q2 = self.q2_target(next_states)
            next_q = torch.min(next_q1, next_q2)
            next_q_max = next_q.max(1, keepdim=True)[0]
            target = rewards + (1 - dones) * self.cfg.gamma * next_q_max
        
        # 当前Q值
        q1 = self.q1(states).gather(1, actions)
        q2 = self.q2(states).gather(1, actions)
        
        # TD损失
        td_loss1 = nn.functional.mse_loss(q1, target)
        td_loss2 = nn.functional.mse_loss(q2, target)
        td_loss = (td_loss1 + td_loss2) / 2
        
        # 简化的CQL损失
        # 数据动作的Q值
        q_data = (q1.mean() + q2.mean()) / 2
        
        # 计算logsumexp（简化版）
        # 使用当前状态的所有动作
        q1_all = self.q1(states)
        q2_all = self.q2(states)
        
        cql_loss1 = torch.logsumexp(q1_all / self.cfg.cql_temp, dim=1).mean() * self.cfg.cql_temp
        cql_loss2 = torch.logsumexp(q2_all / self.cfg.cql_temp, dim=1).mean() * self.cfg.cql_temp
        cql_loss = (cql_loss1 + cql_loss2) / 2 - q_data
        
        # 总损失
        total_loss = td_loss + self.cfg.cql_alpha * cql_loss
        
        # 优化
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            max_norm=1.0
        )
        
        self.optimizer.step()
        
        # 学习率调度
        self.scheduler.step(total_loss.item())
        
        # 软更新目标网络
        tau = self.cfg.tau
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        self.total_steps += 1
        
        return {
            "total_loss": total_loss.item(),
            "td_loss": td_loss.item(),
            "cql_loss": cql_loss.item(),
        }
    
    def act(self, state: np.ndarray, evaluation_mode: bool = False) -> int:
        """选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            q1 = self.q1(state_tensor)
            q2 = self.q2(state_tensor)
            q_min = torch.min(q1, q2)  # 悲观估计
            return q_min.argmax().item()
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'q1_target_state_dict': self.q1_target.state_dict(),
            'q2_target_state_dict': self.q2_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.cfg,
            'total_steps': self.total_steps,
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
        self.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'config' in checkpoint:
            self.cfg = checkpoint['config']
        
        self.total_steps = checkpoint.get('total_steps', 0)


# 为了兼容性，保持原来的类名
OfflineCQL = FastCQL