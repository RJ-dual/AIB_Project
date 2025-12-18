# build_dataset.py - 优化版，快速生成高质量数据
import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ENV_NAME = "CartPole-v1"
NUM_EPISODES = 1000  # 减少episode数量但提高质量
MAX_STEPS = 500
SAVE_PATH = "datasets/cartpole_high_quality.npz"

# 更高效的DQN模型
class EfficientDQN(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class FastDQNAgent:
    def __init__(self, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 使用更小的网络
        self.q_net = EfficientDQN(obs_dim, act_dim).to(self.device)
        self.target_net = EfficientDQN(obs_dim, act_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        
        # 快速衰减的epsilon
        self.epsilon_start = 1.0
        self.epsilon_final = 0.05
        self.epsilon_decay = 0.999
        
        # 经验回放（使用更小的缓冲区）
        self.buffer = []
        self.buffer_size = 5000
        self.batch_size = 128
        
        self.training_steps = 0
    
    def get_epsilon(self):
        return max(self.epsilon_final, 
                  self.epsilon_start * (self.epsilon_decay ** self.training_steps))
    
    def act(self, state, evaluation_mode=False):
        if np.random.random() < self.get_epsilon() and not evaluation_mode:
            return np.random.randint(self.act_dim)
        
        state_t = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return q_values.argmax().item()
    
    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None
        
        # 随机采样
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # 计算当前Q值
        current_q = self.q_net(states).gather(1, actions)
        
        # 计算目标Q值（Double DQN）
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * 0.99 * next_q
        
        # 计算损失
        loss = nn.MSELoss()(current_q, target_q)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        
        # 软更新目标网络
        tau = 0.01
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        self.training_steps += 1
        return loss.item()

def collect_expert_demonstrations():
    """使用预训练模型快速收集专家演示"""
    os.makedirs("datasets", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    print(f"Collecting expert demonstrations for {ENV_NAME}...")
    
    # 使用简单的启发式策略生成高质量数据
    buffer = {
        "states": [],
        "actions": [],
        "rewards": [],
        "next_states": [],
        "dones": [],
    }
    
    total_transitions = 0
    
    # 使用简单的平衡策略快速收集数据
    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < MAX_STEPS:
            # 简单的启发式策略：根据杆的角度和位置选择动作
            position, velocity, angle, angular_velocity = state
            
            # 平衡策略：当杆向右倾斜时推车向右，反之向左
            if angle > 0:
                action = 1  # 向右
            else:
                action = 0  # 向左
            
            # 添加一些随机性以增加数据多样性
            if np.random.random() < 0.1:
                action = 1 - action  # 翻转动作
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 存储转换
            buffer["states"].append(state.copy())
            buffer["actions"].append(action)
            buffer["rewards"].append(1.0)  # 每一步都给予奖励
            buffer["next_states"].append(next_state.copy())
            buffer["dones"].append(done)
            
            state = next_state
            steps += 1
            total_transitions += 1
        
        if (episode + 1) % 100 == 0:
            print(f"Collected {episode + 1} episodes, {total_transitions} transitions")
    
    env.close()
    
    # 转换为numpy数组
    dataset = {k: np.array(v) for k, v in buffer.items()}
    
    # 保存数据集
    np.savez_compressed(SAVE_PATH, **dataset)
    
    # 分析数据集
    dones = dataset["dones"]
    episode_lengths = []
    length = 0
    for d in dones:
        length += 1
        if d:
            episode_lengths.append(length)
            length = 0
    
    print(f"\n✅ Dataset saved to {SAVE_PATH}")
    print(f"   Total transitions: {len(dataset['states'])}")
    print(f"   Average episode length: {np.mean(episode_lengths):.2f}")
    print(f"   Max episode length: {np.max(episode_lengths)}")
    
    return dataset

def build_hybrid_dataset():
    """构建混合数据集：专家演示 + 随机探索"""
    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    buffer = {
        "states": [],
        "actions": [],
        "rewards": [],
        "next_states": [],
        "dones": [],
    }
    
    print("Building hybrid dataset...")
    
    # 第一部分：专家演示（70%）
    expert_episodes = int(NUM_EPISODES * 0.7)
    for episode in range(expert_episodes):
        state, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < MAX_STEPS:
            # 简单的平衡策略
            _, _, angle, _ = state
            action = 1 if angle > 0 else 0
            
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            buffer["states"].append(state.copy())
            buffer["actions"].append(action)
            buffer["rewards"].append(1.0)
            buffer["next_states"].append(next_state.copy())
            buffer["dones"].append(done)
            
            state = next_state
            steps += 1
    
    # 第二部分：随机探索（30%）
    random_episodes = NUM_EPISODES - expert_episodes
    for episode in range(random_episodes):
        state, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < MAX_STEPS:
            # 随机动作以探索状态空间
            action = np.random.randint(act_dim)
            
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            buffer["states"].append(state.copy())
            buffer["actions"].append(action)
            buffer["rewards"].append(1.0 if not done else 0)
            buffer["next_states"].append(next_state.copy())
            buffer["dones"].append(done)
            
            state = next_state
            steps += 1
    
    env.close()
    
    # 保存数据集
    dataset = {k: np.array(v) for k, v in buffer.items()}
    np.savez_compressed("datasets/cartpole_hybrid.npz", **dataset)
    
    print(f"✅ Hybrid dataset saved with {len(dataset['states'])} transitions")
    return dataset

if __name__ == "__main__":
    # 快速生成高质量数据集
    dataset = collect_expert_demonstrations()
    
    # 可选：构建混合数据集
    # dataset = build_hybrid_dataset()