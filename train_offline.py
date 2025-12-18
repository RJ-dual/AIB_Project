# train_offline.py - 优化版，减少训练次数提升效果
import os
import numpy as np
import torch
import gymnasium as gym
import time
from collections import deque
from offline_cql import OfflineCQL, OfflineCQLConfig

ENV_NAME = "CartPole-v1"
MODEL_PATH = "models/cartpole_cql_fast.pth"

def load_and_augment_dataset(path: str):
    """加载并增强数据集"""
    if not os.path.exists(path):
        # 如果没有数据集，使用简单策略快速生成
        print("No dataset found, creating simple dataset...")
        return create_simple_dataset()
    
    data = np.load(path)
    dataset = {k: data[k] for k in data.files}
    
    # 数据增强：添加噪声以增加多样性
    print("Augmenting dataset with noise...")
    n_samples = len(dataset["states"])
    
    # 复制数据并添加轻微噪声
    states_noisy = dataset["states"] + np.random.normal(0, 0.01, dataset["states"].shape)
    next_states_noisy = dataset["next_states"] + np.random.normal(0, 0.01, dataset["next_states"].shape)
    
    # 合并原始数据和增强数据
    augmented = {
        "states": np.concatenate([dataset["states"], states_noisy]),
        "actions": np.concatenate([dataset["actions"], dataset["actions"]]),
        "rewards": np.concatenate([dataset["rewards"], dataset["rewards"]]),
        "next_states": np.concatenate([dataset["next_states"], next_states_noisy]),
        "dones": np.concatenate([dataset["dones"], dataset["dones"]]),
    }
    
    print(f"Dataset augmented from {n_samples} to {len(augmented['states'])} samples")
    return augmented

def create_simple_dataset():
    """快速创建简单数据集"""
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
    
    print("Creating simple dataset with 200 episodes...")
    
    for episode in range(200):
        state, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 500:
            # 简单平衡策略
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
    
    env.close()
    
    dataset = {k: np.array(v) for k, v in buffer.items()}
    print(f"Created dataset with {len(dataset['states'])} transitions")
    return dataset

def fast_train_cql(
    dataset_path: str = "datasets/cartpole_high_quality.npz",
    steps: int = 50_000,  # 大幅减少训练步数
    test_every: int = 5_000,
):
    """快速训练CQL"""
    os.makedirs("models", exist_ok=True)
    
    # 加载并增强数据集
    dataset = load_and_augment_dataset(dataset_path)
    obs_dim = dataset["states"].shape[1]
    act_dim = int(np.max(dataset["actions"]) + 1)
    
    print(f"\n[Fast Training] Dataset: {len(dataset['states'])} transitions")
    print(f"[Fast Training] State dim: {obs_dim}, Action dim: {act_dim}")
    
    # 使用优化配置
    cfg = OfflineCQLConfig(
        lr=5e-4,  # 稍高的学习率
        batch_size=128,  # 较小的批大小
        cql_alpha=1.0,  # 降低CQL权重
        gamma=0.99,
        tau=0.01,  # 更快的目标网络更新
        hidden_dim=128,  # 较小的网络
    )
    
    agent = OfflineCQL(obs_dim, act_dim, cfg)
    
    N = len(dataset["states"])
    batch_size = min(cfg.batch_size, N // 10)
    
    print(f"[Fast Training] Steps: {steps}, Batch size: {batch_size}")
    print(f"[Fast Training] Using device: {agent.device}")
    
    # 创建测试环境
    test_env = gym.make(ENV_NAME)
    
    # 训练循环
    losses = []
    test_scores = []
    
    print("\nStarting fast training...")
    start_time = time.time()
    
    for step in range(1, steps + 1):
        # 采样批次
        idx = np.random.randint(0, N, size=batch_size)
        
        batch = (
            dataset["states"][idx],
            dataset["actions"][idx],
            dataset["rewards"][idx],
            dataset["next_states"][idx],
            dataset["dones"][idx],
        )
        
        # 更新
        info = agent.update(batch)
        losses.append(info['total_loss'])
        
        # 定期测试
        if step % test_every == 0:
            # 快速测试
            test_score = quick_test(agent, test_env, obs_dim)
            test_scores.append(test_score)
            
            avg_loss = np.mean(losses[-100:]) if len(losses) > 100 else np.mean(losses)
            
            print(f"Step {step:6d}/{steps} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Test: {test_score}")
            
            # 如果达到目标，提前停止
            if test_score >= 490:
                print(f"🎉 Early stopping at step {step} (score: {test_score})")
                break
    
    # 训练结束
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.1f} seconds")
    
    # 最终测试
    final_score = evaluate_final(agent, test_env, obs_dim, episodes=5)
    print(f"Final score: {np.mean(final_score):.1f} ± {np.std(final_score):.1f}")
    
    # 保存模型
    agent.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    test_env.close()
    return agent

def quick_test(agent, env, obs_dim, episodes=3):
    """快速测试"""
    scores = []
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 500:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action = agent.act(state_tensor.cpu().numpy(), evaluation_mode=True)
            
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            steps += 1
        
        scores.append(steps)
    
    return np.mean(scores)

def evaluate_final(agent, env, obs_dim, episodes=10):
    """最终评估"""
    scores = []
    for ep in range(episodes):
        state, _ = env.reset(seed=1000 + ep)
        done = False
        steps = 0
        
        while not done and steps < 500:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action = agent.act(state_tensor.cpu().numpy(), evaluation_mode=True)
            
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            steps += 1
        
        scores.append(steps)
        if steps == 500:
            print(f"  Episode {ep+1}: {steps} ✓")
        else:
            print(f"  Episode {ep+1}: {steps}")
    
    return scores

def train_with_curriculum():
    """使用课程学习策略"""
    print("Training with curriculum learning...")
    
    # 第一阶段：使用简单数据集
    print("\nPhase 1: Training with simple data...")
    simple_data = create_simple_dataset()
    
    # 创建临时数据集文件
    temp_path = "datasets/temp_simple.npz"
    np.savez_compressed(temp_path, **simple_data)
    
    # 第一阶段训练
    cfg1 = OfflineCQLConfig(
        lr=1e-3,
        batch_size=64,
        cql_alpha=0.5,  # 很低的保守性
        hidden_dim=64,
        steps=10_000,
    )
    
    # 第二阶段：使用更复杂的数据
    print("\nPhase 2: Training with expert data...")
    if os.path.exists("datasets/cartpole_high_quality.npz"):
        # 加载高质量数据
        data = np.load("datasets/cartpole_high_quality.npz")
        expert_data = {k: data[k] for k in data.files}
        
        # 第二阶段训练
        cfg2 = OfflineCQLConfig(
            lr=5e-4,
            batch_size=128,
            cql_alpha=1.0,
            hidden_dim=128,
            steps=20_000,
        )
        
        # 合并数据
        combined_data = {}
        for key in simple_data:
            combined_data[key] = np.concatenate([simple_data[key], expert_data[key]])
        
        temp_path2 = "datasets/temp_combined.npz"
        np.savez_compressed(temp_path2, **combined_data)
        
        return fast_train_cql(temp_path2, steps=30_000)
    
    return fast_train_cql(temp_path, steps=20_000)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fast", "curriculum", "eval"], default="fast")
    parser.add_argument("--dataset", type=str, default="datasets/cartpole_high_quality.npz")
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    
    args = parser.parse_args()
    
    if args.mode == "fast":
        agent = fast_train_cql(args.dataset, steps=50_000)
    
    elif args.mode == "curriculum":
        agent = train_with_curriculum()
    
    elif args.mode == "eval":
        if not os.path.exists(args.model):
            print(f"Model not found: {args.model}")
            print("Please train a model first with --mode fast")
        else:
            # 加载模型评估
            env = gym.make(ENV_NAME)
            obs_dim = env.observation_space.shape[0]
            
            # 加载agent
            cfg = OfflineCQLConfig()
            agent = OfflineCQL(obs_dim, 2, cfg)
            agent.load(args.model)
            
            scores = evaluate_final(agent, env, obs_dim, episodes=20)
            avg_score = np.mean(scores)
            
            print(f"\nEvaluation over 20 episodes:")
            print(f"Average score: {avg_score:.1f}")
            print(f"Max score: {np.max(scores)}")
            print(f"Success rate: {np.sum(np.array(scores) == 500) / len(scores) * 100:.1f}%")
            
            env.close()