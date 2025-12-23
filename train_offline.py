"""
Offline CQL Training & Evaluation (PyTorch + Gymnasium)
------------------------------------------------------
- Style-aligned with train.py (DQN / PPO / A2C)
- Offline dataset only (NO env.step during training)
- Periodic evaluation & plotting with ScoreLogger
"""

from __future__ import annotations
import os
import time
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from agents.cartpole_offline_cql import OfflineCQL, CQLConfig
from scores.score_logger import ScoreLogger  # 新增

# ----------------------------
# Global Config
# ----------------------------
ENV_NAME = "CartPole-v1"
MODEL_DIR = "models"
DATASET_PATH = "datasets/cartpole_offline.npz"

NUM_EPOCHS = 500              # ≈ num_episodes in online RL
UPDATES_PER_EPOCH = 1000      # gradient steps per epoch
EVAL_EPISODES = 5             # evaluation episodes per epoch


# ----------------------------
# Utils
# ----------------------------
def get_run_id(cfg: CQLConfig) -> str:
    """Generate run id consistent with train.py style."""
    return f"CQL_alpha{cfg.alpha}_lr{cfg.lr}"


def load_dataset(path):
    data = np.load(path)
    return (
        data["states"],
        data["actions"],
        data["rewards"],
        data["next_states"],
        data["dones"],
    )


# ----------------------------
# Evaluation (aligned style)
# ----------------------------
def evaluate(env, agent, episodes=5, render=False, fps=60):
    scores = []
    dt = 1.0 / fps if render else 0.0

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=10_000 + ep)
        done = False
        steps = 0

        while not done:
            action = agent.act(state)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

            if render:
                time.sleep(dt)

        scores.append(steps)

    return np.mean(scores), scores


# ----------------------------
# Training Function
# ----------------------------
def train(
    num_epochs: int = NUM_EPOCHS,
    updates_per_epoch: int = UPDATES_PER_EPOCH,
    render: bool = False,
):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Environment (ONLY for evaluation)
    render_mode = "human" if render else None
    env = gym.make(ENV_NAME, render_mode=render_mode)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Create agent & config
    cfg = CQLConfig(alpha=1.0)
    agent = OfflineCQL(obs_dim, act_dim, cfg)

    run_id = get_run_id(cfg)
    model_path = os.path.join(MODEL_DIR, f"{run_id}.torch")

    print(f"\n[Train] ID: {run_id}")
    print(f"[Train] Model Path: {model_path}")

    # 初始化 ScoreLogger
    logger = ScoreLogger(ENV_NAME, log_name=run_id)

    # Load offline dataset
    dataset = load_dataset(DATASET_PATH)
    N = len(dataset[0])

    # ----------------------------
    # Training Loop
    # ----------------------------
    for epoch in range(1, num_epochs + 1):
        # 离线训练步骤
        for _ in range(updates_per_epoch):
            idx = np.random.randint(0, N, size=cfg.batch_size)
            batch = tuple(d[idx] for d in dataset)
            agent.update(batch)

        # 评估（与 train.py 风格对齐）
        avg_score, _ = evaluate(env, agent, episodes=EVAL_EPISODES)
        
        # 使用 ScoreLogger 记录分数
        logger.add_score(avg_score, epoch)

        # 打印日志（与 train.py 风格对齐）
        if epoch % 10 == 0:
            print(f"[CQL] Epoch {epoch:3d}/{num_epochs} | Eval Avg Score: {avg_score:.1f}")

        # 可选：提前停止
        if avg_score >= 480:
            print("🎉 Reached near-optimal performance, early stopping.")
            break

    # 保存模型
    agent.save(model_path)
    print(f"[Train] ✅ Saved to: {model_path}")

    env.close()
    
    # 注意：ScoreLogger 已经自动生成了图表，所以不需要额外绘图
    print(f"📊 Training curve saved to: {logger.png_path}")
    
    return model_path


# ----------------------------
# Evaluation Function (复用 train.py 中的 evaluate)
# ----------------------------
def evaluate_offline(
    model_path: str | None = None,
    episodes: int = 5,
    render: bool = True,
    fps: int = 60,
):
    """
    Evaluate a trained offline CQL model.
    """
    # 如果未指定模型路径，则查找 CQL 模型
    if model_path is None:
        candidates = [f for f in os.listdir(MODEL_DIR) if f.endswith(".torch") and "CQL" in f]
        if not candidates:
            raise FileNotFoundError(f"No CQL .torch model found in '{MODEL_DIR}'. Please train first.")
        model_path = os.path.join(MODEL_DIR, candidates[0])
        print(f"[Eval] Auto-selected model: {model_path}")
    
    run_id_from_path = os.path.basename(model_path).replace(".torch", "")
    
    print(f"[Eval] Algorithm: OFFLINE-CQL")
    print(f"[Eval] Model: {run_id_from_path}")

    # 创建环境
    render_mode = "human" if render else None
    env = gym.make(ENV_NAME, render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # 创建 agent
    cfg = CQLConfig(alpha=1.0)
    agent = OfflineCQL(obs_dim, act_dim, cfg)

    # 加载模型
    agent.load(model_path)
    print(f"[Eval] ✅ Loaded OFFLINE-CQL model from: {model_path}")

    # 运行评估
    scores = []
    dt = (1.0 / fps) if render and fps else 0.0

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=10_000 + ep)
        done = False
        steps = 0

        while not done:
            action = agent.act(state)
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            steps += 1

            if dt > 0:
                time.sleep(dt)

        scores.append(steps)
        if ep % 10 == 0 or episodes <= 10:
            print(f"[Eval] Episode {ep:2d}: steps = {steps}")

    env.close()
    avg = np.mean(scores) if scores else 0.0
    print(f"\n[Eval] 📊 Average over {episodes} episodes: {avg:.2f}")

    # 可视化评估结果（与 train.py 风格对齐）
    if not os.path.exists("results"):
        os.makedirs("results")
    
    plt.figure(figsize=(10, 5))
    plt.plot(scores, label='Score per Episode', color='skyblue', alpha=0.7)
    
    # 绘制移动平均线
    if len(scores) >= 10:
        ma = np.convolve(scores, np.ones(10)/10, mode='valid')
        plt.plot(range(9, len(scores)), ma, label='Moving Average (10)', color='red', linewidth=2)
    
    plt.axhline(y=475, color='green', linestyle='--', label='Success Threshold (475)')
    plt.title(f"Offline CQL Evaluation: {run_id_from_path}\nAvg: {avg:.2f}")
    plt.xlabel("Episode Index")
    plt.ylabel("Total Steps")
    plt.ylim(0, 520)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = f"results/eval_offline_{run_id_from_path}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Evaluation plot saved to: {save_path}")

    return scores


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # 训练离线 CQL
    # train()
    
    # 评估离线 CQL
    evaluate_offline(episodes=100, render=False)
