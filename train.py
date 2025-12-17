"""
CartPole Training & Evaluation (PyTorch + Gymnasium)
---------------------------------------------------
Unified training/evaluation for DQN, PPO (and future algorithms).
- Saves models to standardized paths: `cartpole_{algo}.torch`
- Evaluation auto-detects algorithm from filename if not specified.
"""

from __future__ import annotations
import os
import time
import numpy as np
import gymnasium as gym
import torch

from agents.cartpole_dqn import DQNSolver, DQNConfig
from agents.cartpole_ppo import PPOSolver, PPOConfig
from agents.cartpole_actorcritic import ActorCriticConfig, ActorCriticNet, ActorCriticSolver
from scores.score_logger import ScoreLogger
from agents.cartpole_cql import CQLSolver,CQLConfig

ENV_NAME = "CartPole-v1"
MODEL_DIR = "models"


# ----------------------------
# Agent Factory & Path Utils
# ----------------------------


def get_run_id(algorithm: str, cfg) -> str:
    """
    根据算法名和关键参数生成唯一 ID。
    这能让你在 Report 中直接引用不同参数的对比图。
    """
    algo = algorithm.lower()
    if algo == "dqn":
        return f"DQN_lr{cfg.lr}_gamma{cfg.gamma}_eps{cfg.eps_decay}"
    elif algo == "ppo":
        return f"PPO_plr{cfg.policy_lr}_vlr{cfg.value_lr}_clip{cfg.clip_epsilon}"
    elif algo == "actorcritic":
        return f"A2C_lr{cfg.lr}_gamma{cfg.gamma}"
    elif algo == "cql":
        return f"CQL_alpha{cfg.alpha}_lr{cfg.lr}"
    return f"{algo}_run"

def create_agent(algorithm: str, obs_dim: int, act_dim: int):
    """Create agent by algorithm name."""
    algorithm = algorithm.lower()
    if algorithm == "dqn":
        return DQNSolver(obs_dim, act_dim, cfg=DQNConfig())
    elif algorithm == "ppo":
        return PPOSolver(obs_dim, act_dim, cfg=PPOConfig())
    elif algorithm == "actorcritic":
        return ActorCriticSolver(obs_dim, act_dim, cfg=ActorCriticConfig()) 
    elif algorithm == "cql":  
        return CQLSolver(obs_dim, act_dim, cfg=CQLConfig())
    else:
        raise ValueError(f"Unsupported algorithm: '{algorithm}'. Choose from ['dqn', 'ppo','actorcritic','cql'].")


def get_model_path(algorithm: str) -> str:
    """Return standardized model save path."""
    algorithm = algorithm.lower()
    if algorithm in ["dqn", "ppo","actorcritic"]:
        return os.path.join(MODEL_DIR, f"cartpole_{algorithm}.torch")
    else:
        return os.path.join(MODEL_DIR, f"cartpole_{algorithm}.torch")

# ----------------------------
# Agent Factory & Path Utils
# ----------------------------
def create_agent_with_config(algorithm: str, obs_dim: int, act_dim: int):
    """创建 agent 及其对应的 config。"""
    algorithm = algorithm.lower()
    if algorithm == "dqn":
        cfg = DQNConfig()
        return DQNSolver(obs_dim, act_dim, cfg=cfg), cfg
    elif algorithm == "ppo":
        cfg = PPOConfig()
        return PPOSolver(obs_dim, act_dim, cfg=cfg), cfg
    elif algorithm == "actorcritic":
        cfg = ActorCriticConfig()
        return ActorCriticSolver(obs_dim, act_dim, cfg=cfg), cfg
    elif algorithm == "cql":  
        cfg = CQLConfig()
        return CQLSolver(obs_dim, act_dim, cfg=cfg), cfg
    else:
        raise ValueError(f"Unsupported algorithm: '{algorithm}'")

# ----------------------------
# Unified Training Function
# ----------------------------
def train(
    algorithm: str = "dqn",
    num_episodes: int = 500,
    terminal_penalty: bool = True,
    render: bool = False,
    seed_offset: int = 0,
):
    os.makedirs(MODEL_DIR, exist_ok=True)
    render_mode = "human" if render else None
    env = gym.make(ENV_NAME, render_mode=render_mode)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # 1. 创建 Agent 和获取 Config
    agent, cfg = create_agent_with_config(algorithm, obs_dim, act_dim)
    
    # 2. 生成唯一的 run_id
    run_id = get_run_id(algorithm, cfg)
    model_path = os.path.join(MODEL_DIR, f"{run_id}.torch")

    print(f"\n[Train] ID: {run_id}")
    print(f"[Train] Model Path: {model_path}")

    # 3. 初始化 ScoreLogger (注意：你需要配套修改 ScoreLogger 的 __init__ 以接收 log_name)
    # 如果你还没改 ScoreLogger，请看下方的修改建议
    logger = ScoreLogger(ENV_NAME, log_name=run_id) 

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset(seed=seed_offset + ep)
        state = np.reshape(state, (1, obs_dim))
        done = False
        steps = 0

        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if terminal_penalty and done:
                reward = -1.0

            next_state = np.reshape(next_state, (1, obs_dim))
            steps += 1

            agent.step(state, action, reward, next_state, done)
            state = next_state

            if render:
                time.sleep(1 / 60)

        # 保存分数并自动更新带 ID 的图片和 CSV
        logger.add_score(steps, ep)

        # 打印日志
        if ep % 10 == 0:
            print(f"[{algorithm.upper()}] Ep {ep:3d}/{num_episodes} | Last Score: {steps}")

    env.close()
    agent.save(model_path)
    print(f"[Train] ✅ Saved to: {model_path}")
    return model_path
# ----------------------------
# Evaluation Function
# ----------------------------
def evaluate(
    model_path: str | None = None,
    algorithm: str | None = None,
    episodes: int = 5,
    render: bool = True,
    fps: int = 60,
):
    """
    Evaluate a trained model.

    If `model_path` is None: picks first .torch file in models/.
    If `algorithm` is None: infers from filename (e.g., '*ppo*' → 'ppo').
    """
    # Resolve model path
    if model_path is None:
        candidates = [f for f in os.listdir(MODEL_DIR) if f.endswith(".torch")]
        if not candidates:
            raise FileNotFoundError(f"No .torch model found in '{MODEL_DIR}'. Please train first.")
        model_path = os.path.join(MODEL_DIR, candidates[0])
        print(f"[Eval] Auto-selected model: {model_path}")
    else:
        print(f"[Eval] Using provided model: {model_path}")

    # Infer algorithm from filename if not given
    if algorithm is None:
        basename = os.path.basename(model_path).lower()
        if "ppo" in basename:
            algorithm = "ppo"
        elif "dqn" in basename:
            algorithm = "dqn"
        elif "actorcritic" in basename:
            algorithm = "actorcritic"
         
        elif "cql" in basename: 
            algorithm = "cql"
        else:
            raise ValueError(
                f"Cannot auto-detect algorithm from filename '{basename}'. "
                f"Please specify `algorithm=` explicitly (e.g., algorithm='ppo')."
            )
    print(f"[Eval] Algorithm: {algorithm.upper()}")

    # Create environment
    render_mode = "human" if render else None
    env = gym.make(ENV_NAME, render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Create agent
    if algorithm.lower() == "dqn":
        agent = DQNSolver(obs_dim, act_dim, cfg=DQNConfig())
    elif algorithm.lower() == "ppo":
        agent = PPOSolver(obs_dim, act_dim, cfg=PPOConfig())
    elif algorithm.lower() == "actorcritic":
        agent = ActorCriticSolver(obs_dim, act_dim, cfg=ActorCriticConfig())
    elif algorithm.lower() == "cql":  
        agent = CQLSolver(obs_dim, act_dim, cfg=CQLConfig())
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Load model
    agent.load(model_path)
    print(f"[Eval] ✅ Loaded {algorithm.upper()} model from: {model_path}")

    # Run evaluation
    scores = []
    dt = (1.0 / fps) if render and fps else 0.0

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=10_000 + ep)
        state = np.reshape(state, (1, obs_dim))
        done = False
        steps = 0

        while not done:
            action = agent.act(state, evaluation_mode=True)
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = np.reshape(next_state, (1, obs_dim))
            steps += 1

            if dt > 0:
                time.sleep(dt)

        scores.append(steps)
        print(f"[Eval] Episode {ep:2d}: steps = {steps}")

    env.close()
    avg = np.mean(scores) if scores else 0.0
    print(f"\n[Eval] 📊 Average over {episodes} episodes: {avg:.2f}")
    return scores


# ----------------------------
# Convenience Aliases (Optional)
# ----------------------------
def train_dqn(**kwargs):
    return train(algorithm="dqn", **kwargs)

def train_ppo(**kwargs):
    return train(algorithm="ppo", **kwargs)

def train_actorcritic(**kwargs):
    return train(algorithm="actorcritic", **kwargs)

def train_cql(**kwargs):
    return train(algorithm="cql", **kwargs)
# ----------------------------
# Main Entry Point
# ----------------------------
if __name__ == "__main__":
    # 🔁 Example workflows:

    # ✅ Train DQN
    # train(algorithm="dqn", num_episodes=500)

    # ✅ Train PPO
    # train(algorithm="ppo", num_episodes=500)
    
    # ✅ Train Actor-Critic
    # train(algorithm="actorcritic", num_episodes=500)
    
    # ✅ Train cql
    # train(algorithm="cql", num_episodes=500)

    # ✅ Evaluate the latest model (auto-detects algo)
    # evaluate(episodes=100, render=False)

    # ✅ Or evaluate specific model & algo:
    evaluate(model_path="models/DQN_lr0.0005_gamma0.99_eps0.99.torch", algorithm="dqn", episodes=100, render=True)